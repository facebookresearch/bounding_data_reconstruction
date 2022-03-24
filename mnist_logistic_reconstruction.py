#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import torch
import random
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import sys
sys.path.append("fisher_information_loss")
import models
import dataloading

def recons_attack(model, X, y, lam, link_func):
    """
    Runs the Balle et al. GLM attack https://arxiv.org/abs/2201.04845.
    """
    def compute_grad(model, X, y):
        return ((X @ model.theta).sigmoid() - y)[:, None] * X
    n = len(y)
    grad = compute_grad(model, X, y)
    B1 = (grad.sum(0)[None, :] - grad)[:, 0]
    denom = B1 + n * lam * model.theta[0][None]
    X_hat = (grad.sum(0)[None, :] - grad + n * lam * model.theta[None, :]) / denom[:, None]
    y_hat = link_func(X_hat @ model.theta) + denom
    return X_hat, y_hat

def compute_correct_ratio(etas, num_bins, predictions, target):
    order = etas.argsort()
    bin_size = len(target) // num_bins + 1
    bin_accs = []
    for prediction in predictions:
        prediction = np.array(prediction)
        correct = (prediction == target)
        bin_accs.append([correct[order[lower:lower + bin_size]].mean()
                or lower in range(0, len(correct), bin_size)])
        return np.array(bin_accs)

parser = argparse.ArgumentParser(description="Evaluate GLM reconstruction attack.")
parser.add_argument("--data_folder", default="data/", type=str,
    help="folder in which to store data")
parser.add_argument("--num_trials", default=10000, type=int,
    help="Number of trials")
parser.add_argument("--lam", default=0.01, type=float,
    help="regularization parameter for logistic regression")
parser.add_argument("--sigma", default=1e-5, type=float,
    help="Gaussian noise parameter for output perturbation")
args = parser.parse_args()

train_data = dataloading.load_dataset(
    name="mnist", split="train", normalize=False,
    num_classes=2, root=args.data_folder, regression=False)
test_data = dataloading.load_dataset(
    name="mnist", split="test", normalize=False,
    num_classes=2, root=args.data_folder, regression=False)
train_data['features'] = torch.cat([torch.ones(len(train_data['targets']), 1), train_data['features']], 1)
test_data['features'] = torch.cat([torch.ones(len(test_data['targets']), 1), test_data['features']], 1)
    
model = models.get_model("logistic")
model.train(train_data, l2=args.lam, weights=None)
true_theta = model.theta.clone()

predictions = model.predict(train_data["features"])
acc = ((predictions == train_data["targets"]).float()).mean()
print(f"Training accuracy of classifier {acc.item():.3f}")

predictions = model.predict(test_data["features"])
acc = ((predictions == test_data["targets"]).float()).mean()
print(f"Test accuracy of classifier {acc.item():.3f}")

J = model.influence_jacobian(train_data)[:, :, 1:-1] / args.sigma
etas = J.pow(2).sum(1).mean(1)
    
X = train_data["features"]
y = train_data["targets"].float()
n, d = X.size(0), X.size(1) - 1
link_func = torch.sigmoid

X_means = torch.zeros(X.shape)
errors = torch.zeros(len(y))
with torch.no_grad():
    print('Running reconstruction attack for %d trials:' % args.num_trials)
    for i in tqdm(range(args.num_trials)):
        model.theta = true_theta + args.sigma * torch.randn(true_theta.size())
        X_hat, y_hat = recons_attack(model, X, y, args.lam, link_func)
        X_means += X_hat / args.num_trials
        errors += (X_hat[:, 1:] - X[:, 1:]).pow(2).sum(1) / (d * args.num_trials)
    X_means = X_means[:, 1:]
    
# filter out examples that the attack failed on
mask = torch.logical_not(torch.isnan(errors))
etas = etas[mask]
errors = errors[mask]
_, order = etas.reciprocal().sort()

# plot MSE lower bound vs. true MSE
plt.figure(figsize=(8,5))
below_bound = etas.reciprocal() < errors
plt.scatter(etas[below_bound].reciprocal().detach(), errors[below_bound].detach(), s=10)
plt.scatter(etas[torch.logical_not(below_bound)].reciprocal().detach(), errors[torch.logical_not(below_bound)].detach(),
            s=10, color='indianred')
plt.plot(np.power(10, np.arange(-5.5, 3, 0.1)), np.power(10, np.arange(-5.5, 3, 0.1)), 'k', label='Lower bound')
plt.axvline(x=1, color='k', linestyle=':')
plt.xticks(fontsize=20)
plt.xlim([1e-6, 1e4])
plt.xlabel('Predicted MSE', fontsize=20)
plt.xscale('log')
plt.yticks(fontsize=20)
plt.ylabel('Recons. attack MSE', fontsize=20)
plt.yscale('log')
plt.legend(loc='lower right', fontsize=20)
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/recons_mse.pdf", bbox_inches="tight")

# plot reconstructed samples
plt.figure(figsize=(48, 6))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(X[mask][order[i], 1:].clamp(0, 1).view(28, 28).detach())
    plt.axis('off')
plt.savefig("figs/orig_highest8.pdf", bbox_inches="tight")

plt.figure(figsize=(48, 6))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(X_means[mask][order[i]].clamp(0, 1).view(28, 28).detach())
    plt.axis('off')
plt.savefig("figs/recons_highest8.pdf", bbox_inches="tight")

plt.figure(figsize=(48, 6))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(X[mask][order[-i-1], 1:].clamp(0, 1).view(28, 28).detach())
    plt.axis('off')
plt.savefig("figs/orig_lowest8.pdf", bbox_inches="tight")

plt.figure(figsize=(48, 6))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(X_means[mask][order[-i-1]].clamp(0, 1).view(28, 28).detach())
    plt.axis('off')
plt.savefig("figs/recons_lowest8.pdf", bbox_inches="tight")
