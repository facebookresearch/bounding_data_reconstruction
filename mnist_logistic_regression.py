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
import numpy as np
import os
import matplotlib.pyplot as plt

import sys
sys.path.append("fisher_information_loss")
import models
import dataloading

parser = argparse.ArgumentParser(description="MNIST training with FIL.")
parser.add_argument("--data_folder", default="data/", type=str,
    help="folder in which to store data")
parser.add_argument("--num_trials", default=10, type=int,
    help="number of repeated trials")
parser.add_argument("--lam", default=0.01, type=float,
    help="l2 regularization parameter")
parser.add_argument("--sigma", default=0.01, type=float,
    help="Gaussian noise multiplier")
args = parser.parse_args()

train_data = dataloading.load_dataset(
    name="mnist", split="train", normalize=False,
    num_classes=2, root=args.data_folder, regression=False)
test_data = dataloading.load_dataset(
    name="mnist", split="test", normalize=False,
    num_classes=2, root=args.data_folder, regression=False)
n = len(train_data["targets"])

all_etas, all_epsilons, all_rdp_epsilons = [], [], []

for i in range(args.num_trials):
    
    model = models.get_model("logistic")
    model.train(train_data, l2=args.lam, weights=None)
    # Renyi-DP accounting
    rdp_eps = 4 / (n * args.lam * args.sigma) ** 2
    # FIL accounting
    J = model.influence_jacobian(train_data)[:, :, :-1] / args.sigma
    etas = J.pow(2).sum(1).mean(1).sqrt()
    print(f"Trial {i+1:d}: RDP epsilon = {rdp_eps:.4f}, Max FIL eta = {etas.max():.4f}")
    model.theta = model.theta + args.sigma * torch.randn_like(model.theta)
        
    all_etas.append(etas.detach().numpy())
    all_rdp_epsilons.append(rdp_eps)

    predictions = model.predict(train_data["features"])
    acc = ((predictions == train_data["targets"]).float()).mean()
    print(f"Training accuracy of classifier {acc.item():.3f}")
    
    predictions = model.predict(test_data["features"])
    acc = ((predictions == test_data["targets"]).float()).mean()
    print(f"Test accuracy of classifier {acc.item():.3f}")
    
all_etas = np.stack(all_etas, 0)
all_rdp_epsilons = np.stack(all_rdp_epsilons, 0)
        
fil_bound = 1 / np.power(all_etas, 2).mean(0)
rdp_bound = 0.25 / (math.exp(all_rdp_epsilons.mean()) - 1)

plt.figure(figsize=(8,5))
_ = plt.hist(np.log10(fil_bound), bins=100, label='dFIL bound', color='silver', edgecolor='black', linewidth=0.3)
plt.axvline(x=np.log10(rdp_bound), color='k', linestyle='--', label='RDP bound')
plt.axvline(x=0, color='k', linestyle=':')
plt.xlabel('MSE lower bound', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(np.arange(-1, 11, 2), labels=['$10^{%d}$' % t for t in np.arange(-1, 11, 2)], fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper left', fontsize=20)
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/mnist_linear_hist.pdf", bbox_inches="tight")
