#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging

import jax
import jax.numpy as jnp
import jax.random as jnr
import hydra

from jax import grad
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten

import math
import accountant
import datasets
import trainer
import utils
import time


def batch_predict(predict, params, images, batch_size):
    num_images = images.shape[0]
    num_batches = int(math.ceil(float(num_images) / float(batch_size)))
    predictions = []
    for i in range(num_batches):
        lower = i * batch_size
        upper = min((i+1) * batch_size, num_images)
        predictions.append(predict(params, images[lower:upper]))
    return jnp.concatenate(predictions)


@hydra.main(config_path="configs", config_name="mnist")
def main(cfg):

    # set up random number generator:
    logging.info(f"Running using JAX {jax.__version__}...")
    rng = jnr.PRNGKey(int(time.time()))

    # create dataloader for MNIST dataset:
    if cfg.dataset.startswith("cifar"):
        num_channels = 3
        image_size = 32
        train_images, train_labels, test_images, test_labels = datasets.cifar(
            dataset=cfg.dataset, binary=cfg.binary, pca_dims=cfg.pca_dims)
    else:
        num_channels = 1
        image_size = 28
        train_images, train_labels, test_images, test_labels = datasets.mnist(
            dataset=cfg.dataset, binary=cfg.binary, pca_dims=cfg.pca_dims)
    logging.info(f"Training set max variance: %.4f" % train_images.var(0).max())
    
    num_samples, d = train_images.shape
    num_labels = train_labels.shape[1]
    if num_labels == 2:
        num_labels = 1
    if cfg.model.startswith("cnn"):
        assert cfg.pca_dims == 0, f"Cannot use PCA with {cfg.model} model."
        image_shape = (-1, image_size, image_size, num_channels)
        train_images = jnp.reshape(train_images, image_shape)
        test_images = jnp.reshape(test_images, image_shape)
    data_stream, num_batches = datasets.get_datastream(
        train_images, train_labels, cfg.batch_size
    )
    batches = data_stream()

    # set up model:
    if cfg.model.startswith("cnn"):
        input_shape = (-1, image_size, image_size, num_channels)
    else:
        input_shape = (-1, d)
    init_params, predict = utils.get_model(rng, cfg.model, input_shape, num_labels)
    num_params = sum(p.size for p in tree_flatten(init_params)[0])

    # create optimizer:
    if cfg.optimizer == "sgd":
        opt_init, opt_update, get_params = optimizers.momentum(
            cfg.step_size, cfg.momentum_mass
        )
    elif cfg.optimizer == "adam":
        opt_init, opt_update, get_params = optimizers.adam(cfg.step_size)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    opt_state = opt_init(init_params)

    # get loss function and update functions:
    loss = trainer.get_loss_func(predict)
    grad_func = trainer.get_grad_func(loss, norm_clip=cfg.norm_clip, soft_clip=True)
    update = trainer.get_update_func(
        get_params, grad_func, opt_update, norm_clip=cfg.norm_clip,
        reshape=cfg.model.startswith("cnn")
    )

    # get function that computes the Jacobian norms for privacy accounting:
    gelu_approx = 1.115
    fil_accountant = accountant.get_grad_jacobian_trace_func(
        grad_func, get_params, reshape=cfg.model.startswith("cnn"),
        label_privacy=cfg.label_privacy
    )
    dp_accountant = accountant.get_dp_accounting_func(cfg.batch_size, cfg.sigma / gelu_approx)
    
    # compute subsampling factor
    if cfg.sigma > 0:
        eps = math.sqrt(2 * math.log(1.25 / cfg.delta)) * 2 * gelu_approx / cfg.sigma
        q = float(cfg.batch_size) / num_samples
        subsampling_factor = q / (q + (1-q) * math.exp(-eps))
    else:
        subsampling_factor = 0
    logging.info(f"Subsampling factor is {subsampling_factor:.4f}")

    # train the model:
    logging.info(f"Training {cfg.model} model with {num_params} parameters using {cfg.optimizer}...")
    etas_squared = jnp.zeros((cfg.num_epochs, train_images.shape[0]))
    epsilons = jnp.zeros(cfg.num_epochs)
    rdp_epsilons = jnp.zeros(cfg.num_epochs)
    train_accs = jnp.zeros(cfg.num_epochs)
    test_accs = jnp.zeros(cfg.num_epochs)
    num_iters = 0
    for epoch in range(cfg.num_epochs):

        # perform full training sweep through the data:
        itercount = itertools.count()
        if epoch > 0:
            etas_squared = etas_squared.at[epoch].set(etas_squared[epoch-1])

        for batch_counter in range(num_batches):

            # get next batch:
            num_iters += 1
            i = next(itercount)
            rng = jnr.fold_in(rng, i)
            images, labels, batch_idx = next(batches)
            batch = (images, labels)

            # update privacy loss:
            if cfg.sigma > 0 and cfg.do_accounting:
                etas_batch = fil_accountant(rng, opt_state, batch) / cfg.sigma / cfg.norm_clip
                etas_squared = etas_squared.at[epoch, batch_idx].add(
                    subsampling_factor * jnp.power(etas_batch, 2), unique_indices=True
                )

            # perform private parameter update:
            opt_state = update(i, rng, opt_state, batch, cfg.sigma, cfg.weight_decay)
            

        # measure training and test accuracy, and average privacy loss:
        params = get_params(opt_state)
        spectral_norm = utils.estimate_spectral_norm(lambda x: predict(params, x), input_shape)
        train_predictions = batch_predict(predict, params, train_images, cfg.batch_size)
        test_predictions = batch_predict(predict, params, test_images, cfg.batch_size)
        train_accuracy = utils.accuracy(train_predictions, train_labels)
        test_accuracy = utils.accuracy(test_predictions, test_labels)
        train_accs = train_accs.at[epoch].set(train_accuracy)
        test_accs = test_accs.at[epoch].set(test_accuracy)
        params, _ = tree_flatten(params)
        params_norm = math.sqrt(sum([jnp.power(p, 2).sum() for p in params]))
        if cfg.sigma > 0 and cfg.do_accounting:
            median_eta = jnp.median(jnp.sqrt(etas_squared[epoch]))
            max_eta = jnp.sqrt(etas_squared[epoch]).max()
            delta = 1e-5
            epsilon = dp_accountant(num_iters, len(train_labels), delta)
            epsilons = epsilons.at[epoch].set(epsilon)
            rdp_epsilon = dp_accountant(num_iters, len(train_labels), delta, alpha=2)
            rdp_epsilons = rdp_epsilons.at[epoch].set(rdp_epsilon)

        # print out progress:
        logging.info(f"Epoch {epoch + 1}:")
        logging.info(f" -> training accuracy = {train_accuracy:.4f}")
        logging.info(f" -> test accuracy = {test_accuracy:.4f}")
        logging.info(f" -> parameter norm = {params_norm:.4f}, spectral norm = {spectral_norm:.4f}")
        if cfg.sigma > 0 and cfg.do_accounting:
            logging.info(f" -> Median FIL privacy loss = {median_eta:.4f}")
            logging.info(f" -> Max FIL privacy loss = {max_eta:.4f}")
            logging.info(f" -> DP privacy loss = ({epsilon:.4f}, {delta:.2e})")
            logging.info(f" -> 2-RDP privacy loss = {rdp_epsilon:.4f}")

    etas = jnp.sqrt(etas_squared) if cfg.sigma > 0 and cfg.do_accounting else float("inf")

    return etas, epsilons, rdp_epsilons, train_accs, test_accs


# run all the things:
if __name__ == "__main__":
    main()
