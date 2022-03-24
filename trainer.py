#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import jax.random as jnr
from jax import jit, grad, vmap, nn
from jax.tree_util import tree_flatten, tree_unflatten
import math


def get_loss_func(predict):
    """
    Returns the loss function for the specified `predict`ion function.
    """

    @jit
    def loss(params, inputs, targets):
        """
        Multi-class loss entropy loss function for model with parameters `params`
        and the specified `inputs` and one-hot `targets`.
        """
        predictions = nn.log_softmax(predict(params, inputs))
        if predictions.ndim == 1:
            return -jnp.sum(predictions * targets)
        return -jnp.mean(jnp.sum(predictions * targets, axis=-1))

    return loss


def get_grad_func(loss, norm_clip=0, soft_clip=False):
    
    @jit
    def clipped_grad(params, inputs, targets):
        grads = grad(loss)(params, inputs, targets)
        if norm_clip == 0:
            return grads
        else:
            nonempty_grads, tree_def = tree_flatten(grads)
            total_grad_norm = jnp.add(jnp.linalg.norm(
                [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]), 1e-7)
            if soft_clip:
                divisor = nn.gelu(total_grad_norm / norm_clip - 1) + 1
            else:
                divisor = jnp.maximum(total_grad_norm / norm_clip, 1.)
            normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
            return tree_unflatten(tree_def, normalized_nonempty_grads)
    
    return clipped_grad


def get_update_func(get_params, grad_func, opt_update, norm_clip=0, reshape=True):
    """
    Returns the parameter update function for the specified `predict`ion function.
    """

    @jit
    def update(i, rng, opt_state, batch, sigma, weight_decay):
        """
        Function that performs `i`-th model update using the specified `batch` on
        optimizer state `opt_state`. Updates are privatized by noise addition
        with variance `sigma`.
        """

        # compute parameter gradient:
        inputs, targets = batch
        if reshape:
            inputs = jnp.expand_dims(inputs, 1)
        params = get_params(opt_state)
        multiplier = 1 if norm_clip == 0 else norm_clip

        # add noise to gradients:
        grads = vmap(grad_func, in_axes=(None, 0, 0))(params, inputs, targets)
        grads_flat, grads_treedef = tree_flatten(grads)
        grads_flat = [g.sum(0) for g in grads_flat]
        rngs = jnr.split(rng, len(grads_flat))
        noisy_grads = [
            (g + multiplier * sigma * jnr.normal(r, g.shape)) / len(targets)
            for r, g in zip(rngs, grads_flat)
        ]
        
        # weight decay
        params_flat, _ = tree_flatten(params)
        noisy_grads = [
            g + weight_decay * param
            for g, param in zip(noisy_grads, params_flat)
        ]
        noisy_grads = tree_unflatten(grads_treedef, noisy_grads)

        # perform parameter update:
        return opt_update(i, noisy_grads, opt_state)

    return update