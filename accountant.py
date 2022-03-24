#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import jax.random as jnr

from jax import jit, jvp, vjp, jacrev, vmap, nn
from jax.tree_util import tree_flatten
import trainer
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent


def get_grad_jacobian_norm_func(grad_func, get_params, method="jvp", reshape=True, label_privacy=False):
    """
    Returns a function that computes norm of the Jacobian of the parameter
    gradients for the specified `loss` function for an optimizer in which the
    `get_params` function returns the model parameters.
    """

    # assertions:
    assert method in ["jvp", "full"], f"Unknown method: {method}"

    @jit
    def compute_power_iteration_jvp(params, w, inputs, targets):
        """
        Computes a single power iteration via the JVP method. Does not include
        Jacobian w.r.t. targets.
        """

        # compute JVP of per-example parameter gradient Jacobian with w:
        if label_privacy:
            perex_grad = lambda x: vmap(grad_func, in_axes=(None, 0, 0))(
                params, inputs, x
            )
            _, w = jvp(perex_grad, (targets,), (w,))
        else:
            perex_grad = lambda x: vmap(grad_func, in_axes=(None, 0, 0))(
                params, x, targets
            )
            _, w = jvp(perex_grad, (inputs,), (w,))
            
        # compute norm of the JVP:
        w_flattened, _ = tree_flatten(w)
        norms = [
            jnp.power(jnp.reshape(v, (v.shape[0], -1)), 2).sum(axis=1)
            for v in w_flattened
        ]
        norms = jnp.sqrt(sum(norms) + 1e-7)

        # compute VJP of per-example parameter gradient Jacobian with w:
        if label_privacy:
            _, f_vjp = vjp(perex_grad, targets)
        else:
            _, f_vjp = vjp(perex_grad, inputs)
        w_out = f_vjp(w)[0]
            
        return norms, w_out

    @jit
    def compute_power_iteration_full(params, w, inputs, targets):
        """
        Computes a single power iteration by computing the full Jacobian and
        right-multiplying it. Does not include Jacobian w.r.t. targets.
        """

        # compute per-example parameter gradient Jacobian:
        J = jacrev(grad_func, 1)(params, inputs, targets)
        J_flattened, _ = tree_flatten(J)

        # compute JVP with w:
        jvp_exact = [(v * w).sum(-1) for v in J_flattened]

        # compute norm of the JVP:
        norms = [
            jnp.power(jnp.reshape(v, (-1, v.shape[-1])), 2).sum(axis=0)
            for v in jvp_exact
        ]
        norms = jnp.sqrt(sum(norms))

        # compute VJP of per-example parameter gradient Jacobian with w:
        vjp_exact = [
            J_flattened[i] * jnp.expand_dims(jvp_exact[i], -1)
            for i in jnp.arange(len(jvp_exact))
        ]
        w_out = sum(
            [jnp.reshape(v, (-1, v.shape[-2], v.shape[-1])).sum(0) for v in vjp_exact]
        )
        return norms, w_out

    @jit
    def grad_jacobian_norm(rng, opt_state, batch, num_iters=20):
        """
        Computes norm of the Jacobian of the parameter gradients. The function
        performs `num_iters` power iterations.
        """

        # initialize power iterates:
        inputs, targets = batch
        if reshape:
            inputs = jnp.expand_dims(inputs, 1)
        
        w = jnr.normal(rng, shape=(targets.shape if label_privacy else inputs.shape))
        w_norm = jnp.sqrt(jnp.power(w.reshape(w.shape[0], -1), 2).sum(axis=1) + 1e-7)
        w = w / jnp.expand_dims(w_norm, tuple(range(1, len(w.shape))))

        # perform power iterations:
        params = get_params(opt_state)
        for i in jnp.arange(num_iters):
            if method == "jvp":
                norms, w = compute_power_iteration_jvp(params, w, inputs, targets)
            elif method == "full":
                norms, w = compute_power_iteration_full(params, w, inputs, targets)
            w_norm = jnp.sqrt(jnp.power(w.reshape(w.shape[0], -1), 2).sum(axis=1) + 1e-7)
            w = w / jnp.expand_dims(w_norm, tuple(range(1, len(w.shape))))
        
        # set nan values to 0 because gradient is 0
        norms = jnp.nan_to_num(norms)
        return norms

    # return the function:
    return grad_jacobian_norm


def get_grad_jacobian_trace_func(grad_func, get_params, reshape=True, label_privacy=False):
    """
    Returns a function that computes the (square root of the) trace of the Jacobian
    of the parameters.
    """

    @jit
    def grad_jacobian_trace(rng, opt_state, batch, num_iters=50):

        params = get_params(opt_state)
        inputs, targets = batch
        if reshape:
            inputs = jnp.expand_dims(inputs, 1)
            
        if label_privacy:
            flattened_shape = jnp.reshape(targets, (targets.shape[0], -1)).shape
            perex_grad = lambda x: vmap(grad_func, in_axes=(None, 0, 0))(
                params, inputs, x
            )
        else:
            flattened_shape = jnp.reshape(inputs, (inputs.shape[0], -1)).shape
            perex_grad = lambda x: vmap(grad_func, in_axes=(None, 0, 0))(
                params, x, targets
            )
        
        num_iters = targets.shape[1] if label_privacy else num_iters
        rngs = jnr.split(rng, num_iters)
        trace = jnp.zeros(inputs.shape[0])
        for i, g in zip(jnp.arange(num_iters), rngs):
            indices = jnr.categorical(g, jnp.ones(shape=flattened_shape))
            if label_privacy:
                indices = i * jnp.ones(flattened_shape[0])
                w = jnp.reshape(nn.one_hot(indices, flattened_shape[1]), targets.shape)
                _, w = jvp(perex_grad, (targets,), (w,))
            else:
                indices = jnr.categorical(rng, jnp.ones(shape=flattened_shape))
                w = jnp.reshape(nn.one_hot(indices, flattened_shape[1]), inputs.shape)
                _, w = jvp(perex_grad, (inputs,), (w,))
            # compute norm of the JVP:
            w_flattened, _ = tree_flatten(w)
            norms = [
                jnp.power(jnp.reshape(v, (v.shape[0], -1)), 2).sum(axis=1)
                for v in w_flattened
            ]
            trace = trace + sum(norms) / num_iters
        
        # set nan values to 0 because gradient is 0
        trace = jnp.nan_to_num(trace)
        return jnp.sqrt(trace + 1e-7)

    # return the function:
    return grad_jacobian_trace


def get_dp_accounting_func(batch_size, sigma):
    """
    Returns the (eps, delta)-DP accountant if alpha=None,
    or the (alpha, eps)-RDP accountant otherwise.
    """
    
    def compute_epsilon(steps, num_examples, target_delta=1e-5, alpha=None):
        if num_examples * target_delta > 1.:
            warnings.warn('Your delta might be too high.')
        q = batch_size / float(num_examples)
        if alpha is None:
            orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
            rdp_const = compute_rdp(q, sigma, steps, orders)
            eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
        else:
            eps = compute_rdp(q, sigma, steps, alpha)
        return eps
    
    return compute_epsilon