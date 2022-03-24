#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from jax.experimental import stax

DTYPE_MAPPING = {
    "float32": "f32",
    "float64": "f64",
    "int32": "s32",
    "int64": "s64",
    "uint32": "u32",
    "uint64": "u64",
}


def _l2_normalize(x, eps=1e-7):
    return x * jax.lax.rsqrt((x ** 2).sum() + eps)


def estimate_spectral_norm(f, input_shape, seed=0, n_steps=20):
    input_shape = tuple([1] + [input_shape[i] for i in range(1, len(input_shape))])
    rng = jax.random.PRNGKey(seed)
    u0 = jax.random.normal(rng, input_shape)
    v0 = jnp.zeros_like(f(u0))
    def fun(carry, _):
        u, v = carry
        v, f_vjp = jax.vjp(f, u)
        v = _l2_normalize(v)
        u, = f_vjp(v)
        u = _l2_normalize(u)
        return (u, v), None
    (u, v), _ = jax.lax.scan(fun, (u0, v0), xs=None, length=n_steps)
    return jnp.vdot(v, f(u))


def accuracy(predictions, targets):
    """
    Compute accuracy of `predictions` given the associated `targets`.
    """
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(predictions, axis=-1)
    return jnp.mean(predicted_class == target_class)


def get_model(rng, model_name, input_shape, num_labels):
    """
    Returns model specified by `model_name`. Model is initialized using the
    specified random number generator `rng`.

    Optionally, the input image `height` and `width` can be specified as well.
    """

    # initialize convolutional network:
    if model_name == "cnn":
        init_random_params, predict = stax.serial(
            stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
            stax.Gelu,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Flatten,
            stax.Dense(32),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
        
    elif model_name == "cnn_tanh":
        init_random_params, predict = stax.serial(
            stax.Conv(16, (8, 8), padding="SAME", strides=(2, 2)),
            stax.Tanh,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Conv(32, (4, 4), padding="VALID", strides=(2, 2)),
            stax.Tanh,
            stax.AvgPool((2, 2), (1, 1)),
            stax.Flatten,
            stax.Dense(32),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
        
    elif model_name == "cnn_cifar":
        init_random_params, predict = stax.serial(
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(32, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(64, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.Conv(128, (3, 3), padding="SAME", strides=(1, 1)),
            stax.Tanh,
            stax.AvgPool((2, 2), (2, 2)),
            stax.Flatten,
            stax.Dense(128),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)

    # initialize multi-layer perceptron:
    elif model_name == "mlp":
        init_random_params, predict = stax.serial(
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(256),
            stax.Gelu,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
        
    elif model_name == "mlp_tanh":
        init_random_params, predict = stax.serial(
            stax.Dense(256),
            stax.Tanh,
            stax.Dense(256),
            stax.Tanh,
            stax.Dense(num_labels),
        )
        _, init_params = init_random_params(rng, input_shape)
    # initialize linear model:
    elif model_name == "linear":
        init_random_params, predict_raw = stax.Dense(num_labels)
        def predict(params, inputs):
            logits = predict_raw(params, inputs)
            return jnp.hstack([logits, jnp.zeros(logits.shape)])
        _, init_params = init_random_params(rng, input_shape)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # return initial model parameters and prediction function:
    return init_params, predict
