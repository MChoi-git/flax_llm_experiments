import einops
import jax
from jax import numpy as jnp, random


def log_softmax(logits, axis=-1):
    log_softmax = logits - jax.scipy.special.logsumexp(
        logits, axis=axis, keepdims=True)
    return log_softmax


def stable_cross_entropy_loss_with_logits(logits, targets, z_loss=0):
    log_sum_exp = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - log_sum_exp
    loss = -jnp.sum(targets * log_softmax, axis=-1)

    log_z = jnp.squeeze(log_sum_exp, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss

    return loss
