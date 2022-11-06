from functools import partial
from typing import Any, Optional, Callable, Tuple, Union

import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
import flax.linen.partitioning as nn_partitioning

from partitioning import with_sharding_constraint


Dtype = Any


def make_causal_mask(seq_len: int, param_dtype: Dtype):
    mask = jnp.triu(
        jnp.zeros((seq_len, seq_len), dtype=param_dtype) - jnp.inf, k=1)
    return mask


def default_attn_norm(x: jnp.ndarray, hidden_dim: int):
    return x / ((2 * hidden_dim) ** -0.5)


def _normalize_axes(axes, ndim):
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


class Dense(nn.Module):
    features: Union[Tuple[int]]
    axis: Tuple[int] = (-1,)

    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Optional[Dtype] = jnp.float32

    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    named_axes: Optional[Tuple[str, ...]] = ()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):

        axis = _normalize_axes(self.axis, inputs.ndim)

        # Left axes to be (reduced and) multiplied
        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + self.features

        kernel = nn_partitioning.param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.param_dtype,
            axes=self.named_axes,
        )

        inputs, kernel = promote_dtype(inputs, kernel, dtype=self.dtype)

        contract_idx = tuple(range(0, len(axis)))
        return jax.lax.dot_general(
            inputs, kernel, ((axis, contract_idx), ((), ())))


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray],
    deterministic: bool,
    qkv_dropout: float,
    dropout_rng: jnp.ndarray,
    norm_fn: Callable,
):
    """
    Given query and key, calculate the attention weights for some mask and
    normalizer before the softmax. Mask could be attention bias, mask, dropout,
    or some combination.
    """
    qk = jnp.einsum("bshe,bShe->bhsS", query, key)

    attn_weights = jax.nn.softmax(norm_fn(qk + mask), axis=-1)

    # TODO: qkv dropout not implemented yet

    self_attn = jnp.einsum("bhsS,bShe->bshe", attn_weights, value)

    return self_attn


class MultiheadAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    qkv_dropout: float

    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Optional[Dtype] = jnp.float32

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    attn_norm_fn: Callable = default_attn_norm
    attn_mask_fn: Callable = make_causal_mask

    named_axes: Tuple[str, ...] = ("embed", "kv")

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray, train: bool):
        # inputs (batch_size, seq_len, embedding)
        B, S, E = inputs.shape
        assert E % self.num_heads == 0, (
            f"{self.num_heads} heads doesn't " f"split {E} dimension evenly"
        )

        # Promote params and input to operation dtype
        # NOTE: All ops will be done in self.dtype
        inputs = promote_dtype(inputs, dtype=self.dtype)[0]

        projection = partial(
            Dense,
            features=(self.num_heads, self.hidden_dim // self.num_heads),
            axis=(-1,),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            named_axes=("embed", "heads", "kv"),
        )

        # (..., embed) sharded (..., "mp")-wise can be cleanly resharded across
        # (..., heads, head_dim) (..., "mp", None)-wise. Reshape happens
        # without resharding overhead.
        q = projection(name="q_proj")(inputs)
        k = projection(name="k_proj")(inputs)
        v = projection(name="v_proj")(inputs)
        q = q.reshape(B, S, self.num_heads, -1)
        k = k.reshape(B, S, self.num_heads, -1)
        v = v.reshape(B, S, self.num_heads, -1)
        q = with_sharding_constraint(q, ("batch", "seq", "heads", "kv"))
        k = with_sharding_constraint(k, ("batch", "seq", "heads", "kv"))
        v = with_sharding_constraint(v, ("batch", "seq", "heads", "kv"))

        dropout_rng = (
            self.make_rng("dropout") if train and self.qkv_dropout < 0.0 else None
        )

        self_attn = dot_product_attention(
            q,
            k,
            v,
            mask,
            train,
            self.qkv_dropout,
            dropout_rng,
            partial(default_attn_norm, hidden_dim=self.hidden_dim),
        )
        self_attn = with_sharding_constraint(
            self_attn, ("batch", "seq", "heads", "kv"))

        out_projection = Dense(
            name="out_proj",
            features=(self.hidden_dim,),
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            named_axes=("heads", "kv", "embed"),
        )(self_attn)
        out_projection = with_sharding_constraint(
            out_projection, ("batch", "seq", "embed")
        )

        return out_projection


class LayerNorm(nn.Module):
    eps: float = 1e-6
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    scale_init: Callable = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = jnp.asarray(inputs, dtype=jnp.float32)
        features = x.shape[-1]
        mean = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * jax.lax.rsqrt(mean + self.eps), self.dtype)
        scale = nn_partitioning.param_with_axes(
            "scale",
            self.scale_init,
            (features,),
            self.param_dtype,
            axes=("embed",))
        scale = jnp.asarray(scale, self.dtype)
        return y * scale


class DecoderLayer(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_hidden_multiplier: int

    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float

    nonlin_fn: Callable = jax.nn.gelu

    param_dtype: Optional[Dtype] = jnp.float32
    dtype: Optional[Dtype] = jnp.float32

    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    ln_scale_init: Optional[Callable] = jax.nn.initializers.ones

    attn_norm_fn: Callable = default_attn_norm
    attn_mask_fn: Callable = make_causal_mask

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool):
        """
        Note: The f and g functions for tensor and sequence parallel are
        included at the last layer they are present in.
        """
        # (None, "mp", None)
        msa_residual = inputs
        # (None, "mp", None)

        # (None, "mp", None)
        inputs = with_sharding_constraint(inputs, ("batch", "seq", "embed"))
        ln1_out = LayerNorm(
            name="ln1",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=self.ln_scale_init,
        )(inputs)
        ln1_out = with_sharding_constraint(ln1_out, ("batch", "seq", "embed"))
        # (None, None, None)

        # TODO: Implement fn to get mask
        mask = jnp.zeros(1)

        # (None, None, None)
        multihead_out = MultiheadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_dropout=self.qkv_dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            named_axes=("embed", "kv"),
        )(ln1_out, mask, train)
        # (None, None, None)

        # (None, "mp", None)
        msa_dropout = nn.Dropout(rate=self.msa_dropout)(
            multihead_out, deterministic=not train
        )
        # (None, "mp", None)

        # (None, "mp", None)
        residual1_out = msa_dropout + msa_residual
        # (None, "mp", None)

        # (None, "mp", None)
        mlp_residual = residual1_out
        # (None, "mp", None)

        # (None, "mp", None)
        ln2_out = LayerNorm(
            name="ln2",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=self.ln_scale_init,
        )(residual1_out)
        ln2_out = with_sharding_constraint(ln2_out, ("batch", "seq", "embed"))
        # (None, None, None)

        # (None, None, None)
        mlp1_out = Dense(
            name="fc1",
            features=(self.hidden_dim * self.mlp_hidden_multiplier,),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            named_axes=("embed", "mlp"),
        )(ln2_out)
        mlp1_out = with_sharding_constraint(mlp1_out, ("batch", "seq", "mlp"))
        # (None, None, "mp")

        # (None, None, "mp")
        nonlin_out = jax.nn.gelu(mlp1_out)
        # (None, None, "mp")

        # (None, None, "mp")
        mlp2_out = Dense(
            name="fc2",
            features=(self.hidden_dim,),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            named_axes=("mlp", "embed"),
        )(nonlin_out)
        mlp2_out = with_sharding_constraint(
            mlp2_out, ("batch", "seq", "embed"))
        # (None, "mp", None)

        # (None, "mp", None)
        mlp_dropout = nn.Dropout(rate=self.mlp_dropout)(
            mlp2_out, deterministic=not train
        )
        # (None, "mp", None)

        # (None, "mp", None)
        layer_output = mlp_dropout + mlp_residual
        # (None, "mp", None)

        return layer_output
