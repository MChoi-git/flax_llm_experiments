from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Callable, Tuple, Union

import einops
import numpy as np
import jax
from jax import numpy as jnp
import flax.linen as nn
import flax.linen.initializers as nn_initializers
from flax.linen.dtypes import promote_dtype
import flax.linen.partitioning as nn_partitioning

from partitioning import with_sharding_constraint


"""
NN transformer layer implementations. Some code taken from the T5x repo
(https://github.com/google-research/t5x/tree/1d3ebe5e8aed098d986694479ac864380273338a)
and flax (https://github.com/google/flax). This code is solely for pedagogical
purposes.
"""


Dtype = Any


def make_causal_mask(seq_len: int, param_dtype: Dtype):
    mask = jnp.triu(
        jnp.zeros((seq_len, seq_len), dtype=param_dtype) - jnp.inf, k=1)
    return mask


def default_attn_norm(x: jnp.ndarray, hidden_dim: int):
    return x / ((2 * hidden_dim) ** -0.5)


def _normalize_axes(axes, ndim):
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _fixed_pos_embed(x, seq_dim=1):
    """
    Sequence dimension alternates sin and cosine as well as their frequency in
    an increasing scheme. The hidden dimension is the wave function for the
    specific sinusoid function defined in the specific sequence position.
    """
    dim = x.shape[-1]

    # Hidden axis
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

    # Sequence axis
    sinusoid_inp = np.einsum("i,j->ij", np.arange(x.shape[seq_dim]), inv_freq)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


def _rotate_every_two(x):
    """
    Return the interleaved vector for the sin addition. Ie. Something like
    [-1, 0, -3, 2, -5, 4, ...], where the integers are the indices of the
    elements in the input vector.
    """
    # Separate evens and odds
    evens = x[:, ::2, :, :]
    odds = x[:, 1::2, :, :]

    # Stack and flatten
    x = jnp.stack((-odds, evens), axis=1)

    return einops.rearrange(x, "b d j ... -> b (d j) ...")


def apply_rotary_pos_embed(x: jnp.ndarray, seq_dim: int):
    """
    Applies RoPE to a tensor x. See:
        `https://blog.eleuther.ai/rotary-embeddings/` and
        `https://arxiv.org/pdf/2104.09864v2.pdf` (RoPE paper)
        for more details.
    Splits each d-dimensional hidden vector in a sequence into d/2 pairs. Each
    pair defines a vector in 2-D space. Each pair has an associated angle
    theta, defining a 2-D rotation. Theta is sourced from a schedule, which
    is a hyperparameter. Each pair also has a magnitude m, which is equivalent
    to the pair's position in the hidden vector, from [0, (d-1)/2]. This
    magnitude defines how much to scale its respective pair's theta parameter
    by. This pair-wise vector rotation (about the origin) is repeated for all
    d/2 pairs in the hidden vector, which is then done again for each position
    in the sequence.

    Example:
        For hidden vectors of length d:
        h1 = [1, 5, 23, 7, ...]
            -> rotate([1, 5], angle=theta1 * 1)
            -> rotate([23, 7], angle=theta2 * 1)
            -> ...
        h2 = [6, 3, 7, 2, ...]
            -> rotate([6, 3], angle=theta1 * 2)
            -> rotate([7, 2], angle=theta2 * 2)
            -> ...
        theta_schedule = {10000 ** (-2i/d)|i in [0, 1, 2, ..., (d-1)/2]}
        rotate = dot(2d_rotation_matrix, x)
    """
    sincos = _fixed_pos_embed(x, seq_dim=seq_dim)

    sin, cos = map(
        lambda t: einops.repeat(
            t, "b n -> b (n j)", j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (_rotate_every_two(x) * sin)


@dataclass
class TransformerConfig:
    num_layers: int = 16
    vocab_size: int = 50000

    hidden_dim: int = 8
    num_heads: int = 2
    mlp_hidden_multiplier: int = 4

    qkv_dropout: float = 0.1
    msa_dropout: float = 0.1
    mlp_dropout: float = 0.1

    nonlin_fn: Callable = jax.nn.gelu

    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    embedding_init: Callable = nn_initializers.variance_scaling(
        1.0, "fan_in", "normal", out_axis=0)
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros
    ln_scale_init: Optional[Callable] = jax.nn.initializers.ones
    attn_norm_fn: Callable = default_attn_norm

    use_scan: bool = True
    use_shared_vocab_embed: bool = True


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

        # TODO: Additional sharding constraints here may not be necessary
        q = apply_rotary_pos_embed(q, seq_dim=1)
        q = with_sharding_constraint(q, ("batch", "seq", "heads", "kv"))
        k = apply_rotary_pos_embed(k, seq_dim=1)
        k = with_sharding_constraint(k, ("batch", "seq", "heads", "kv"))

        dropout_rng = (
            self.make_rng("dropout") if train and self.qkv_dropout > 0.0 else None
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


class Embed(nn.Module):
    num_embeddings: int
    hidden_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    embedding_init: Callable = nn_initializers.variance_scaling(
        1.0, "fan_in", "normal", out_axis=0)

    def setup(self):
        self.embedding = nn_partitioning.param_with_axes(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.hidden_dim),
            self.param_dtype,
            axes=("vocab", "embed"))

    def __call__(self, inputs: jnp.ndarray):
        if not jnp.issubdtype(inputs, jnp.integer):
            raise ValueError("Input type must be an integer type")

        embedding = promote_dtype(
            self.embedding, dtype=self.dtype, inexact=False)[0]

        return jnp.take(embedding, inputs, axis=0)

    def attend(self, inputs: jnp.ndarray):
        inputs, embedding = promote_dtype(
            inputs, self.embedding, dtype=self.dtype)

        return jnp.dot(inputs, embedding.T)


class DecoderLayer(nn.Module):
    hidden_dim: int
    num_heads: int
    mlp_hidden_multiplier: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    dtype: Dtype
    param_dtype: Dtype
    attn_norm_fn: Callable
    use_scan: bool
    embedding_init: Callable
    kernel_init: Callable
    bias_init: Callable
    ln_scale_init: Callable

    @nn.compact
    def __call__(self,
                 inputs: jnp.ndarray,
                 decoder_mask: jnp.ndarray,
                 train: bool):
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

        # (None, None, None)
        multihead_out = MultiheadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_dropout=self.qkv_dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(ln1_out, decoder_mask, train)
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
        outputs = mlp_dropout + mlp_residual
        # (None, "mp", None)

        if self.use_scan:
            return outputs, None

        else:
            return outputs


class Decoder(nn.Module):
    num_layers: int
    hidden_dim: int
    num_heads: int
    mlp_hidden_multiplier: int
    vocab_size: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    dtype: Dtype
    param_dtype: Dtype
    attn_norm_fn: Callable
    use_scan: bool
    embedding_init: Callable
    kernel_init: Callable
    bias_init: Callable
    ln_scale_init: Callable
    shared_embed: Optional[nn.Module] = None

    @nn.compact
    def __call__(self,
                 inputs: jnp.ndarray,
                 decoder_mask: jnp.ndarray,
                 train: bool):
        # ("batch", "seq", "embed")

        # Run through core decoder layers with or w/o scan
        if self.use_scan:
            scan_layer = nn_partitioning.remat(
                DecoderLayer,
                prevent_cse=False,
                policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
                static_argnums=(2,))

            out, _ = nn_partitioning.scan_with_axes(
                scan_layer,
                variable_axes={"params": 0},
                split_rngs={
                    "params": True,
                    "dropout": True
                },
                in_axes=(nn.broadcast, nn.broadcast),
                length=self.num_layers,
                axis_name="layers")(
                    name="decoder",
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_hidden_multiplier=self.mlp_hidden_multiplier,
                    qkv_dropout=self.qkv_dropout,
                    msa_dropout=self.msa_dropout,
                    mlp_dropout=self.mlp_dropout,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    attn_norm_fn=self.attn_norm_fn,
                    use_scan=self.use_scan,
                    embedding_init=self.embedding_init,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    ln_scale_init=self.ln_scale_init,
                )(inputs, decoder_mask, train)

        else:
            layer_out = inputs
            for i in range(self.num_layers):
                layer_out = DecoderLayer(
                    name=f"layer_{i}",
                    config=self,
                )(layer_out, decoder_mask, train)
            out = layer_out

        ln_out = LayerNorm(
            name="ln_decoder",
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=self.ln_scale_init,
        )(out)

        if self.shared_embed is None:
            output = Dense(
                name="decoder_out_proj",
                features=(self.vocab_size,),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                named_axes=("embed", "vocab"))(ln_out)
        else:
            output = self.shared_embed.attend(ln_out)

        return output


class Transformer(nn.Module):
    num_layers: int
    hidden_dim: int
    num_heads: int
    mlp_hidden_multiplier: int
    vocab_size: int
    qkv_dropout: float
    msa_dropout: float
    mlp_dropout: float
    dtype: Dtype
    param_dtype: Dtype
    attn_norm_fn: Callable
    use_scan: bool
    use_shared_vocab_embed: bool
    embedding_init: Callable
    kernel_init: Callable
    bias_init: Callable
    ln_scale_init: Callable

    def setup(self):
        self.vocab_embed = Embed(
            num_embeddings=self.vocab_size,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=self.embedding_init)

        self.decoder = Decoder(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_hidden_multiplier=self.mlp_hidden_multiplier,
            vocab_size=self.vocab_size,
            qkv_dropout=self.qkv_dropout,
            msa_dropout=self.msa_dropout,
            mlp_dropout=self.mlp_dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            attn_norm_fn=self.attn_norm_fn,
            use_scan=self.use_scan,
            embedding_init=self.embedding_init,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            ln_scale_init=self.ln_scale_init,
            shared_embed=self.vocab_embed if self.use_shared_vocab_embed else None)

    def decode(self,
               embeddings: jnp.ndarray,
               decoder_mask: jnp.ndarray,
               train: bool):
        decoder_out = self.decoder(embeddings, decoder_mask, train)
        return decoder_out

    def __call__(self,
                 inputs: jnp.ndarray,
                 decoder_mask: jnp.ndarray,
                 train: bool):
        embeddings = self.vocab_embed(inputs)
        out = self.decode(embeddings, decoder_mask, train)
        return out
