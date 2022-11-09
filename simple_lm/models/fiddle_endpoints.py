import flax.linen.initializers as nn_initializers
import jax
import jax.numpy as jnp
import fiddle as fdl

import layers


def make_test_multihead_attention():
    mha = fdl.Config(
        layers.MultiheadAttention,
        hidden_dim=8,
        num_heads=2,
        qkv_dropout=0.1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        attn_norm_fn=layers.default_attn_norm,
        kernel_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros,
    )
    return mha


def make_test_decoder():
    decoder = fdl.Config(
        layers.Decoder,
        num_layers=16,
        hidden_dim=8,
        num_heads=2,
        mlp_hidden_multiplier=4,
        vocab_size=50000,
        qkv_dropout=0.1,
        msa_dropout=0.1,
        mlp_dropout=0.1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        attn_norm_fn=layers.default_attn_norm,
        use_scan=True,
        shared_embed=None,
        embedding_init=nn_initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0),
        kernel_init=jax.nn.initializers.lecun_normal(),
        bias_init=jax.nn.initializers.zeros,
        ln_scale_init=jax.nn.initializers.ones,
    )
    return decoder


def make_test_transformer():
    """Construct a Transformer class."""
    transformer = fdl.Config(
        layers.Transformer,
        num_layers=16,
        hidden_dim=8,
        num_heads=2,
        mlp_hidden_multiplier=4,
        qkv_dropout=0.1,
        msa_dropout=0.1,
        mlp_dropout=0.1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        attn_norm_fn=layers.default_attn_norm,
        use_scan=True,
        use_shared_vocab_embed=True,
    )
    return transformer
