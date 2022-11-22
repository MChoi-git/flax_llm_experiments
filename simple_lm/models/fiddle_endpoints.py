import flax.linen.initializers as nn_initializers
import jax
from jax import numpy as jnp, random
import fiddle as fdl

import layers
import optax
import datasets


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
        vocab_size=50,
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
        num_layers=2,
        hidden_dim=8,
        num_heads=2,
        mlp_hidden_multiplier=4,
        vocab_size=50,
        qkv_dropout=0.1,
        msa_dropout=0.1,
        mlp_dropout=0.1,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        attn_norm_fn=layers.default_attn_norm,
        use_scan=True,
        use_shared_vocab_embed=True,
        embedding_init=nn_initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0),
        kernel_init=jax.nn.initializers.lecun_normal(),
        bias_init=jax.nn.initializers.zeros,
        ln_scale_init=jax.nn.initializers.ones,
    )
    return transformer


def make_transformer():
    """PALM 8B"""
    transformer = fdl.Config(
        layers.Transformer,
        num_layers=32,
        hidden_dim=4096,
        num_heads=16,
        mlp_hidden_multiplier=4,
        # Regular PALM is sentencepiece 256k vocab size, but we use GPT2 here
        vocab_size=50265,
        qkv_dropout=0.,
        msa_dropout=0.,
        mlp_dropout=0.,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        attn_norm_fn=layers.default_attn_norm,
        use_scan=True,
        use_shared_vocab_embed=True,
        embedding_init=nn_initializers.variance_scaling(
            1.0, "fan_in", "normal", out_axis=0),
        kernel_init=jax.nn.initializers.lecun_normal(),
        bias_init=jax.nn.initializers.zeros,
        ln_scale_init=jax.nn.initializers.ones,
    )
    return transformer


def make_test_synth_dataset():
    dataset = fdl.Config(
        datasets.SyntheticDataset,
        rng=random.PRNGKey(1),
        batch_size=2,
        unbatched_shape=(4,),
        vocab_size=50,
        dtype=jnp.int32,
    )
    return dataset


def make_test_dummy_dataset():
    dataset = fdl.Config(
        datasets.DummyDataset,
        rng=random.PRNGKey(1),
        num_examples=1,
        batch_size=2,
        unbatched_shape=(4,),
        vocab_size=50,
        dtype=jnp.int32,
    )
    return dataset


def make_test_optimizer():
    optim = fdl.Config(
        optax.adam,
        learning_rate=1e-1,
        b1=0.9,
        b2=0.999,
        eps_root=1e-6,
        mu_dtype=jnp.float32,
    )
    return optim
