from dataclasses import dataclass
from typing import Callable

import flax.linen.partitioning as nn_partitioning
import jax
from jax import numpy as jnp, random
from jax.experimental import maps
from jax.experimental.pjit import pjit

import layers
from layers import Dtype
import partitioning


def _init_test(
    module,
    init_args,
    apply_args,
    in_sharding_spec,
    out_sharding_spec
):
    # initialize layer to test
    sa_layer = module(**init_args)

    # get device mesh for pjit
    mesh = partitioning.get_mesh(1, 2, 2, 1)

    param_sharding_specs = partitioning.construct_module_sharding_spec(
        module, init_args, apply_args
    )

    module_in_sharding_specs = nn_partitioning.logical_to_mesh_axes(
        in_sharding_spec)
    module_out_sharding_specs = nn_partitioning.logical_to_mesh_axes(
        out_sharding_spec)

    return (
        sa_layer,
        mesh,
        param_sharding_specs,
        module_in_sharding_specs,
        module_out_sharding_specs,
    )


@dataclass(frozen=True)
class MultiheadAttentionArgs:
    hidden_dim: int = 8
    num_heads: int = 2
    qkv_dropout: float = 0.1
    msa_dropout: float = 0.1
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    attn_norm_fn: Callable = layers.default_attn_norm
    attn_mask_fn: Callable = layers.make_causal_mask
    # Input params
    batch_size: int = 2
    seq_len: int = 4

    def __post_init__(self):
        pass

    def init_args(self):
        return dict(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            qkv_dropout=self.qkv_dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            attn_norm_fn=self.attn_norm_fn,
            attn_mask_fn=self.attn_mask_fn,
        )

    def apply_args(self):
        inputs = jnp.ones((self.batch_size, self.seq_len, self.hidden_dim))
        # TODO: Maybe make attention mask an input
        return {"inputs": inputs, "mask": jnp.zeros(1), "train": False}


class MultiheadAttentionTest:
    def __init__(self):
        self.args_cls = MultiheadAttentionArgs()
        self.module = layers.MultiheadAttention

        partitioning.init_partitioning_rules()

    def test_sharding(self):
        num_gpus = jax.device_count()
        assert num_gpus > 1, "Running sharding test on 1 GPU!"

        init_args = self.args_cls.init_args()
        apply_args = self.args_cls.apply_args()

        (
            layer,
            mesh,
            param_sharding_specs,
            module_in_sharding_specs,
            module_out_sharding_specs,
        ) = _init_test(
            self.module,
            init_args,
            apply_args,
            ("batch", "seq", "embed"),
            ("batch", "seq", "embed"),
        )

        # pjit `module.apply` and `module.init`
        pjit_init = pjit(
            lambda x: layer.init(
                random.PRNGKey(0), x, apply_args["mask"], apply_args["train"]
            ),
            in_axis_resources=module_in_sharding_specs,
            out_axis_resources=param_sharding_specs,
        )
        pjit_apply = pjit(
            lambda p, x: layer.apply(
                p, x, apply_args["mask"], apply_args["train"]),
            in_axis_resources=(param_sharding_specs, module_in_sharding_specs),
            out_axis_resources=module_in_sharding_specs,
        )

        # Init params and run forward pass
        with maps.Mesh(mesh.devices, mesh.axis_names):
            params = pjit_init(apply_args["inputs"])
            output = pjit_apply(params, apply_args["inputs"])

        # Test param and output sharding
        assert output.device_buffers[0].shape == (
            self.args_cls.batch_size,
            self.args_cls.seq_len // num_gpus,
            self.args_cls.hidden_dim,
        )
        assert params["params"]["q_proj"]["kernel"].shape == (
            self.args_cls.hidden_dim,
            self.args_cls.num_heads,
            self.args_cls.hidden_dim // self.args_cls.num_heads,
        )
        assert params["params"]["q_proj"]["kernel"].device_buffers[0].shape == (
            self.args_cls.hidden_dim,
            self.args_cls.num_heads // 2,
            self.args_cls.hidden_dim // self.args_cls.num_heads,
        )


@dataclass(frozen=True)
class DecoderLayerArgs:
    hidden_dim: int = 8
    num_heads: int = 2
    mlp_hidden_multiplier: int = 4
    qkv_dropout: float = 0.1
    msa_dropout: float = 0.1
    mlp_dropout: float = 0.1
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    attn_norm_fn: Callable = layers.default_attn_norm
    attn_mask_fn: Callable = layers.make_causal_mask
    # Input params
    batch_size: int = 2
    seq_len: int = 4

    def __post_init__(self):
        pass

    def init_args(self):
        return dict(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_hidden_multiplier=self.mlp_hidden_multiplier,
            qkv_dropout=self.qkv_dropout,
            msa_dropout=self.msa_dropout,
            mlp_dropout=self.mlp_dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            attn_norm_fn=self.attn_norm_fn,
            attn_mask_fn=self.attn_mask_fn,
        )

    def apply_args(self):
        inputs = jnp.ones((self.batch_size, self.seq_len, self.hidden_dim))
        # TODO: Maybe make attention mask an input
        return {"inputs": inputs, "train": False}


class DecoderLayerTest:
    def __init__(self):
        self.args_cls = DecoderLayerArgs()
        self.module = layers.DecoderLayer

        partitioning.init_partitioning_rules()

    def test_sharding(self):
        num_gpus = jax.device_count()
        assert num_gpus > 1, "Running sharding test on 1 GPU!"

        init_args = self.args_cls.init_args()
        apply_args = self.args_cls.apply_args()

        (
            layer,
            mesh,
            param_sharding_specs,
            module_in_sharding_specs,
            module_out_sharding_specs,
        ) = _init_test(
            self.module,
            init_args,
            apply_args,
            ("batch", "seq", "embed"),
            ("batch", "seq", "embed"),
        )

        pjit_init = pjit(
            lambda x: layer.init(random.PRNGKey(0), x, apply_args["train"]),
            in_axis_resources=module_in_sharding_specs,
            out_axis_resources=param_sharding_specs,
        )
        pjit_apply = pjit(
            lambda p, x: layer.apply(p, x, apply_args["train"]),
            in_axis_resources=(param_sharding_specs, module_in_sharding_specs),
            out_axis_resources=module_out_sharding_specs,
        )

        # Init params and run forward pass
        with maps.Mesh(mesh.devices, mesh.axis_names):
            params = pjit_init(apply_args["inputs"])
            output = pjit_apply(params, apply_args["inputs"])

        # TODO: Add tests
        assert output.device_buffers[0].shape == (2, 2, 8)


def main():
    self_attention_test = MultiheadAttentionTest()
    self_attention_test.test_sharding()
    decoder_layer_test = DecoderLayerTest()
    decoder_layer_test.test_sharding()


if __name__ == "__main__":
    main()
