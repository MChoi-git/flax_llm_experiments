from dataclasses import dataclass

import fiddle as fdl
import flax.linen.partitioning as nn_partitioning
import jax
from jax import numpy as jnp, random

import layers
import partitioning
import fiddle_endpoints


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
class RotaryPositionEmbedArgs:
    batch_size: int = 2
    seq_len: int = 4
    num_heads: int = 2
    hidden_dim: int = 8
    vocab_size: int = 6969

    def __post_init__(self):
        pass

    def init_args(self):
        return dict(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size)

    def apply_args(self):
        inputs = random.randint(
            random.PRNGKey(0),
            shape=(self.batch_size, self.seq_len, self.num_heads, self.hidden_dim //self.num_heads),
            minval=0,
            maxval=6969)
        return {"inputs": inputs}


class RotaryPositionEmbedTest:
    def __init__(self):
        self.args_cls = RotaryPositionEmbedArgs()
        self.params = self.args_cls.init_args()
        self.inputs = self.args_cls.apply_args()

    def test_fixed_pos_embed_fn(self):
        inputs = self.args_cls.apply_args()
        outputs = layers._fixed_pos_embed(x=inputs["inputs"], seq_dim=1)

        assert map(
            lambda x: x.shape[0] == (
                self.params["seq_len"], self.params["hidden_dim"] // self.params["num_heads"] // 2),
            outputs,
        )

    def test_rotate_every_two_fn(self):
        inputs = self.args_cls.apply_args()
        outputs = layers._rotate_every_two(inputs["inputs"])

        assert outputs.shape == inputs.shape

    def test_apply_rotary_pos_embed_fn(self):
        inputs = self.args_cls.apply_args()
        outputs = layers.apply_rotary_pos_embed(inputs["inputs"])

        assert outputs.shape == inputs.shape


class BaseTest:
    def __init__(self, cfg_constructor_fn):
        self.cfg_constructor_fn = cfg_constructor_fn
        self.cfg = self.cfg_constructor_fn()

    def make_model(self):
        return fdl.build(self.cfg)

    def apply_args(self):
        raise NotImplementedError("Test classes must implement this")


class MultiheadAttentionTest(BaseTest):
    def __init__(self, cfg_constructor_fn):
        super().__init__(cfg_constructor_fn)

    def apply_args(self):
        bsz = 2
        seq_len = 4

        inputs = jnp.ones((bsz, seq_len, self.cfg.hidden_dim))
        decoder_mask = layers.make_causal_mask(seq_len, self.cfg.param_dtype)

        return inputs, decoder_mask, False

    def test_sharding(self):
        num_gpus = jax.device_count()
        assert num_gpus > 1, "Running sharding test on 1 GPU!"

        mesh = partitioning.get_mesh(1, 2, 2, 1)

        model = self.make_model()
        inputs, decoder_mask, train = self.apply_args()

        apply_args = (decoder_mask, train)

        model = partitioning.PartitionedModel(
            model=model,
            rng=random.PRNGKey(0),
            mesh=mesh,
            dummy_data=inputs,
            apply_args=apply_args,
            module_in_sharding_specs=("batch", "seq", "embed"),
            module_out_sharding_specs=("batch", "seq", "embed"))

        params = model.init_model()
        output = model.forward(params, inputs)

        assert output.device_buffers[0].shape == (2, 4, 8)


class DecoderTest(BaseTest):
    def __init__(self, cfg_constructor_fn):
        super().__init__(cfg_constructor_fn)

    def apply_args(self):
        bsz = 2
        seq_len = 4

        inputs = jnp.ones((bsz, seq_len, self.cfg.hidden_dim))
        decoder_mask = layers.make_causal_mask(seq_len, self.cfg.param_dtype)

        return inputs, decoder_mask, False  # pjit can't take kwargs

    def test_sharding(self):
        num_gpus = jax.device_count()
        assert num_gpus > 1, "Running sharding test on 1 GPU!"

        mesh = partitioning.get_mesh(1, 2, 2, 1)

        model = self.make_model()
        inputs, decoder_mask, train = self.apply_args()

        apply_args = (decoder_mask, train)

        model = partitioning.PartitionedModel(
            model=model,
            rng=random.PRNGKey(0),
            mesh=mesh,
            dummy_data=inputs,
            apply_args=apply_args,
            module_in_sharding_specs=("batch", "seq", "embed"),
            module_out_sharding_specs=("batch", "seq", "vocab"))

        params = model.init_model()
        output = model.forward(params, inputs)

        assert output.device_buffers[0].shape == (2, 4, 50000)


class TransformerTest(BaseTest):
    def __init__(self, cfg_constructor_fn):
        super().__init__(cfg_constructor_fn)

    def apply_args(self):
        bsz = 2
        seq_len = 4
        inputs = random.randint(
            random.PRNGKey(0), minval=0, maxval=50000, shape=(bsz, seq_len))
        decoder_mask = layers.make_causal_mask(seq_len, self.cfg.param_dtype)

        return inputs, decoder_mask, False

    def test_sharding(self):
        num_gpus = jax.device_count()
        assert num_gpus > 1, "Running sharding test on 1 GPU!"

        mesh = partitioning.get_mesh(1, 2, 2, 1)

        model = self.make_model()
        inputs, decoder_mask, train = self.apply_args()

        apply_args = (decoder_mask, train)

        model = partitioning.PartitionedModel(
            model=model,
            rng=random.PRNGKey(0),
            mesh=mesh,
            dummy_data=inputs,
            apply_args=apply_args,
            module_in_sharding_specs=("batch", "seq"),
            module_out_sharding_specs=("batch", "seq", "vocab"))

        params = model.init_model()
        output = model.forward(params, inputs)

        assert output.device_buffers[0].shape == (2, 4, 50000)


def main():
    self_attention_test = MultiheadAttentionTest(fiddle_endpoints.make_test_multihead_attention)
    self_attention_test.test_sharding()
    print("MultiheadAttention test passed")
    decoder_test = DecoderTest(fiddle_endpoints.make_test_decoder)
    decoder_test.test_sharding()
    print("Decoder test passed")
    transformer_test = TransformerTest(fiddle_endpoints.make_test_transformer)
    transformer_test.test_sharding()
    print("Transformer test passed")
    """
    rotary_embed_test = RotaryPositionEmbedTest()
    rotary_embed_test.test_fixed_pos_embed_fn()
    rotary_embed_test.test_rotate_every_two_fn()
    rotary_embed_test.test_apply_rotary_pos_embed()
    print("Rotary embeddings test passed")
    """


if __name__ == "__main__":
    main()
