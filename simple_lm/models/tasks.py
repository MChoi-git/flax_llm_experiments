import functools

import optax
import fiddle as fdl
import jax
from jax import numpy as jnp, random

import fiddle_endpoints
import partitioning
import layers
import optimizers
import datasets
import losses


class BaseTask:
    def __init__(self, model_cfg_fn, dataset_cfg_fn, optim_cfg_fn):
        self.model_cfg = model_cfg_fn()
        self.dataset_cfg = dataset_cfg_fn()
        self.optim_cfg = optim_cfg_fn()

    def make_model(self):
        return fdl.build(self.model_cfg)

    def make_dataset(self):
        return fdl.build(self.dataset_cfg)

    def make_optimizer(self):
        return fdl.build(self.optim_cfg)


class TrainTask(BaseTask):
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def right_shift_batch(batch, pad_token=0):
        return jnp.roll(
            jax.lax.pad(
                batch,
                padding_value=pad_token,
                padding_config=((0, 0, 0), (0, 1, 0))),
            shift=1,
            axis=-1)[:, :-1]

    def make_partitioned_model(self, mesh, module_in_sharding_specs, module_out_sharding_specs):
        model = self.make_model()
        return partitioning.PartitionedModel(
            model,
            mesh,
            module_in_sharding_specs,
            module_out_sharding_specs)

    def make_partitioned_optim(self, mesh, param_sharding_specs):
        optim = self.make_optimizer()
        return optimizers.PartitionedOptim(
            optim,
            mesh,
            param_sharding_specs)

    def setup(self, mesh, module_in_sharding_specs, module_out_sharding_specs):
        model = self.make_partitioned_model(
            mesh,
            module_in_sharding_specs,
            module_out_sharding_specs)
        optim = self.make_partitioned_optim(mesh, model.param_sharding_specs)
        dataset = self.make_dataset()
        return model, optim, dataset

    def train(
        self,
        model,
        optim,
        dataset,
        rng,
        dummy_data,
        num_epochs,
        max_tokens,
        init_args,
        apply_args,
    ):
        rng, init_rng = random.split(rng)
        params = model.init(init_rng, dummy_data, *init_args)
        optim_state = optim.init(params)

        def generic_loss_fn(params, batch, targets, tgt_loss_fn, apply_rngs, loss_fn_kwargs={}):
            logits = model.apply(params, batch, *apply_args, rngs=apply_rngs)
            loss = tgt_loss_fn(logits, targets, **loss_fn_kwargs)
            return jnp.mean(loss)

        loss_fn = functools.partial(
            generic_loss_fn,
            tgt_loss_fn=losses.stable_cross_entropy_loss_with_logits,
        )

        dataset_generator = iter(dataset)

        for i in range(num_epochs):
            rng, dropout_rng = random.split(rng)
            data = next(dataset_generator)

            batch = self.right_shift_batch(data)
            targets = jax.nn.one_hot(data, num_classes=dataset.vocab_size)

            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                batch,
                targets,
                apply_rngs={"dropout": dropout_rng})

            updates, optim_state = optim.update(grads, optim_state)

            params = optim.apply_updates(params, updates)

            print(f"Loss at epoch {i}: {loss}")


SEQ_LEN = 4

dummy_data = jnp.ones(
    (2, 4),
    dtype=jnp.int32)

num_gpus = jax.device_count()
mesh = partitioning.get_mesh(1, 2, 2, 1)

task = TrainTask(
    fiddle_endpoints.make_test_transformer,
    fiddle_endpoints.make_test_synth_dataset,
    fiddle_endpoints.make_test_optimizer,
)

decoder_mask = layers.make_causal_mask(SEQ_LEN, dtype=jnp.float32)
init_args = (decoder_mask, False)
apply_args = (decoder_mask, True)

# TODO: Include module sharding specs in fiddle configs
model, optim, dataset = task.setup(
    mesh=mesh,
    module_in_sharding_specs=("batch", "seq"),
    module_out_sharding_specs=("batch", "seq", "vocab"))

rng = random.PRNGKey(0)
task.train(
    model,
    optim,
    dataset,
    rng,
    dummy_data,
    num_epochs=10,
    max_tokens=10000000,
    init_args=init_args,
    apply_args=apply_args)
