import functools
import logging
import os
import sys

import optax
import fiddle as fdl
import jax
from jax import numpy as jnp, random

import partitioning
import optimizers
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

    def make_partitioned_model(
        self,
        mesh,
        module_in_sharding_specs,
        module_out_sharding_specs
    ):
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

    def get_example_and_target(self, dset_gen, **kwargs):
        data = next(dset_gen)
        batch = self.right_shift_batch(data)
        target = jax.nn.one_hot(data, num_classes=kwargs["vocab_size"])
        return batch, target

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
        loss_fn_kwargs,
    ):
        # Initialize mode and optimizer states
        rng, init_rng = random.split(rng)
        params = model.init(init_rng, dummy_data, *init_args)
        optim_state = optim.init(params)

        # Prepare loss function
        loss_fn = functools.partial(
            losses.generic_loss_fn,
            model=model,
            tgt_loss_fn=losses.stable_cross_entropy_loss_with_logits,
            apply_args=apply_args,
            loss_fn_kwargs=loss_fn_kwargs,
        )

        # Start dataset iteration
        dataset_generator = iter(dataset)

        total_seen_tokens = 0

        # Training loop
        for i in range(num_epochs):
            if total_seen_tokens > max_tokens:
                break

            # Split rngs as needed
            rng, dropout_rng = random.split(rng)

            # Get batch of data and targets
            batch, targets = self.get_example_and_target(
                dataset_generator, vocab_size=dataset.vocab_size)

            # Calculate loss and raw grads
            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                batch,
                targets,
                apply_rngs={"dropout": dropout_rng})

            # Update optimizer state and calculate parameter updates
            updates, optim_state = optim.update(grads, optim_state)

            # New set of parameters
            params = optim.apply_updates(params, updates)

            total_seen_tokens += batch.size

            print(f"Loss at epoch {i}: {loss}")
