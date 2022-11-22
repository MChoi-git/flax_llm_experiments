import jax
from jax import numpy as jnp, random

import partitioning
import fiddle_endpoints
import layers
from tasks import TrainTask


def main():
    BSZ = 2
    SEQ_LEN = 4

    dummy_data = jnp.ones(
        (BSZ, SEQ_LEN),
        dtype=jnp.int32)

    num_gpus = jax.device_count()
    mesh = partitioning.get_mesh(
        num_nodes=1,
        gpus_per_node=num_gpus,
        mp_size=2,
        dp_size=1)

    task = TrainTask(
        fiddle_endpoints.make_test_transformer,
        fiddle_endpoints.make_test_dummy_dataset,
        fiddle_endpoints.make_test_optimizer,
    )

    decoder_mask = layers.make_causal_mask(SEQ_LEN, dtype=jnp.float32)
    init_args = (decoder_mask, False)
    apply_args = (decoder_mask, True)
    loss_fn_kwargs = {"z_loss": 10e-4}

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
        num_epochs=100,
        max_tokens=10000000,
        init_args=init_args,
        apply_args=apply_args,
        loss_fn_kwargs=loss_fn_kwargs,
    )


if __name__ == "__main__":
    main()
