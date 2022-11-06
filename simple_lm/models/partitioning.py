import flax
import flax.linen.partitioning as nn_partitioning
import jax
from jax import random
from jax.experimental import maps
from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint
import numpy as np


GPUS_PER_NODE = 4


def get_mesh(num_nodes, gpus_per_node, mp_size, dp_size):
    """Parse desired mesh given some hardware topology. Return a mesh."""
    world_size = num_nodes * gpus_per_node

    assert world_size % mp_size == 0
    assert world_size % dp_size == 0
    assert gpus_per_node <= GPUS_PER_NODE
    # Model parallel between nodes is bad
    assert mp_size <= GPUS_PER_NODE

    mesh_shape = (dp_size, mp_size)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))

    return mesh


DEFAULT_RULES = {
    "batch": None,
    "heads": "mp",
    "embed": None,
    "mlp": "mp",
    "joined_kv": "mp",
    "kv": None,
    "seq": "mp",
}


def init_partitioning_rules(override_rules=()) -> None:
    """Set global axis rules for flax partitioning."""
    for override in override_rules:
        if override[0] in DEFAULT_RULES:
            DEFAULT_RULES[override[0]] = override[1]

    RULES = tuple([(k, v) for k, v in DEFAULT_RULES.items()])

    nn_partitioning.set_axis_rules(RULES)


def construct_module_sharding_spec(cls, init_args, apply_args):
    """
    Initialize a model with some dummy initialization and apply args, then
    return its
    """
    module = cls(**init_args)
    params = module.init(random.PRNGKey(0), **apply_args)

    # Convert every leaf to PartitionSpec
    param_axes = nn_partitioning.get_axis_names(params["params_axes"])

    # Convert every leaf from logical to physical specs
    param_sharding_specs = jax.tree_util.tree_map(
        nn_partitioning.logical_to_mesh_axes, param_axes
    )

    # Return in a compatible pytree format with params
    param_sharding_specs = flax.core.FrozenDict(
        {"params": param_sharding_specs, "params_axes": None}
    )

    return param_sharding_specs


def with_sharding_constraint(x, axis_resources):
    """
    Wrapper around `pjit.with_sharding_constraint` that no-ops when not called
    in the context of a mesh.
    """
    maps_env = jax.experimental.maps.thread_resources.env

    if maps_env.physical_mesh.devices.shape == ():
        return x
    else:
        axis_resources = nn_partitioning.logical_to_mesh_axes(axis_resources)
        return _with_sharding_constraint(x, axis_resources)
