import flax
import flax.linen.partitioning as nn_partitioning
import jax
from jax import random, numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint, pjit
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
    "seq": None,    # TODO: Can we use sequence parallel?
    "vocab": "mp",
}


def init_partitioning_rules(override_rules=()) -> None:
    """Set global axis rules for flax partitioning."""
    for override in override_rules:
        if override[0] in DEFAULT_RULES:
            DEFAULT_RULES[override[0]] = override[1]

    RULES = tuple([(k, v) for k, v in DEFAULT_RULES.items()])

    nn_partitioning.set_axis_rules(RULES)


def construct_module_sharding_spec(module, dummy_data, *apply_args):
    """
    Initialize a model with some dummy initialization and apply args, then
    return its
    """
    params = module.init(random.PRNGKey(0), dummy_data, *apply_args)

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


def is_in_pjit_mesh_context():
    maps_env = jax.experimental.maps.thread_resources.env
    return maps_env.physical_mesh.devices.shape == ()


def with_sharding_constraint(x, axis_resources):
    """
    Wrapper around `pjit.with_sharding_constraint` that no-ops when not called
    in the context of a mesh.
    """

    if is_in_pjit_mesh_context():
        return x
    else:
        axis_resources = nn_partitioning.logical_to_mesh_axes(axis_resources)
        return _with_sharding_constraint(x, axis_resources)


class PartitionedModel:
    def __init__(self,
                 model,
                 rng,
                 mesh,
                 dummy_data,
                 apply_args,
                 module_in_sharding_specs,
                 module_out_sharding_specs):
        self.model = model
        self.rng = rng
        self.mesh = mesh
        self.dummy_data = dummy_data

        # Make module IO and parameter sharding specs
        self.module_in_sharding_specs = nn_partitioning.logical_to_mesh_axes(
            module_in_sharding_specs)
        self.module_out_sharding_specs = nn_partitioning.logical_to_mesh_axes(
            module_out_sharding_specs)
        self.param_sharding_specs = construct_module_sharding_spec(
            model, self.dummy_data, *apply_args)

        self.rng, init_rng = random.split(self.rng)

        self.pjit_init_fn = pjit(
            lambda x: self.model.init(init_rng, x, *apply_args),
            in_axis_resources=self.module_in_sharding_specs,
            out_axis_resources=self.param_sharding_specs)

        self.pjit_apply_fn = pjit(
            lambda p, x: self.model.apply(p, x, *apply_args),
            in_axis_resources=(
                self.param_sharding_specs, self.module_in_sharding_specs),
            out_axis_resources=self.module_out_sharding_specs)

    def init_model(self):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_init_fn(self.dummy_data)

    def forward(self, params, batch: jnp.ndarray):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_apply_fn(params, batch)
