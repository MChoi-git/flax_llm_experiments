from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import pjit
import optax

from layers import Dtype


class PartitionedOptim:
    def __init__(self,
                 optim,
                 mesh,
                 param_sharding_specs):
        self.param_sharding_specs = param_sharding_specs
        self.mesh = mesh
        self.optim = optim

        self.optim_state_sharding_spec = optax._src.transform.ScaleByAdamState(
            None,                       # count
            self.param_sharding_specs,  # mu
            self.param_sharding_specs)  # nu

        self.pjit_init_fn = pjit(
            lambda p: self.optim.init(p),
            in_axis_resources=(self.param_sharding_specs,),
            out_axis_resources=(self.optim_state_sharding_spec, None))

        self.pjit_update_fn = pjit(
            lambda g, s: self.optim.update(g, s),
            in_axis_resources=(
                self.param_sharding_specs,
                (self.optim_state_sharding_spec, None)),
            out_axis_resources=(
                self.param_sharding_specs,
                (self.optim_state_sharding_spec, None)))

        self.pjit_apply_updates_fn = pjit(
            lambda p, u: optax.apply_updates(p, u),
            in_axis_resources=(
                self.param_sharding_specs,
                self.param_sharding_specs),
            out_axis_resources=self.param_sharding_specs)

    def init(self, params):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_init_fn(params)

    def update(self, grads, optim_state):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_update_fn(grads, optim_state)

    def apply_updates(self, params, updates):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_apply_updates_fn(params, updates)
