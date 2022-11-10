from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import pjit
import optax

from layers import Dtype


class BasePartitionedOptimizer:
    def __init__(self, param_sharding_specs, mesh, optim_fn, *args):
        self.param_sharding_specs = param_sharding_specs
        self.mesh = mesh
        self.optim = optim_fn(*args)

    def init(self, params):
        raise NotImplementedError("Optimizer classes must implement this")

    def update(self, grads, optim_state, params):
        raise NotImplementedError("Optimizer classes must implement this")


class PartitionedAdam(BasePartitionedOptimizer):
    def __init__(self,
                 param_sharding_specs,  # Pytree
                 mesh,
                 b1: float,
                 b2: float,
                 eps_root: float,
                 mu_dtype: Optional[Dtype] = jnp.float32):
        self.hparams = (b1, b2, eps_root, mu_dtype)
        super().__init__(param_sharding_specs,
                         mesh,
                         optax._src.transform.scale_by_adam,
                         self.hparams)
        self.optim_state_sharding_spec = optax._src.transform.ScaleByAdamState(
            None,                       # count
            self.param_sharding_specs,  # mu
            self.param_sharding_specs)  # nu

        self.pjit_init_fn = pjit(
            lambda p: self.optim.init(p),
            in_axis_resources=self.param_sharding_specs,
            out_axis_resources=self.optim_state_sharding_spec)

        self.pjit_update_fn = pjit(
            lambda g, s, p: self.optim.update(g, s, p),
            in_axis_resources=(
                self.param_sharding_specs,
                self.param_sharding_specs,
                self.param_sharding_specs),
            out_axis_resources=(
                self.param_sharding_specs,
                self.optim_state_sharding_spec))

    def init(self, params):
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            return self.pjit_init_fn(params)

    def update(self, grads, optim_state, params):
        with maps.Mesh(self.mesh.devices, self.mesh_axis_names):
            return self.pjit_udpate_fn(grads, optim_state, params)
