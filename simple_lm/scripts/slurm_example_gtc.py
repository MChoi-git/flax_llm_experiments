import argparse
import os

import jax


def main():
    print(f"Global rank{os.environ['SLURM_PROCID']}/{int(os.environ['SLURM_NTASKS']) - 1}, node{os.environ['SLURM_NODEID']}/{int(os.environ['SLURM_NNODES']) - 1}")
    jax.distributed.initialize()

    xs = jax.numpy.ones(jax.local_device_count())

    y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(xs)

    print(f"\nProcess: {jax.process_index()}\nGlobal device count: {jax.device_count()}"
          f"\nLocal device count: {jax.local_device_count()}"
          f"\nInput: {xs}\nOutput: {y}\n\n")


if __name__ == "__main__":
    main()
