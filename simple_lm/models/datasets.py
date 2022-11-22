from itertools import cycle

import jax
from jax import numpy as jnp, random


class BaseDataset:
    def __getitem__(self, idx: int):
        raise NotImplementedError("Dataset class must implement this")

    def __iter__(self):
        raise NotImplementedError("Dataset class must implement this")


class DummyDataset(BaseDataset):
    def __init__(self, rng, num_examples, batch_size, unbatched_shape, vocab_size, dtype):
        """
        Dataset which simply returns the same number of examples infinitely.
        Usefule when sanity-checking networks to overfit to a few batches of
        data.
        """
        self.rng = rng
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.unbatched_shape = unbatched_shape
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.dset = random.randint(
            rng,
            minval=0,
            maxval=self.vocab_size,
            shape=(self.num_examples, self.batch_size) + self.unbatched_shape,
            dtype=self.dtype,
        )

    def __getitem__(self, idx: int):
        raise Exception("Synthetic dataset examples are lazily generated. "
                        "Call iter(...) instead.")

    def __iter__(self):
        for data in cycle(self.dset):
            yield data


class SyntheticDataset(BaseDataset):
    def __init__(self, rng, batch_size, unbatched_shape, vocab_size, dtype):
        """
        Synthetic dataset which continuously returns randomly generated
        integer batches. No batch is the same, and there is no notion of
        epochs in this case.
        """
        self.rng = rng
        self.batch_size = batch_size
        self.unbatched_shape = unbatched_shape
        self.vocab_size = vocab_size
        self.dtype = dtype

    def __getitem__(self, idx: int):
        raise Exception("Synthetic dataset examples are lazily generated. "
                        "Call iter(...) instead.")

    def __iter__(self):
        while True:
            self.rng, key = random.split(self.rng)
            yield random.randint(
                key,
                minval=0,
                maxval=self.vocab_size,
                shape=(self.batch_size,) + self.unbatched_shape,
                dtype=self.dtype,
            )
