import jax

from data.device_handling import get_default_device


class JAXDataLoader:
    """
    More JAX-idiomatic data loader using JAX's random number generation.
    Useful when you need reproducible shuffling with PRNG keys.
    """

    def __init__(
        self,
        data,
        targets,
        batch_size=None,
        shuffle=True,
        rng_key=jax.random.PRNGKey(42),
    ):
        self.data = data
        self.targets = targets

        self.shuffle = shuffle
        self.n_samples = len(self.data)
        if batch_size is None:
            self.batch_size = JAXDataLoader.get_batch_size()
        else:
            self.batch_size = batch_size
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
        self.rng_key = rng_key

    def __iter__(self):
        if self.shuffle:
            # Use JAX's PRNG for reproducible shuffling
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, self.n_samples)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.n_samples)

        batch_data = self.data[start_idx:end_idx]
        batch_targets = self.targets[start_idx:end_idx]

        self.current_idx = end_idx

        return batch_data, batch_targets

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        """Property to mimic PyTorch DataLoader's dataset attribute."""
        return type("Dataset", (), {"__len__": lambda self: self.n_samples})()

    @staticmethod
    def get_batch_size(default_cpu_bs=32, gpu_bs=128) -> int:
        """Determine batch size based on available device."""
        device = get_default_device()
        if device.platform == "gpu" or device.platform == "tpu":
            return gpu_bs
        else:
            return default_cpu_bs
