import jax
import jax.numpy as jnp


class JAXDataLoader:
    """
    More JAX-idiomatic data loader using JAX's random number generation.
    Useful when you need reproducible shuffling with PRNG keys.
    """

    def __init__(
        self,
        inputs,
        targets,
        batch_size=None,
        shuffle=True,
        rng_key=jax.random.PRNGKey(42),
        pad_to_batch_size: bool = False,
    ):
        # Move data to device once here so that per-epoch shuffling and
        # per-batch slicing are fully device-side (no repeated host→device
        # copies when inputs/targets are numpy arrays).
        self.data = jnp.asarray(inputs)
        self.targets = jnp.asarray(targets)

        self.shuffle = shuffle
        if batch_size is None:
            self.batch_size = JAXDataLoader.get_batch_size()
        else:
            self.batch_size = batch_size

        # Pad dataset so its length is an exact multiple of batch_size.
        # This guarantees every batch has the same shape, avoiding XLA
        # recompilation triggered by a variable-size last batch on GPU.
        # Padding wraps around from the start of the dataset; the duplicated
        # samples have negligible effect on gradient quality.
        if pad_to_batch_size:
            remainder = len(self.data) % self.batch_size
            if remainder != 0:
                pad = self.batch_size - remainder
                self.data = jnp.concatenate([self.data, self.data[:pad]], axis=0)
                self.targets = jnp.concatenate(
                    [self.targets, self.targets[:pad]], axis=0
                )

        self.n_samples = len(self.data)
        self.n_batches = self.n_samples // self.batch_size
        self.rng_key = rng_key

    def __iter__(self):
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(
                self.rng_key
            )  # allow for reproducibility
            indices = jax.random.permutation(subkey, self.n_samples)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_data = self.data[start_idx:end_idx]
        batch_targets = self.targets[start_idx:end_idx]

        self.current_idx = end_idx
        return batch_data, batch_targets

    def __len__(self):
        return self.n_batches

    @staticmethod
    def get_batch_size(default_cpu_bs=32, gpu_bs=128) -> int:
        """Determine batch size based on available device."""
        device = jax.devices()[0]
        if device.platform == "gpu" or device.platform == "tpu":
            return gpu_bs
        else:
            return default_cpu_bs
