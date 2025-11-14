import jax

from config.config import Config


def get_default_device() -> jax.Device:  # type: ignore
    """Get the default device, GPU > TPU > CPU as"""
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            return gpu_devices[0]
    except RuntimeError:
        pass

    try:
        tpu_devices = jax.devices("tpu")
        if tpu_devices:
            return tpu_devices[0]
    except RuntimeError:
        pass

    return jax.devices()[0]


def get_device_from_config(config: Config):
    """Get device based on config setting."""
    if config.device == "auto":
        return get_default_device()
    else:
        return jax.devices(config.device)[0]


def get_device_by_name(device_name: str | None) -> jax.Device:  # type: ignore
    """Get device by its name."""
    if device_name is None or device_name == "auto":
        return get_default_device()
    return jax.devices(device_name)[0]
