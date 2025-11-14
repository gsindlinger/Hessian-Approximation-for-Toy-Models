import jax
import jax.numpy as jnp

device = jax.devices()[0]

A = jax.random.normal(jax.random.PRNGKey(0), (1700, 1700))
B = jax.random.normal(jax.random.PRNGKey(1), (10, 10))
C = jax.random.normal(jax.random.PRNGKey(2), (17000, 17000))


def peak_memory_in_gb():
    memory_stats = device.memory_stats()
    memory_ram = memory_stats["peak_bytes_in_use"]
    # in GB
    memory_ram_gb = memory_ram / (1024**3)
    return memory_ram_gb


@jax.jit
def l2_diff(A, B, C):
    D = jnp.kron(A, B)
    diff = D - C
    diff_l2 = jnp.linalg.norm(diff, "fro")
    return diff_l2


@jax.jit
def l2_diff_without_kron(D, C):
    diff = D - C
    diff_l2 = jnp.linalg.norm(diff, "fro")
    return diff_l2


@jax.jit
def jitted_test(A, B, metric_fn):
    jnp.kron(A, B)
    return metric_fn(A, B)


l2_diff(A, B, C)
peak_memory_in_gb()
print(f"Peak memory with Kronecker product: {peak_memory_in_gb():.2f} GB")

# reset memory stats
_ = device.memory_stats()


jitted_test(A, B, l2_diff_without_kron)
peak_memory_in_gb()
print(f"Peak memory without Kronecker product: {peak_memory_in_gb():.2f} GB")
