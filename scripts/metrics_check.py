import gc

import jax
import jax.numpy as jnp

from src.utils.utils import get_device_memory_stats

device = jax.devices()[0]

A = jax.random.normal(jax.random.PRNGKey(0), (1700, 1700))
B = jax.random.normal(jax.random.PRNGKey(1), (10, 10))
C = jax.random.normal(jax.random.PRNGKey(2), (17000, 17000))

lamdba = jax.random.normal(jax.random.PRNGKey(3), (1700, 10))


@jax.jit
def l2_diff(A, B, C):
    D = jnp.kron(A, B)
    D = jnp.einsum("ij, jl -> il", D, D.T)
    diff = D - C
    diff_l2 = jnp.linalg.norm(diff, "fro")
    return diff_l2


def l2_diff_without_jit(A, B, C):
    D = jnp.kron(A, B)
    D = jnp.einsum("ij, jl -> il", D, D.T)
    diff = D - C
    diff_l2 = jnp.linalg.norm(diff, "fro")
    return diff_l2


result = l2_diff(A, B, C)
print(get_device_memory_stats("After JITed l2_diff"))

del result

gc.collect()

print(get_device_memory_stats("After garbage collection"))
result = l2_diff_without_jit(A, B, C)
print(get_device_memory_stats("After non-JITed l2_diff"))
