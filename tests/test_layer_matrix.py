"""Unit tests for src/hessians/layer_matrix.py."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.hessians.layer_matrix import (
    DenseBlock,
    KroneckerFactors,
    LayerMatrix,
    LayerVector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_psd(key, n):
    """Random symmetric positive-definite (n, n) matrix."""
    A = jax.random.normal(key, (n, n))
    return A @ A.T + n * jnp.eye(n)


def _random_orthogonal_and_lambda(key, I, O):
    """Return `(Q_A, Q_G, Lambda)` suitable for a `KroneckerFactors` block."""
    key_A, key_G, key_L = jax.random.split(key, 3)
    A = _random_psd(key_A, I)
    G = _random_psd(key_G, O)
    _, Q_A = jnp.linalg.eigh(A)
    _, Q_G = jnp.linalg.eigh(G)
    # Random but positive (so inverse is well-defined) Lambda.
    Lambda = jax.random.uniform(key_L, (I, O), minval=0.5, maxval=3.0)
    return Q_A, Q_G, Lambda


# ---------------------------------------------------------------------------
# KroneckerFactors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("I,O", [(3, 4), (5, 2), (1, 6)])
def test_kronecker_factors_matvec_matches_dense(I, O):
    key = jax.random.PRNGKey(0)
    Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(key, I, O)
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    v = jax.random.normal(jax.random.PRNGKey(1), (I, O))
    y_fast = block.matvec(v)
    y_dense = (block.to_dense() @ v.flatten()).reshape(I, O)

    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_kronecker_factors_matvec_batched():
    I, O, B = 3, 4, 7
    key = jax.random.PRNGKey(42)
    Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(key, I, O)
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    v = jax.random.normal(jax.random.PRNGKey(7), (B, I, O))
    y_fast = block.matvec(v)
    dense = block.to_dense()
    y_dense = jnp.stack(
        [(dense @ v[b].flatten()).reshape(I, O) for b in range(B)], axis=0
    )

    assert y_fast.shape == (B, I, O)
    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_kronecker_factors_inverse_roundtrip():
    I, O = 4, 3
    key = jax.random.PRNGKey(1)
    Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(key, I, O)
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)
    inv = block.inverse()

    v = jax.random.normal(jax.random.PRNGKey(2), (I, O))
    round_trip = inv.matvec(block.matvec(v))
    assert jnp.allclose(round_trip, v, atol=1e-4, rtol=1e-4)


def test_kronecker_factors_inverse_with_damping():
    I, O = 3, 3
    key = jax.random.PRNGKey(3)
    Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(key, I, O)
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)
    damping = 0.5

    damped = block.damped(damping)
    inv = block.inverse(damping=damping)

    v = jax.random.normal(jax.random.PRNGKey(4), (I, O))
    round_trip = inv.matvec(damped.matvec(v))
    assert jnp.allclose(round_trip, v, atol=1e-4, rtol=1e-4)


def test_kronecker_factors_pseudo_inverse():
    I, O = 3, 3
    Q_A = jnp.eye(I)
    Q_G = jnp.eye(O)
    Lambda = jnp.array([[1.0, 2.0, 0.01], [0.02, 3.0, 4.0], [0.0, 5.0, 6.0]])
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)
    # Threshold 0.05 — anything below becomes 0 in the inverse.
    inv = block.inverse(pseudo_inverse_factor=0.05)
    expected = jnp.where(jnp.abs(Lambda) > 0.05, 1.0 / Lambda, 0.0)
    assert jnp.allclose(inv.Lambda, expected)


def test_kronecker_factors_damped_dense_matches_dense_plus_identity():
    I, O = 3, 4
    key = jax.random.PRNGKey(5)
    Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(key, I, O)
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    damping = 0.37
    damped_dense = block.damped(damping).to_dense()
    expected = block.to_dense() + damping * jnp.eye(I * O)
    assert jnp.allclose(damped_dense, expected, atol=1e-5)


def test_kronecker_factors_matmat_shared_basis():
    I, O = 3, 4
    key = jax.random.PRNGKey(6)
    Q_A, Q_G, Lambda_1 = _random_orthogonal_and_lambda(key, I, O)
    Lambda_2 = jax.random.uniform(jax.random.PRNGKey(7), (I, O), minval=0.5, maxval=3.0)

    b1 = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda_1)
    b2 = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda_2)

    product = b1.matmat(b2).to_dense()
    expected = b1.to_dense() @ b2.to_dense()
    assert jnp.allclose(product, expected, atol=1e-4, rtol=1e-4)


def test_kronecker_factors_matmat_rejects_different_type():
    I, O = 2, 3
    Q_A = jnp.eye(I)
    Q_G = jnp.eye(O)
    Lambda = jnp.ones((I, O))
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    class Dummy:
        pass

    with pytest.raises(TypeError):
        block.matmat(Dummy())  # type: ignore


def test_kronecker_factors_is_pytree():
    I, O = 2, 3
    Q_A = jnp.eye(I)
    Q_G = jnp.eye(O)
    Lambda = jnp.ones((I, O))
    block = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    leaves, treedef = jax.tree_util.tree_flatten(block)
    assert len(leaves) == 3
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, KroneckerFactors)
    assert jnp.allclose(reconstructed.Lambda, Lambda)


# ---------------------------------------------------------------------------
# LayerVector
# ---------------------------------------------------------------------------


def _multi_layer_shapes():
    return {
        "linear_0": (3, 4),
        "linear_1": (4, 5),
        "output": (5, 2),
    }


def test_layer_vector_to_flat_from_flat_roundtrip():
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())

    flat = jax.random.normal(jax.random.PRNGKey(0), (n,))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)
    round_trip = lvec.to_flat()
    assert jnp.allclose(round_trip, flat)


def test_layer_vector_to_flat_from_flat_batched():
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())
    B = 6

    flat = jax.random.normal(jax.random.PRNGKey(0), (B, n))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)
    for name in groups:
        I, O = shapes[name]
        assert lvec.blocks[name].shape == (B, I, O)
    round_trip = lvec.to_flat()
    assert round_trip.shape == (B, n)
    assert jnp.allclose(round_trip, flat)


def test_layer_vector_add_sub_mul():
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())

    a = LayerVector.from_flat(
        jax.random.normal(jax.random.PRNGKey(0), (n,)),
        shapes=shapes,
        param_groups=groups,
    )
    b = LayerVector.from_flat(
        jax.random.normal(jax.random.PRNGKey(1), (n,)),
        shapes=shapes,
        param_groups=groups,
    )

    assert jnp.allclose((a + b).to_flat(), a.to_flat() + b.to_flat())
    assert jnp.allclose((a - b).to_flat(), a.to_flat() - b.to_flat())
    assert jnp.allclose((2.0 * a).to_flat(), 2.0 * a.to_flat())


def test_layer_vector_is_pytree():
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())

    lvec = LayerVector.from_flat(
        jnp.arange(n, dtype=jnp.float32), shapes=shapes, param_groups=groups
    )
    leaves, treedef = jax.tree_util.tree_flatten(lvec)
    assert len(leaves) == len(groups)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, LayerVector)
    assert jnp.allclose(reconstructed.to_flat(), lvec.to_flat())


# ---------------------------------------------------------------------------
# LayerMatrix
# ---------------------------------------------------------------------------


def _random_layer_matrix(seed=0):
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(groups))
    diag_blocks = {}
    for k, g in zip(keys, groups):
        I, O = shapes[g]
        Q_A, Q_G, Lambda = _random_orthogonal_and_lambda(k, I, O)
        diag_blocks[g] = KroneckerFactors(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)
    return LayerMatrix.block_diagonal(diag_blocks, groups), shapes, groups


def test_layer_matrix_to_dense_matches_block_diag():
    lmat, shapes, groups = _random_layer_matrix()
    dense = lmat.to_dense()
    expected_blocks = [lmat.blocks[(g, g)].to_dense() for g in groups]
    expected = jax.scipy.linalg.block_diag(*expected_blocks)
    assert jnp.allclose(dense, expected)


def test_layer_matrix_matvec_matches_dense():
    lmat, shapes, groups = _random_layer_matrix()
    n = sum(I * O for I, O in shapes.values())
    flat = jax.random.normal(jax.random.PRNGKey(11), (n,))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    y_fast = (lmat @ lvec).to_flat()
    y_dense = lmat.to_dense() @ flat
    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_layer_matrix_matvec_batched():
    lmat, shapes, groups = _random_layer_matrix()
    n = sum(I * O for I, O in shapes.values())
    B = 4
    flat = jax.random.normal(jax.random.PRNGKey(12), (B, n))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    y_fast = (lmat @ lvec).to_flat()
    dense = lmat.to_dense()
    y_dense = flat @ dense.T
    assert y_fast.shape == (B, n)
    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_layer_matrix_matmat_shared_basis():
    lmat, _, _ = _random_layer_matrix(seed=0)
    lmat2, _, _ = _random_layer_matrix(seed=1)
    # Share the eigenbasis: take lmat2's Lambdas but lmat's Qs.
    shared_blocks = {}
    for (g, _), block in lmat.blocks.items():
        other = lmat2.blocks[(g, g)]
        shared_blocks[(g, g)] = KroneckerFactors(
            Q_A=block.Q_A, Q_G=block.Q_G, Lambda=other.Lambda
        )
    lmat2_shared = LayerMatrix(
        blocks=shared_blocks,
        param_groups=lmat.param_groups,
        layer_shapes=lmat.layer_shapes,
    )

    product = (lmat @ lmat2_shared).to_dense()
    expected = lmat.to_dense() @ lmat2_shared.to_dense()
    assert jnp.allclose(product, expected, atol=1e-4, rtol=1e-4)


def test_layer_matrix_inverse_roundtrip():
    lmat, shapes, groups = _random_layer_matrix()
    n = sum(I * O for I, O in shapes.values())
    flat = jax.random.normal(jax.random.PRNGKey(20), (n,))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    lmat_inv = lmat.inverse()
    round_trip = (lmat_inv @ (lmat @ lvec)).to_flat()
    assert jnp.allclose(round_trip, flat, atol=1e-4, rtol=1e-4)


def test_layer_matrix_damped_dense_matches_dense_plus_identity():
    lmat, _, _ = _random_layer_matrix()
    damping = 0.42
    dense = lmat.damped(damping).to_dense()
    expected = lmat.to_dense() + damping * jnp.eye(dense.shape[0])
    assert jnp.allclose(dense, expected, atol=1e-5)


def test_layer_matrix_inverse_with_damping_matches_dense_inverse():
    lmat, shapes, groups = _random_layer_matrix()
    damping = 0.1

    # (lmat + damping*I) @ v
    y_estimator = (lmat.damped(damping).inverse() @ lmat.damped(damping)).to_dense()
    assert jnp.allclose(y_estimator, jnp.eye(y_estimator.shape[0]), atol=1e-4)


def test_layer_matrix_is_pytree():
    lmat, _, _ = _random_layer_matrix()
    leaves, treedef = jax.tree_util.tree_flatten(lmat)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, LayerMatrix)
    assert jnp.allclose(reconstructed.to_dense(), lmat.to_dense())


# ---------------------------------------------------------------------------
# DenseBlock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("I,O", [(3, 4), (5, 2), (1, 6)])
def test_dense_block_matvec_matches_dense(I, O):
    """For a square diagonal block, matvec must agree with raw matmul."""
    n = I * O
    key = jax.random.PRNGKey(0)
    M = jax.random.normal(key, (n, n))
    block = DenseBlock(matrix=M, row_shape=(I, O), col_shape=(I, O))

    v = jax.random.normal(jax.random.PRNGKey(1), (I, O))
    y_fast = block.matvec(v)
    y_dense = (M @ v.flatten()).reshape(I, O)

    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_dense_block_matvec_off_diagonal_shapes():
    """Off-diagonal blocks: row_shape != col_shape; matvec must respect both."""
    I_i, O_i = 3, 4   # row layer
    I_j, O_j = 5, 2   # col layer
    n_i, n_j = I_i * O_i, I_j * O_j
    M = jax.random.normal(jax.random.PRNGKey(2), (n_i, n_j))
    block = DenseBlock(matrix=M, row_shape=(I_i, O_i), col_shape=(I_j, O_j))

    v = jax.random.normal(jax.random.PRNGKey(3), (I_j, O_j))
    y_fast = block.matvec(v)
    y_dense = (M @ v.flatten()).reshape(I_i, O_i)
    assert y_fast.shape == (I_i, O_i)
    assert jnp.allclose(y_fast, y_dense, atol=1e-5, rtol=1e-5)


def test_dense_block_matvec_batched():
    I, O, B = 3, 4, 5
    n = I * O
    M = jax.random.normal(jax.random.PRNGKey(4), (n, n))
    block = DenseBlock(matrix=M, row_shape=(I, O), col_shape=(I, O))

    v = jax.random.normal(jax.random.PRNGKey(5), (B, I, O))
    y_fast = block.matvec(v)
    expected = jnp.stack(
        [(M @ v[b].flatten()).reshape(I, O) for b in range(B)], axis=0
    )
    assert y_fast.shape == (B, I, O)
    assert jnp.allclose(y_fast, expected, atol=1e-5, rtol=1e-5)


def test_dense_block_inverse_roundtrip():
    I, O = 3, 4
    n = I * O
    key = jax.random.PRNGKey(6)
    M = _random_psd(key, n)   # PSD so the inverse is well-defined
    block = DenseBlock(matrix=M, row_shape=(I, O), col_shape=(I, O))
    inv = block.inverse()

    v = jax.random.normal(jax.random.PRNGKey(7), (I, O))
    round_trip = inv.matvec(block.matvec(v))
    assert jnp.allclose(round_trip, v, atol=1e-4, rtol=1e-4)


def test_dense_block_damped_matches_dense_plus_identity():
    I, O = 2, 3
    n = I * O
    M = jax.random.normal(jax.random.PRNGKey(8), (n, n))
    block = DenseBlock(matrix=M, row_shape=(I, O), col_shape=(I, O))

    damping = 0.42
    damped = block.damped(damping).to_dense()
    expected = M + damping * jnp.eye(n)
    assert jnp.allclose(damped, expected, atol=1e-5)


def test_dense_block_matmat_shape_check():
    """matmat: only legal when self.col_shape == other.row_shape."""
    a = DenseBlock(
        matrix=jnp.zeros((6, 12)),
        row_shape=(2, 3),
        col_shape=(4, 3),
    )
    b = DenseBlock(
        matrix=jnp.zeros((12, 8)),
        row_shape=(4, 3),
        col_shape=(2, 4),
    )
    product = a.matmat(b)
    assert product.matrix.shape == (6, 8)
    assert product.row_shape == (2, 3)
    assert product.col_shape == (2, 4)

    # Mismatched shapes should raise.
    bad = DenseBlock(
        matrix=jnp.zeros((9, 8)), row_shape=(3, 3), col_shape=(2, 4)
    )
    with pytest.raises(ValueError):
        a.matmat(bad)


def test_dense_block_inverse_rejects_non_square():
    block = DenseBlock(
        matrix=jnp.zeros((6, 12)),
        row_shape=(2, 3),
        col_shape=(4, 3),
    )
    with pytest.raises(ValueError):
        block.inverse()


def test_dense_block_is_pytree():
    I, O = 2, 3
    n = I * O
    M = jax.random.normal(jax.random.PRNGKey(9), (n, n))
    block = DenseBlock(matrix=M, row_shape=(I, O), col_shape=(I, O))

    leaves, treedef = jax.tree_util.tree_flatten(block)
    assert len(leaves) == 1   # only `matrix` is a child; shapes are static aux
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, DenseBlock)
    assert reconstructed.row_shape == (I, O)
    assert reconstructed.col_shape == (I, O)
    assert jnp.allclose(reconstructed.matrix, M)


# ---------------------------------------------------------------------------
# LayerMatrix.from_dense — full (i, j) grid case
# ---------------------------------------------------------------------------


def _random_dense_layer_matrix(seed=0):
    """Build a fully-populated `LayerMatrix` from a random PSD `(n, n)` dense."""
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())
    M = _random_psd(jax.random.PRNGKey(seed), n)
    return LayerMatrix.from_dense(M, param_groups=groups, layer_shapes=shapes), M, shapes, groups


def test_layer_matrix_from_dense_to_dense_roundtrip():
    lmat, M, _, _ = _random_dense_layer_matrix()
    assert jnp.allclose(lmat.to_dense(), M, atol=1e-5)


def test_layer_matrix_from_dense_is_not_block_diagonal():
    lmat, _, _, _ = _random_dense_layer_matrix()
    assert not lmat.is_block_diagonal()


def test_layer_matrix_from_dense_blocks_have_correct_shapes():
    lmat, _, shapes, groups = _random_dense_layer_matrix()
    for gi in groups:
        I_i, O_i = shapes[gi]
        for gj in groups:
            I_j, O_j = shapes[gj]
            block = lmat.blocks[(gi, gj)]
            assert isinstance(block, DenseBlock)
            assert block.matrix.shape == (I_i * O_i, I_j * O_j)
            assert block.row_shape == shapes[gi]
            assert block.col_shape == shapes[gj]


def test_layer_matrix_from_dense_matvec_matches_dense():
    lmat, M, shapes, groups = _random_dense_layer_matrix()
    n = M.shape[0]
    flat = jax.random.normal(jax.random.PRNGKey(33), (n,))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    y_fast = (lmat @ lvec).to_flat()
    y_dense = M @ flat
    assert jnp.allclose(y_fast, y_dense, atol=1e-4, rtol=1e-4)


def test_layer_matrix_from_dense_matvec_batched():
    lmat, M, shapes, groups = _random_dense_layer_matrix()
    n = M.shape[0]
    B = 4
    flat = jax.random.normal(jax.random.PRNGKey(34), (B, n))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    y_fast = (lmat @ lvec).to_flat()
    y_dense = flat @ M.T
    assert y_fast.shape == (B, n)
    assert jnp.allclose(y_fast, y_dense, atol=1e-4, rtol=1e-4)


def test_layer_matrix_from_dense_inverse_roundtrip():
    lmat, M, shapes, groups = _random_dense_layer_matrix()
    n = M.shape[0]
    flat = jax.random.normal(jax.random.PRNGKey(35), (n,))
    lvec = LayerVector.from_flat(flat, shapes=shapes, param_groups=groups)

    inv = lmat.inverse()
    round_trip = (inv @ (lmat @ lvec)).to_flat()
    assert jnp.allclose(round_trip, flat, atol=1e-3, rtol=1e-3)


def test_layer_matrix_from_dense_damped_dense_matches_dense_plus_identity():
    lmat, M, _, _ = _random_dense_layer_matrix()
    damping = 0.37
    out = lmat.damped(damping).to_dense()
    expected = M + damping * jnp.eye(M.shape[0])
    assert jnp.allclose(out, expected, atol=1e-5)


def test_layer_matrix_from_dense_size_mismatch_raises():
    shapes = _multi_layer_shapes()
    groups = list(shapes.keys())
    n = sum(I * O for I, O in shapes.values())
    bad = jnp.zeros((n + 1, n + 1))
    with pytest.raises(ValueError):
        LayerMatrix.from_dense(bad, param_groups=groups, layer_shapes=shapes)


def test_layer_matrix_from_dense_is_pytree():
    lmat, M, _, _ = _random_dense_layer_matrix()
    leaves, treedef = jax.tree_util.tree_flatten(lmat)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, LayerMatrix)
    assert jnp.allclose(reconstructed.to_dense(), M, atol=1e-5)
