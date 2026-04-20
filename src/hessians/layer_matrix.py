"""
LayerVector / LayerMatrix abstractions for block-structured Hessian approximations.

A `LayerMatrix` is a block matrix whose blocks are keyed by pairs of
*param groups* (layer names).  Each block is a `LayerBlock` subclass that
chooses its own storage format; the top-level container stays storage-agnostic
and dispatches per-block operations.

For the (E)KFAC family we use `KroneckerFactors`, which stores each diagonal
block in the eigendecomposed Kronecker form

    block = (Q_A ⊗ Q_G) diag(vec(Lambda)) (Q_A ⊗ Q_G)^T

and implements matvec via the reshape trick `Q_A @ (Λ ⊙ (Q_A.T @ V @ Q_G)) @ Q_G.T`
so that the (I*O, I*O) matrix is never materialized.

`LayerVector` stores the per-group vector as `(..., I, O)` so that the
Kronecker matvec is a zero-copy per-group op.  `to_flat` / `from_flat` give
interop with the existing `(*batch, n_params)` API.

All classes are registered as JAX pytrees so they play nicely with
`jax.jit` / `jax.vmap`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# LayerBlock — abstract base
# ---------------------------------------------------------------------------


class LayerBlock(ABC):
    """
    Single (n_i, n_j) block of a LayerMatrix.  Concrete subclasses pick a
    storage format (Kronecker-factored, dense, diagonal, ...).

    `matvec` / `matmat` / `inverse` / `damped` / `to_dense` dispatch to the
    subclass implementation so that `LayerMatrix` stays storage-agnostic.
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Dense shape (n_i, n_j) of the block."""

    @property
    @abstractmethod
    def vector_shape(self) -> Tuple[int, int]:
        """Shape `(I, O)` such that a compatible vector is `(..., I, O)`."""

    @abstractmethod
    def matvec(self, v: Float[Array, "... I O"]) -> Float[Array, "... I O"]:
        """Apply the block to a vector shaped `(..., I, O)`."""

    @abstractmethod
    def matmat(self, other: "LayerBlock") -> "LayerBlock":
        """Multiply two blocks of compatible shape."""

    @abstractmethod
    def inverse(
        self,
        damping: Float = 0.0,
        pseudo_inverse_factor: Float = 0.0,
    ) -> "LayerBlock":
        """Return (block + damping*I)^{-1}, optionally with pseudo-inverse."""

    @abstractmethod
    def damped(self, damping: Float) -> "LayerBlock":
        """Return block + damping*I."""

    @abstractmethod
    def to_dense(self) -> Float[Array, "n n"]:
        """Materialize the dense (n_i*n_j, n_i*n_j) matrix."""

    # ---- Persistence ----
    # Every concrete LayerBlock subclass advertises a BLOCK_TYPE string and
    # implements `to_disk_dict` / `from_disk_dict` for `LayerMatrix.save`/`load`.

    BLOCK_TYPE: ClassVar[str]

    @abstractmethod
    def to_disk_dict(self) -> Dict[str, np.ndarray]:
        """Return a `{name: np.ndarray}` dict that fully describes this block."""

    @classmethod
    @abstractmethod
    def from_disk_dict(cls, d: Dict[str, np.ndarray]) -> "LayerBlock":
        """Inverse of `to_disk_dict`."""


# ---------------------------------------------------------------------------
# KroneckerFactors — concrete block stored in eigendecomposed Kronecker form
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class KroneckerFactors(LayerBlock):
    """
    A (I*O, I*O) block stored as the eigendecomposed Kronecker form:

        block = (Q_A ⊗ Q_G) diag(vec(Lambda)) (Q_A ⊗ Q_G)^T

    - EKFAC uses this with `Lambda = eigenvalue_corrections`.
    - KFAC  uses this with `Lambda = outer(λ_A, λ_G)`.

    `lambda_A` / `lambda_G` are the eigenvalues of the raw activation /
    gradient covariances `A` / `G` whose eigenvectors populate `Q_A` / `Q_G`.
    They describe the basis itself (not the current `Lambda`) so they stay
    meaningful under the operations that preserve `Q_A` / `Q_G` (damping,
    inverse, arithmetic) even when the resulting `Lambda` is no longer
    `outer(lambda_A, lambda_G)`.  They are optional (legacy blocks and
    direct test-level construction may omit them), but the EKFAC / KFAC
    pipeline always populates them so downstream consumers — factor-wise
    matmul, spectrum diagnostics, KFAC-style reconstruction of `A` / `G` —
    can rely on their presence.

    All ops exploit the Kronecker structure — matvec is O(I^2 O + I O^2)
    instead of O((I O)^2), and no (I*O, I*O) matrix is ever materialized.
    """

    Q_A: Float[Array, "I I"]
    Q_G: Float[Array, "O O"]
    Lambda: Float[Array, "I O"]
    lambda_A: Optional[Float[Array, "I"]] = None
    lambda_G: Optional[Float[Array, "O"]] = None

    # ---- Pytree ----
    def tree_flatten(self):
        # `lambda_A` / `lambda_G` are dynamic children (may be arrays) but
        # None is a valid pytree subtree with zero leaves — so blocks with
        # and without them have different treedefs and cannot be mixed in a
        # single jit trace.  That matches the intended usage: any single
        # LayerMatrix is built with a consistent convention.
        return (self.Q_A, self.Q_G, self.Lambda, self.lambda_A, self.lambda_G), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Q_A, Q_G, Lambda, lambda_A, lambda_G = children
        return cls(
            Q_A=Q_A,
            Q_G=Q_G,
            Lambda=Lambda,
            lambda_A=lambda_A,
            lambda_G=lambda_G,
        )

    # ---- LayerBlock interface ----
    @property
    def shape(self) -> Tuple[int, int]:
        n = self.Q_A.shape[0] * self.Q_G.shape[0]
        return (n, n)

    @property
    def vector_shape(self) -> Tuple[int, int]:
        return (self.Q_A.shape[0], self.Q_G.shape[0])

    def matvec(self, v: Float[Array, "... I O"]) -> Float[Array, "... I O"]:
        """(Q_A ⊗ Q_G) diag(Λ) (Q_A ⊗ Q_G)^T vec(V) via the reshape trick.

        Equivalent to: reshape vec(V) to (I, O), then
            Q_A @ (Λ ⊙ (Q_A.T @ V @ Q_G)) @ Q_G.T
        """
        v_tilde = self.Q_A.T @ v @ self.Q_G
        scaled = v_tilde * self.Lambda
        return self.Q_A @ scaled @ self.Q_G.T

    def matmat(self, other: "LayerBlock") -> "LayerBlock":
        """Multiply two Kronecker blocks.

        Fast path: `other` is a `KroneckerFactors` sharing the same
        eigenbasis (same `Q_A`, `Q_G` arrays).  This is always true within
        a single estimator pipeline (e.g. `M` and `M.inverse()`) and the
        result stays in eigendecomposed form.  Otherwise falls back to
        materializing both sides and returning a `DenseBlock`.
        """
        if isinstance(other, KroneckerFactors) and self._same_basis(other):
            return KroneckerFactors(
                Q_A=self.Q_A,
                Q_G=self.Q_G,
                Lambda=self.Lambda * other.Lambda,
                lambda_A=self.lambda_A,
                lambda_G=self.lambda_G,
            )
        return _fallback_block_matmat(self, other)

    def _same_basis(
        self,
        other: "KroneckerFactors",
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> bool:
        """Approximate eigenbasis-equality check for closed-form Kronecker arithmetic.

        Uses `np.allclose` rather than object identity — the same basis can flow
        through `tree_unflatten` / `jit` and come out as distinct array objects,
        so `is` would miss it.  The shape check short-circuits the common
        "different layers / different-sized bases" case for free.
        """
        if self.Q_A.shape != other.Q_A.shape or self.Q_G.shape != other.Q_G.shape:
            return False
        return bool(
            np.allclose(
                np.asarray(self.Q_A), np.asarray(other.Q_A), atol=atol, rtol=rtol
            )
        ) and bool(
            np.allclose(
                np.asarray(self.Q_G), np.asarray(other.Q_G), atol=atol, rtol=rtol
            )
        )

    # ---- arithmetic ----
    #
    # Every operation below preserves `Q_A` and `Q_G`, so it also preserves
    # `lambda_A` / `lambda_G` (which describe the basis itself, not the
    # current `Lambda`).  We thread them through unchanged.
    def __add__(self, other: "LayerBlock") -> "LayerBlock":
        if isinstance(other, KroneckerFactors) and self._same_basis(other):
            return KroneckerFactors(
                Q_A=self.Q_A,
                Q_G=self.Q_G,
                Lambda=self.Lambda + other.Lambda,
                lambda_A=self.lambda_A,
                lambda_G=self.lambda_G,
            )
        return _fallback_block_add(self, other)

    def __sub__(self, other: "LayerBlock") -> "LayerBlock":
        if isinstance(other, KroneckerFactors) and self._same_basis(other):
            return KroneckerFactors(
                Q_A=self.Q_A,
                Q_G=self.Q_G,
                Lambda=self.Lambda - other.Lambda,
                lambda_A=self.lambda_A,
                lambda_G=self.lambda_G,
            )
        return _fallback_block_add(self, -other)

    def __neg__(self) -> "KroneckerFactors":
        return KroneckerFactors(
            Q_A=self.Q_A,
            Q_G=self.Q_G,
            Lambda=-self.Lambda,
            lambda_A=self.lambda_A,
            lambda_G=self.lambda_G,
        )

    def __mul__(self, scalar: Float) -> "KroneckerFactors":
        return KroneckerFactors(
            Q_A=self.Q_A,
            Q_G=self.Q_G,
            Lambda=self.Lambda * scalar,
            lambda_A=self.lambda_A,
            lambda_G=self.lambda_G,
        )

    __rmul__ = __mul__

    def inverse(
        self,
        damping: Float = 0.0,
        pseudo_inverse_factor: Float = 0.0,
    ) -> "KroneckerFactors":
        """Per-eigenvalue inverse, optionally with pseudo-inverse threshold."""
        if pseudo_inverse_factor is not None and pseudo_inverse_factor > 0.0:
            inv = jnp.where(
                jnp.abs(self.Lambda) > pseudo_inverse_factor,
                1.0 / (self.Lambda + damping),
                0.0,
            )
        else:
            inv = 1.0 / (self.Lambda + damping)
        return KroneckerFactors(
            Q_A=self.Q_A,
            Q_G=self.Q_G,
            Lambda=inv,
            lambda_A=self.lambda_A,
            lambda_G=self.lambda_G,
        )

    def damped(self, damping: Float) -> "KroneckerFactors":
        """block + damping*I.  In the eigenbasis: Lambda + damping."""
        return KroneckerFactors(
            Q_A=self.Q_A,
            Q_G=self.Q_G,
            Lambda=self.Lambda + damping,
            lambda_A=self.lambda_A,
            lambda_G=self.lambda_G,
        )

    def to_dense(self) -> Float[Array, "n n"]:
        """Materialize the (I*O, I*O) block."""
        Q = jnp.kron(self.Q_A, self.Q_G)
        return Q @ jnp.diag(self.Lambda.flatten()) @ Q.T

    # ---- factories ----
    @classmethod
    def from_raw_factors(
        cls,
        A: Float[Array, "I I"],
        G: Float[Array, "O O"],
    ) -> "KroneckerFactors":
        """Build from raw A, G covariance factors (KFAC-style inputs)."""
        lam_A, Q_A = jnp.linalg.eigh(A)
        lam_G, Q_G = jnp.linalg.eigh(G)
        return cls(
            Q_A=Q_A,
            Q_G=Q_G,
            Lambda=jnp.outer(lam_A, lam_G),
            lambda_A=lam_A,
            lambda_G=lam_G,
        )

    @classmethod
    def from_eigendecomposition(
        cls,
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
        lambda_A: Optional[Float[Array, "I"]] = None,
        lambda_G: Optional[Float[Array, "O"]] = None,
    ) -> "KroneckerFactors":
        """Build directly from precomputed eigenvectors and Lambda.

        `lambda_A` / `lambda_G` are optional: pass them when the caller has
        the raw activation/gradient covariance eigenvalues (the EKFAC / KFAC
        pipeline does).  They describe the basis itself and survive any op
        that preserves `Q_A` / `Q_G`.
        """
        return cls(
            Q_A=Q_A,
            Q_G=Q_G,
            Lambda=Lambda,
            lambda_A=lambda_A,
            lambda_G=lambda_G,
        )

    # ---- Persistence ----
    BLOCK_TYPE: ClassVar[str] = "kronecker"

    def to_disk_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "Q_A": np.asarray(self.Q_A),
            "Q_G": np.asarray(self.Q_G),
            "Lambda": np.asarray(self.Lambda),
        }
        if self.lambda_A is not None:
            out["lambda_A"] = np.asarray(self.lambda_A)
        if self.lambda_G is not None:
            out["lambda_G"] = np.asarray(self.lambda_G)
        return out

    @classmethod
    def from_disk_dict(cls, d: Dict[str, np.ndarray]) -> "KroneckerFactors":
        # `lambda_A` / `lambda_G` are optional for backward compat with
        # blocks saved before they were tracked.
        lambda_A = jnp.asarray(d["lambda_A"]) if "lambda_A" in d else None
        lambda_G = jnp.asarray(d["lambda_G"]) if "lambda_G" in d else None
        return cls(
            Q_A=jnp.asarray(d["Q_A"]),
            Q_G=jnp.asarray(d["Q_G"]),
            Lambda=jnp.asarray(d["Lambda"]),
            lambda_A=lambda_A,
            lambda_G=lambda_G,
        )


# ---------------------------------------------------------------------------
# DenseBlock — concrete block stored as a dense materialized matrix
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class DenseBlock(LayerBlock):
    """
    A dense materialized `(n_i, n_j)` block.

    `row_shape` / `col_shape` record the `(I, O)` reshapes between flat
    `(n_i,)` / `(n_j,)` views and the natural per-layer shapes.  For
    diagonal blocks `row_shape == col_shape`; for off-diagonal blocks
    they may differ.
    """

    matrix: Float[Array, "n_i n_j"]
    row_shape: Tuple[int, int] = field(metadata={"static": True})
    col_shape: Tuple[int, int] = field(metadata={"static": True})

    # ---- Pytree ----
    def tree_flatten(self):
        return (self.matrix,), (self.row_shape, self.col_shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (matrix,) = children
        row_shape, col_shape = aux_data
        return cls(matrix=matrix, row_shape=row_shape, col_shape=col_shape)

    # ---- LayerBlock interface ----
    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape

    @property
    def vector_shape(self) -> Tuple[int, int]:
        return self.row_shape

    def matvec(self, v: Float[Array, "... I_j O_j"]) -> Float[Array, "... I_i O_i"]:
        """Apply the block to a vector shaped `(..., I_j, O_j)`."""
        I_j, O_j = self.col_shape
        I_i, O_i = self.row_shape
        batch_shape = v.shape[:-2]
        flat = v.reshape(*batch_shape, I_j * O_j)  # (..., n_j)
        y = flat @ self.matrix.T  # (..., n_i)
        return y.reshape(*batch_shape, I_i, O_i)

    def matmat(self, other: "LayerBlock") -> "DenseBlock":
        if isinstance(other, DenseBlock):
            if self.col_shape != other.row_shape:
                raise ValueError(
                    f"Shape mismatch: self.col_shape={self.col_shape} "
                    f"vs other.row_shape={other.row_shape}"
                )
            return DenseBlock(
                matrix=self.matrix @ other.matrix,
                row_shape=self.row_shape,
                col_shape=other.col_shape,
            )
        return _fallback_block_matmat(self, other)

    def inverse(
        self,
        damping: Float = 0.0,
        pseudo_inverse_factor: Float = 0.0,
    ) -> "DenseBlock":
        """Return `(block + damping*I)^{-1}`, optionally with pseudo-inverse."""
        if self.row_shape != self.col_shape:
            raise ValueError(
                "DenseBlock.inverse only defined for square diagonal blocks"
            )
        n = self.matrix.shape[0]
        orig_dtype = self.matrix.dtype
        # eigh on a near-rank-deficient matrix is unreliable in float32; always
        # run the decomposition in float64 and cast the result back.
        M = self.matrix.astype(jnp.float64) + jnp.float64(damping) * jnp.eye(
            n, dtype=jnp.float64
        )
        if pseudo_inverse_factor is not None and pseudo_inverse_factor > 0.0:
            eigvals, eigvecs = jnp.linalg.eigh(0.5 * (M + M.T))
            eigvals_inv = jnp.where(
                jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
            )
            M_inv = (eigvecs * eigvals_inv) @ eigvecs.T
        else:
            M_inv = jnp.linalg.inv(M)
        return DenseBlock(
            matrix=M_inv.astype(orig_dtype),
            row_shape=self.row_shape,
            col_shape=self.col_shape,
        )

    def damped(self, damping: Float) -> "DenseBlock":
        if self.row_shape != self.col_shape:
            raise ValueError(
                "DenseBlock.damped only defined for square diagonal blocks"
            )
        n = self.matrix.shape[0]
        return DenseBlock(
            matrix=self.matrix + damping * jnp.eye(n, dtype=self.matrix.dtype),
            row_shape=self.row_shape,
            col_shape=self.col_shape,
        )

    def to_dense(self) -> Float[Array, "n_i n_j"]:
        return self.matrix

    # ---- arithmetic ----
    def __add__(self, other: "LayerBlock") -> "LayerBlock":
        if isinstance(other, DenseBlock):
            if self.row_shape != other.row_shape or self.col_shape != other.col_shape:
                raise ValueError(
                    f"DenseBlock addition: shape mismatch "
                    f"({self.row_shape}, {self.col_shape}) vs "
                    f"({other.row_shape}, {other.col_shape})"
                )
            return DenseBlock(
                matrix=self.matrix + other.matrix,
                row_shape=self.row_shape,
                col_shape=self.col_shape,
            )
        return _fallback_block_add(self, other)

    def __sub__(self, other: "LayerBlock") -> "LayerBlock":
        if isinstance(other, DenseBlock):
            if self.row_shape != other.row_shape or self.col_shape != other.col_shape:
                raise ValueError(
                    f"DenseBlock subtraction: shape mismatch "
                    f"({self.row_shape}, {self.col_shape}) vs "
                    f"({other.row_shape}, {other.col_shape})"
                )
            return DenseBlock(
                matrix=self.matrix - other.matrix,
                row_shape=self.row_shape,
                col_shape=self.col_shape,
            )
        return _fallback_block_add(self, -other)

    def __neg__(self) -> "DenseBlock":
        return DenseBlock(
            matrix=-self.matrix,
            row_shape=self.row_shape,
            col_shape=self.col_shape,
        )

    def __mul__(self, scalar: Float) -> "DenseBlock":
        return DenseBlock(
            matrix=self.matrix * scalar,
            row_shape=self.row_shape,
            col_shape=self.col_shape,
        )

    __rmul__ = __mul__

    # ---- Persistence ----
    BLOCK_TYPE: ClassVar[str] = "dense"

    def to_disk_dict(self) -> Dict[str, np.ndarray]:
        return {
            "matrix": np.asarray(self.matrix),
            "row_shape": np.asarray(self.row_shape, dtype=np.int64),
            "col_shape": np.asarray(self.col_shape, dtype=np.int64),
        }

    @classmethod
    def from_disk_dict(cls, d: Dict[str, np.ndarray]) -> "DenseBlock":
        return cls(
            matrix=jnp.asarray(d["matrix"]),
            row_shape=tuple(int(x) for x in np.asarray(d["row_shape"]).tolist()),
            col_shape=tuple(int(x) for x in np.asarray(d["col_shape"]).tolist()),
        )


# Registry of concrete LayerBlock subclasses keyed by BLOCK_TYPE.  Used by
# LayerMatrix.load to instantiate the right subclass from a manifest.
_BLOCK_TYPE_REGISTRY: Dict[str, type] = {
    KroneckerFactors.BLOCK_TYPE: KroneckerFactors,
    DenseBlock.BLOCK_TYPE: DenseBlock,
}


# ---------------------------------------------------------------------------
# Cross-type block arithmetic fallbacks
#
# When two blocks don't share a storage type (e.g. adding a KroneckerFactors
# to a DenseBlock), we materialize both sides and return a DenseBlock.  This
# is the pragmatic degradation path so `LayerMatrix` arithmetic / matmat
# works uniformly across storage modes.
# ---------------------------------------------------------------------------


def _block_row_shape(b: LayerBlock) -> Tuple[int, int]:
    """`(I, O)` vector shape of rows — `row_shape` for Dense, `vector_shape` otherwise."""
    if isinstance(b, DenseBlock):
        return b.row_shape
    return b.vector_shape


def _block_col_shape(b: LayerBlock) -> Tuple[int, int]:
    """`(I, O)` vector shape of columns — `col_shape` for Dense, `vector_shape` otherwise."""
    if isinstance(b, DenseBlock):
        return b.col_shape
    return b.vector_shape


def _fallback_block_add(a: LayerBlock, b: LayerBlock) -> DenseBlock:
    """Materialize both blocks and sum them into a `DenseBlock`."""
    if not isinstance(b, LayerBlock):
        raise TypeError(
            f"LayerBlock addition expects another LayerBlock, got {type(b).__name__}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"LayerBlock addition: dense shape mismatch {a.shape} vs {b.shape}"
        )
    return DenseBlock(
        matrix=a.to_dense() + b.to_dense(),
        row_shape=_block_row_shape(a),
        col_shape=_block_col_shape(a),
    )


def _fallback_block_matmat(a: LayerBlock, b: LayerBlock) -> DenseBlock:
    """Materialize both blocks and multiply them into a `DenseBlock`."""
    if not isinstance(b, LayerBlock):
        raise TypeError(
            f"LayerBlock matmat expects another LayerBlock, got {type(b).__name__}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"LayerBlock matmat: inner dim mismatch {a.shape} @ {b.shape}")
    return DenseBlock(
        matrix=a.to_dense() @ b.to_dense(),
        row_shape=_block_row_shape(a),
        col_shape=_block_col_shape(b),
    )


# ---------------------------------------------------------------------------
# LayerVector — per-param-group dict of (..., I, O) arrays
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class LayerVector:
    """
    Per-param-group dict of arrays shaped like each group's weight matrix.

    `blocks[group_name]` has shape `(I_l, O_l)` — or `(*batch, I_l, O_l)`
    when batched over multiple vectors.  `param_groups` defines the canonical
    ordering for `to_flat` / `from_flat`.
    """

    blocks: Dict[str, Float[Array, "... I O"]]
    param_groups: List[str]

    # ---- Pytree ----
    def tree_flatten(self):
        children = tuple(self.blocks[name] for name in self.param_groups)
        aux = tuple(self.param_groups)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        param_groups = list(aux)
        blocks = {name: child for name, child in zip(param_groups, children)}
        return cls(blocks=blocks, param_groups=param_groups)

    # ---- construction ----
    @classmethod
    def from_flat(
        cls,
        flat: Float[Array, "... n_params"],
        shapes: Dict[str, Tuple[int, int]],
        param_groups: List[str],
    ) -> "LayerVector":
        """Split a flat `(..., n_params)` array into per-group `(..., I, O)` blocks."""
        blocks: Dict[str, Array] = {}
        offset = 0
        for name in param_groups:
            I, O = shapes[name]
            size = I * O
            chunk = flat[..., offset : offset + size]
            blocks[name] = chunk.reshape(chunk.shape[:-1] + (I, O))
            offset += size
        return cls(blocks=blocks, param_groups=param_groups)

    def to_flat(self) -> Float[Array, "... n_params"]:
        """Row-major flatten each group and concatenate along the last axis."""
        pieces = []
        for name in self.param_groups:
            block = self.blocks[name]
            batch_shape = block.shape[:-2]
            flat_size = block.shape[-2] * block.shape[-1]
            pieces.append(block.reshape(*batch_shape, flat_size))
        return jnp.concatenate(pieces, axis=-1)

    def shapes(self) -> Dict[str, Tuple[int, int]]:
        """Per-group `(I, O)` shapes (ignoring any leading batch axes)."""
        return {
            name: (self.blocks[name].shape[-2], self.blocks[name].shape[-1])
            for name in self.param_groups
        }

    # ---- ops ----
    def __add__(self, other: "LayerVector") -> "LayerVector":
        return LayerVector(
            blocks={
                name: self.blocks[name] + other.blocks[name]
                for name in self.param_groups
            },
            param_groups=self.param_groups,
        )

    def __sub__(self, other: "LayerVector") -> "LayerVector":
        return LayerVector(
            blocks={
                name: self.blocks[name] - other.blocks[name]
                for name in self.param_groups
            },
            param_groups=self.param_groups,
        )

    def __mul__(self, scalar: Float) -> "LayerVector":
        return LayerVector(
            blocks={name: self.blocks[name] * scalar for name in self.param_groups},
            param_groups=self.param_groups,
        )

    __rmul__ = __mul__

    def __neg__(self) -> "LayerVector":
        return self * -1.0


# ---------------------------------------------------------------------------
# LayerMatrix — block matrix over param groups
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass
class LayerMatrix:
    """
    Block matrix over param groups.  Keys are `(group_i, group_j)` pairs;
    values are `LayerBlock`s.

    Two storage modes are supported:

    - **Block-diagonal**: only `(l, l)` diagonal keys are populated.
      Used by KFAC / EKFAC / TKFAC / ... and by per-layer FIM/Hessian
      block estimators.  All ops dispatch through the per-block methods
      and avoid materializing the full matrix.

    - **Fully populated `(i, j)` grid**: every `(g_i, g_j)` pair is
      populated, typically with `DenseBlock`s sliced from a materialized
      `(n_params, n_params)` matrix via `LayerMatrix.from_dense(...)`.
      Used by FIM, GNH, and the exact Hessian.  All ops still go through
      the per-block methods, so the layer-dict view is preserved.

    `LayerMatrix` is strictly a *materialized* matrix abstraction: every
    instance has a full dense representation that `to_dense()` returns.
    Lazy HVP paths (JVP/VJP through the model) live outside `LayerMatrix`.
    """

    blocks: Dict[Tuple[str, str], LayerBlock]
    param_groups: List[str]
    layer_shapes: Dict[str, Tuple[int, int]]

    # ---- Pytree ----
    def tree_flatten(self):
        keys = sorted(self.blocks.keys())
        children = tuple(self.blocks[k] for k in keys)
        aux = (
            tuple(keys),
            tuple(self.param_groups),
            tuple((g, self.layer_shapes[g]) for g in self.param_groups),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        keys, param_groups, layer_shapes_items = aux
        blocks = {k: child for k, child in zip(keys, children)}
        layer_shapes = {g: tuple(s) for g, s in layer_shapes_items}
        return cls(
            blocks=blocks,
            param_groups=list(param_groups),
            layer_shapes=layer_shapes,
        )

    # ---- constructors ----
    @classmethod
    def block_diagonal(
        cls,
        diag_blocks: Dict[str, LayerBlock],
        param_groups: List[str],
        layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> "LayerMatrix":
        """Build a block-diagonal `LayerMatrix` from `{group: block}`."""
        if layer_shapes is None:
            layer_shapes = {g: diag_blocks[g].vector_shape for g in param_groups}
        blocks = {(g, g): diag_blocks[g] for g in param_groups}
        return cls(blocks=blocks, param_groups=param_groups, layer_shapes=layer_shapes)

    @classmethod
    def from_dense(
        cls,
        dense: Float[Array, "n_params n_params"],
        param_groups: List[str],
        layer_shapes: Dict[str, Tuple[int, int]],
    ) -> "LayerMatrix":
        """
        Slice a materialized `(n_params, n_params)` matrix into per-layer
        `DenseBlock`s for *every* `(i, j)` pair.

        Used by estimators that compute the full matrix (FIM, GNH, exact
        Hessian) and want to present it as a per-layer block dict matching
        the KFAC layer-dict convention.
        """
        offsets: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for g in param_groups:
            I, O = layer_shapes[g]
            offsets[g] = (offset, offset + I * O)
            offset += I * O
        if not (offset == dense.shape[0] == dense.shape[1]):
            raise ValueError(
                f"Sum of layer sizes ({offset}) does not match dense matrix "
                f"shape {dense.shape}"
            )

        blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for gi in param_groups:
            si, ei = offsets[gi]
            for gj in param_groups:
                sj, ej = offsets[gj]
                blocks[(gi, gj)] = DenseBlock(
                    matrix=dense[si:ei, sj:ej],
                    row_shape=layer_shapes[gi],
                    col_shape=layer_shapes[gj],
                )
        return cls(blocks=blocks, param_groups=param_groups, layer_shapes=layer_shapes)

    # ---- introspection ----
    def diagonal_blocks(self) -> Dict[str, LayerBlock]:
        """Return `{group: block}` for diagonal entries only."""
        return {
            g: self.blocks[(g, g)] for g in self.param_groups if (g, g) in self.blocks
        }

    def is_block_diagonal(self) -> bool:
        return all(i == j for (i, j) in self.blocks.keys())

    def vector_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Per-group `(I, O)` vector shapes."""
        return self.layer_shapes

    # ---- ops ----
    def __matmul__(
        self, other: Union["LayerVector", "LayerMatrix"]
    ) -> Union["LayerVector", "LayerMatrix"]:
        if isinstance(other, LayerVector):
            return self._matvec(other)
        if isinstance(other, LayerMatrix):
            return self._matmat(other)
        raise TypeError(f"LayerMatrix @ {type(other).__name__} is not supported")

    def _matvec(self, v: LayerVector) -> LayerVector:
        """Apply `M @ v` block-by-block: y[i] = Σ_j M[i,j] @ v[j]."""
        out_blocks: Dict[str, Float[Array, "..."]] = {}
        for gi in self.param_groups:
            acc = None
            for gj in self.param_groups:
                if (gi, gj) not in self.blocks:
                    continue
                contrib = self.blocks[(gi, gj)].matvec(v.blocks[gj])
                acc = contrib if acc is None else acc + contrib
            if acc is None:
                # No blocks in this row — produce a zero block of the right shape.
                I_i, O_i = self.layer_shapes[gi]
                acc = jnp.zeros(v.blocks[self.param_groups[0]].shape[:-2] + (I_i, O_i))
            out_blocks[gi] = acc
        return LayerVector(blocks=out_blocks, param_groups=self.param_groups)

    def _matmat(self, other: "LayerMatrix") -> "LayerMatrix":
        """Block matmat: `C[i,j] = Σ_k A[i,k] @ B[k,j]`.

        Works for any combination of block-diagonal and fully-populated
        `(i, j)` grids: missing blocks on either side are treated as zero,
        so block-diag × block-diag stays block-diag, grid × grid stays a
        grid, and mixed products produce a grid.  Per-block matmat falls
        back to dense when storage types disagree.
        """
        if (
            self.param_groups != other.param_groups
            or self.layer_shapes != other.layer_shapes
        ):
            raise ValueError(
                "LayerMatrix matmat requires matching param_groups and layer_shapes"
            )
        out_blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for gi in self.param_groups:
            for gj in self.param_groups:
                acc: Optional[LayerBlock] = None
                for gk in self.param_groups:
                    if (gi, gk) not in self.blocks or (gk, gj) not in other.blocks:
                        continue
                    contrib = self.blocks[(gi, gk)].matmat(other.blocks[(gk, gj)])
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    out_blocks[(gi, gj)] = acc
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def __add__(self, other: "LayerMatrix") -> "LayerMatrix":
        """Block-wise `+`.  Missing blocks on either side are treated as zero,
        so a block-diagonal plus a grid yields a grid."""
        if (
            self.param_groups != other.param_groups
            or self.layer_shapes != other.layer_shapes
        ):
            raise ValueError(
                "LayerMatrix addition requires matching param_groups and layer_shapes"
            )
        out_blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for key in set(self.blocks.keys()) | set(other.blocks.keys()):
            if key in self.blocks and key in other.blocks:
                out_blocks[key] = self.blocks[key] + other.blocks[key]
            elif key in self.blocks:
                out_blocks[key] = self.blocks[key]
            else:
                out_blocks[key] = other.blocks[key]
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def __sub__(self, other: "LayerMatrix") -> "LayerMatrix":
        """Block-wise `-`.  Delegates to per-block `__sub__` / `__neg__`."""
        if (
            self.param_groups != other.param_groups
            or self.layer_shapes != other.layer_shapes
        ):
            raise ValueError(
                "LayerMatrix subtraction requires matching param_groups and layer_shapes"
            )
        out_blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for key in set(self.blocks.keys()) | set(other.blocks.keys()):
            if key in self.blocks and key in other.blocks:
                out_blocks[key] = self.blocks[key] - other.blocks[key]
            elif key in self.blocks:
                out_blocks[key] = self.blocks[key]
            else:
                out_blocks[key] = -other.blocks[key]
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def __neg__(self) -> "LayerMatrix":
        return LayerMatrix(
            blocks={k: -b for k, b in self.blocks.items()},
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def __mul__(self, scalar: Float) -> "LayerMatrix":
        return LayerMatrix(
            blocks={k: b * scalar for k, b in self.blocks.items()},
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    __rmul__ = __mul__

    def damped(self, damping: Float) -> "LayerMatrix":
        """Return `M + damping*I`.  Damping adds to diagonal blocks only."""
        out_blocks: Dict[Tuple[str, str], LayerBlock] = dict(self.blocks)
        for g in self.param_groups:
            out_blocks[(g, g)] = self.blocks[(g, g)].damped(damping)
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def inverse(
        self,
        damping: Float = 0.0,
        pseudo_inverse_factor: Float = 0.0,
    ) -> "LayerMatrix":
        """
        Per-block inverse for block-diagonal matrices, or solve-via-dense
        for the fully-populated `(i, j)` grid case.
        """
        if self.is_block_diagonal():
            out_blocks: Dict[Tuple[str, str], LayerBlock] = {
                (g, g): self.blocks[(g, g)].inverse(
                    damping=damping, pseudo_inverse_factor=pseudo_inverse_factor
                )
                for g in self.param_groups
            }
            return LayerMatrix(
                blocks=out_blocks,
                param_groups=self.param_groups,
                layer_shapes=self.layer_shapes,
            )
        # Fully-populated grid: the inverse of a non-block-diagonal matrix
        # is not itself sparse, so go through dense (assemble → solve → re-slice).
        dense = self.to_dense()
        n = dense.shape[0]
        M = dense + damping * jnp.eye(n, dtype=dense.dtype)
        if pseudo_inverse_factor is not None and pseudo_inverse_factor > 0.0:
            eigvals, eigvecs = jnp.linalg.eigh(0.5 * (M + M.T))
            eigvals_inv = jnp.where(
                jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
            )
            M_inv = (eigvecs * eigvals_inv) @ eigvecs.T
        else:
            M_inv = jnp.linalg.inv(M)
        return LayerMatrix.from_dense(M_inv, self.param_groups, self.layer_shapes)

    def to_dense(self) -> Float[Array, "n n"]:
        """Materialize the full `(n_params, n_params)` matrix."""
        if self.is_block_diagonal():
            dense_blocks = [self.blocks[(g, g)].to_dense() for g in self.param_groups]
            return jax.scipy.linalg.block_diag(*dense_blocks)
        # Fully-populated `(i, j)` grid: assemble row-by-row.
        rows = []
        for gi in self.param_groups:
            row_pieces = [self.blocks[(gi, gj)].to_dense() for gj in self.param_groups]
            rows.append(jnp.concatenate(row_pieces, axis=1))
        return jnp.concatenate(rows, axis=0)

    # ---- Persistence ----
    # On disk:
    #   {directory}/manifest.json          — param_groups, layer_shapes,
    #                                         and block metadata (keys + block type)
    #   {directory}/blocks.npz             — one flattened array per
    #                                         (block_key, field_name) pair
    _MANIFEST_FILENAME: ClassVar[str] = "manifest.json"
    _BLOCKS_FILENAME: ClassVar[str] = "blocks.npz"
    _KEY_DELIMITER: ClassVar[str] = "::"

    @classmethod
    def _encode_block_key(cls, key: Tuple[str, str]) -> str:
        gi, gj = key
        if cls._KEY_DELIMITER in gi or cls._KEY_DELIMITER in gj:
            raise ValueError(
                f"Layer names must not contain the reserved delimiter "
                f"'{cls._KEY_DELIMITER}': got {key!r}"
            )
        return f"{gi}{cls._KEY_DELIMITER}{gj}"

    @classmethod
    def _decode_block_key(cls, s: str) -> Tuple[str, str]:
        gi, gj = s.split(cls._KEY_DELIMITER)
        return (gi, gj)

    def save(self, directory: Union[str, Path]) -> None:
        """Persist this `LayerMatrix` to `directory` (created if missing)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        block_entries: List[Dict[str, object]] = []
        for key in sorted(self.blocks.keys()):
            block = self.blocks[key]
            encoded_key = self._encode_block_key(key)
            field_names: List[str] = []
            for field_name, array in block.to_disk_dict().items():
                arrays[f"{encoded_key}//{field_name}"] = array
                field_names.append(field_name)
            block_entries.append(
                {
                    "key": encoded_key,
                    "block_type": block.BLOCK_TYPE,
                    "fields": field_names,
                }
            )

        manifest = {
            "param_groups": list(self.param_groups),
            "layer_shapes": {g: list(self.layer_shapes[g]) for g in self.param_groups},
            "blocks": block_entries,
        }
        with open(directory / self._MANIFEST_FILENAME, "w") as f:
            json.dump(manifest, f, indent=2)
        np.savez_compressed(directory / self._BLOCKS_FILENAME, **arrays)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "LayerMatrix":
        directory = Path(directory)
        with open(directory / cls._MANIFEST_FILENAME, "r") as f:
            manifest = json.load(f)
        npz = np.load(directory / cls._BLOCKS_FILENAME)

        param_groups: List[str] = list(manifest["param_groups"])
        layer_shapes: Dict[str, Tuple[int, int]] = {
            g: tuple(int(x) for x in manifest["layer_shapes"][g]) for g in param_groups
        }

        blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for entry in manifest["blocks"]:
            encoded_key: str = entry["key"]
            block_type: str = entry["block_type"]
            field_names: List[str] = entry["fields"]
            key = cls._decode_block_key(encoded_key)
            block_cls = _BLOCK_TYPE_REGISTRY[block_type]
            disk_dict = {
                field_name: npz[f"{encoded_key}//{field_name}"]
                for field_name in field_names
            }
            blocks[key] = block_cls.from_disk_dict(disk_dict)

        return cls(
            blocks=blocks,
            param_groups=param_groups,
            layer_shapes=layer_shapes,
        )

    @staticmethod
    def exists(directory: Union[str, Path]) -> bool:
        directory = Path(directory)
        return (directory / LayerMatrix._MANIFEST_FILENAME).is_file() and (
            directory / LayerMatrix._BLOCKS_FILENAME
        ).is_file()
