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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
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
    def matvec(
        self, v: Float[Array, "... I O"]
    ) -> Float[Array, "... I O"]:
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

    All ops exploit the Kronecker structure — matvec is O(I^2 O + I O^2)
    instead of O((I O)^2), and no (I*O, I*O) matrix is ever materialized.
    """

    Q_A: Float[Array, "I I"]
    Q_G: Float[Array, "O O"]
    Lambda: Float[Array, "I O"]

    # ---- Pytree ----
    def tree_flatten(self):
        return (self.Q_A, self.Q_G, self.Lambda), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Q_A, Q_G, Lambda = children
        return cls(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)

    # ---- LayerBlock interface ----
    @property
    def shape(self) -> Tuple[int, int]:
        n = self.Q_A.shape[0] * self.Q_G.shape[0]
        return (n, n)

    @property
    def vector_shape(self) -> Tuple[int, int]:
        return (self.Q_A.shape[0], self.Q_G.shape[0])

    def matvec(
        self, v: Float[Array, "... I O"]
    ) -> Float[Array, "... I O"]:
        """(Q_A ⊗ Q_G) diag(Λ) (Q_A ⊗ Q_G)^T vec(V) via the reshape trick.

        Equivalent to: reshape vec(V) to (I, O), then
            Q_A @ (Λ ⊙ (Q_A.T @ V @ Q_G)) @ Q_G.T
        """
        v_tilde = self.Q_A.T @ v @ self.Q_G
        scaled = v_tilde * self.Lambda
        return self.Q_A @ scaled @ self.Q_G.T

    def matmat(self, other: "LayerBlock") -> "KroneckerFactors":
        """Multiply two Kronecker blocks.

        Requires `other` to be a `KroneckerFactors` sharing the same
        eigenbasis (same `Q_A`, `Q_G` arrays).  This is always true within
        a single estimator pipeline (e.g. `M` and `M.inverse()`).
        """
        if not isinstance(other, KroneckerFactors):
            raise TypeError(
                "KroneckerFactors.matmat expects another KroneckerFactors"
            )
        return KroneckerFactors(
            Q_A=self.Q_A,
            Q_G=self.Q_G,
            Lambda=self.Lambda * other.Lambda,
        )

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
        return KroneckerFactors(Q_A=self.Q_A, Q_G=self.Q_G, Lambda=inv)

    def damped(self, damping: Float) -> "KroneckerFactors":
        """block + damping*I.  In the eigenbasis: Lambda + damping."""
        return KroneckerFactors(
            Q_A=self.Q_A, Q_G=self.Q_G, Lambda=self.Lambda + damping
        )

    def to_dense(self) -> Float[Array, "n n"]:
        """Materialize the (I*O, I*O) block.  Only used for tests / small problems."""
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
        return cls(Q_A=Q_A, Q_G=Q_G, Lambda=jnp.outer(lam_A, lam_G))

    @classmethod
    def from_eigendecomposition(
        cls,
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
    ) -> "KroneckerFactors":
        """Build directly from precomputed eigenvectors and Lambda."""
        return cls(Q_A=Q_A, Q_G=Q_G, Lambda=Lambda)


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

    def matvec(
        self, v: Float[Array, "... I_j O_j"]
    ) -> Float[Array, "... I_i O_i"]:
        """Apply the block to a vector shaped `(..., I_j, O_j)`."""
        I_j, O_j = self.col_shape
        I_i, O_i = self.row_shape
        batch_shape = v.shape[:-2]
        flat = v.reshape(*batch_shape, I_j * O_j)              # (..., n_j)
        y = flat @ self.matrix.T                                # (..., n_i)
        return y.reshape(*batch_shape, I_i, O_i)

    def matmat(self, other: "LayerBlock") -> "DenseBlock":
        if not isinstance(other, DenseBlock):
            raise TypeError(
                "DenseBlock.matmat expects another DenseBlock"
            )
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
        M = self.matrix + damping * jnp.eye(n, dtype=self.matrix.dtype)
        if pseudo_inverse_factor is not None and pseudo_inverse_factor > 0.0:
            eigvals, eigvecs = jnp.linalg.eigh(0.5 * (M + M.T))
            eigvals_inv = jnp.where(
                jnp.abs(eigvals) > pseudo_inverse_factor, 1.0 / eigvals, 0.0
            )
            M_inv = (eigvecs * eigvals_inv) @ eigvecs.T
        else:
            M_inv = jnp.linalg.inv(M)
        return DenseBlock(
            matrix=M_inv, row_shape=self.row_shape, col_shape=self.col_shape
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
        return cls(
            blocks=blocks, param_groups=param_groups, layer_shapes=layer_shapes
        )

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
        return cls(
            blocks=blocks, param_groups=param_groups, layer_shapes=layer_shapes
        )

    # ---- introspection ----
    def diagonal_blocks(self) -> Dict[str, LayerBlock]:
        """Return `{group: block}` for diagonal entries only."""
        return {g: self.blocks[(g, g)] for g in self.param_groups if (g, g) in self.blocks}

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
        raise TypeError(
            f"LayerMatrix @ {type(other).__name__} is not supported"
        )

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
        if not (self.is_block_diagonal() and other.is_block_diagonal()):
            raise NotImplementedError(
                "LayerMatrix matmat is only implemented for block-diagonal matrices"
            )
        out_blocks: Dict[Tuple[str, str], LayerBlock] = {
            (g, g): self.blocks[(g, g)].matmat(other.blocks[(g, g)])
            for g in self.param_groups
        }
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

    def __add__(self, other: "LayerMatrix") -> "LayerMatrix":
        if not (self.is_block_diagonal() and other.is_block_diagonal()):
            raise NotImplementedError(
                "LayerMatrix __add__ is only implemented for block-diagonal matrices"
            )
        out_blocks: Dict[Tuple[str, str], LayerBlock] = {}
        for g in self.param_groups:
            a = self.blocks[(g, g)]
            b = other.blocks[(g, g)]
            if isinstance(a, KroneckerFactors) and isinstance(b, KroneckerFactors):
                out_blocks[(g, g)] = KroneckerFactors(
                    Q_A=a.Q_A, Q_G=a.Q_G, Lambda=a.Lambda + b.Lambda
                )
            else:
                raise TypeError(
                    "LayerMatrix addition only supported for KroneckerFactors blocks"
                )
        return LayerMatrix(
            blocks=out_blocks,
            param_groups=self.param_groups,
            layer_shapes=self.layer_shapes,
        )

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
        return LayerMatrix.from_dense(
            M_inv, self.param_groups, self.layer_shapes
        )

    def to_dense(self) -> Float[Array, "n n"]:
        """Materialize the full `(n_params, n_params)` matrix."""
        if self.is_block_diagonal():
            dense_blocks = [
                self.blocks[(g, g)].to_dense() for g in self.param_groups
            ]
            return jax.scipy.linalg.block_diag(*dense_blocks)
        # Fully-populated `(i, j)` grid: assemble row-by-row.
        rows = []
        for gi in self.param_groups:
            row_pieces = [
                self.blocks[(gi, gj)].to_dense() for gj in self.param_groups
            ]
            rows.append(jnp.concatenate(row_pieces, axis=1))
        return jnp.concatenate(rows, axis=0)
