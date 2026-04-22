from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.config import RegularizationStrategy
from src.hessians.layer_matrix import KroneckerFactors, LayerMatrix, LayerVector
from src.hessians.utils.data import (
    DataActivationsGradients,
    ModelContext,
)
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric

logger = logging.getLogger(__name__)


def _iter_slices(N: int, batch_size: Optional[int]):
    """Yield slices of length ≤ batch_size covering range(N)."""
    if batch_size is None or batch_size >= N:
        yield slice(0, N)
        return
    for start in range(0, N, batch_size):
        yield slice(start, min(start + batch_size, N))


def _accumulate_chunks(N: int, batch_size: Optional[int], chunk_fn):
    """Sum per-chunk pytrees returned by `chunk_fn(slice)`.

    Works uniformly on arrays, dicts, and nested dicts via `jax.tree.map`.
    """
    total = None
    for sl in _iter_slices(N, batch_size):
        contrib = chunk_fn(sl)
        total = contrib if total is None else jax.tree.map(jnp.add, total, contrib)
    return total


@dataclass
class HessianEstimator(ABC):
    """
    Base class for Hessian approximators.  Every subclass overrides `_build`
    to return a fully materialized `LayerMatrix`; the `estimate_*` public API
    is a thin wrapper over that matrix.
    """

    compute_context: ModelContext | DataActivationsGradients

    is_built: bool = False
    layer_matrix: Optional[LayerMatrix] = None
    layer_matrix_directory: Optional[str] = None

    # ---- build / load ----
    def build(
        self, base_directory: Optional[str] = None, try_load: bool = True
    ) -> "HessianEstimator":
        """
        Scaffolding to check if a built `LayerMatrix` already exists on disk, and if not, build it and save it.
        """
        if self.is_built:
            return self

        directory_path: Optional[str] = None
        if base_directory is not None:
            if self.layer_matrix_directory is None:
                directory_name = (
                    type(self).__name__.lower().replace("computer", "_layer_matrix")
                )
            else:
                directory_name = self.layer_matrix_directory
            directory_path = f"{base_directory}/{directory_name}"

        if (
            directory_path is not None
            and try_load
            and LayerMatrix.exists(directory_path)
        ):
            self.layer_matrix = LayerMatrix.load(directory_path)
            self.is_built = True
            logger.info(f"Loaded LayerMatrix from directory: {directory_path}")
        else:
            self.layer_matrix = self._build()
            if directory_path is not None and self.layer_matrix is not None:
                self.layer_matrix.save(directory=directory_path)
                logger.info(f"Saved LayerMatrix to directory: {directory_path}")
            self.is_built = True
        return self

    @abstractmethod
    def _build(self) -> LayerMatrix:
        """Produce the `LayerMatrix` this estimator approximates."""

    # ---- estimate_* public API ----
    def _require_built(self, op: str) -> LayerMatrix:
        if not self.is_built or self.layer_matrix is None:
            raise RuntimeError(
                f"HessianEstimator not built. Please call `.build()` "
                f"before {op}."
            )
        return self.layer_matrix

    def estimate_hessian(
        self, damping: Optional[Float] = None
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full `(n_params, n_params)` Hessian approximation."""
        M = self._require_built("estimating the Hessian")
        d = 0.0 if damping is None else damping
        return M.damped(d).to_dense()

    def estimate_inverse_hessian(
        self,
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "n_params n_params"]:
        """Compute the full `(n_params, n_params)` inverse Hessian approximation."""
        if pseudo_inverse_factor is not None and damping is not None:
            raise ValueError(
                "Cannot use both damping and pseudo-inverse factor simultaneously."
            )
        M = self._require_built("estimating the inverse Hessian")
        d = 0.0 if damping is None else damping
        p = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        return M.inverse(damping=d, pseudo_inverse_factor=p).to_dense()

    def compare_full_hessian_estimates(
        self,
        comparison_matrix: Float[Array, "n_params n_params"],
        damping: Optional[Float] = None,
        metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS,
    ) -> Float:
        """Compare the estimated matrix against a reference matrix."""
        return metric.compute(comparison_matrix, self.estimate_hessian(damping))

    def estimate_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute the Hessian-vector product `(H + dI) @ v` for each row of `vectors`.

        If `.build()` has been called, uses the cached `LayerMatrix`.  Otherwise
        dispatches to `_lazy_hvp` — subclasses with a true JVP-through-the-model
        path (`HessianComputer`, `GNHComputer`) override it to skip the
        `(n_params, n_params)` materialization entirely.  Estimators without a
        lazy path raise `NotImplementedError` and require `.build()` first.
        """
        d = 0.0 if damping is None else damping
        if self.is_built and self.layer_matrix is not None:
            M = self.layer_matrix
            lvec = LayerVector.from_flat(
                jnp.asarray(vectors),
                shapes=M.layer_shapes,
                param_groups=M.param_groups,
            )
            return (M.damped(d) @ lvec).to_flat()
        return self._lazy_hvp(jnp.asarray(vectors), d)

    def _lazy_hvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Float,
    ) -> Float[Array, "*batch_size n_params"]:
        """Lazy HVP hook — override in estimators with a JVP-through-the-model path."""
        raise NotImplementedError(
            f"{type(self).__name__} has no lazy HVP path — call `.build()` "
            f"before `estimate_hvp`."
        )

    def estimate_ihvp(
        self,
        vectors: Float[Array, "*batch_size n_params"],
        damping: Optional[Float] = None,
        pseudo_inverse_factor: Optional[float] = None,
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute the inverse-Hessian-vector product `(H + dI)^{-1} @ v`.

        Always requires `.build()` — there's no lazy IHVP path (materializing
        the dense matrix + solve is how every estimator does it today).
        """
        if pseudo_inverse_factor is not None and damping is not None:
            raise ValueError(
                "Cannot use both damping and pseudo-inverse factor simultaneously."
            )
        M = self._require_built("estimating the inverse-Hessian-vector product")
        d = 0.0 if damping is None else damping
        p = 0.0 if pseudo_inverse_factor is None else pseudo_inverse_factor
        lvec = LayerVector.from_flat(
            jnp.asarray(vectors),
            shapes=M.layer_shapes,
            param_groups=M.param_groups,
        )
        return (
            M.inverse(damping=d, pseudo_inverse_factor=p) @ lvec
        ).to_flat()


@dataclass
class KroneckerEstimator(HessianEstimator):
    """
    Base for Kronecker-factored Hessian approximators (KFAC / EKFAC /
    Shampoo / EShampoo / ...).

    Assembles a block-diagonal `LayerMatrix` of `KroneckerFactors(Q_A, Q_G,
    Lambda, lambda_A?, lambda_G?)` blocks.  Subclasses supply only two
    hooks describing how the raw covariance factors A, G are estimated from
    activations/gradients:

      * `_cov_chunk_sums` — unnormalized per-chunk outer-product sums
      * `_finalize_covariances` — normalization applied once to the full sums

    Everything else (chunked accumulation, eigendecomposition, optional
    eigenvalue correction, block assembly, damping) is shared.

    Two `DataActivationsGradients` contexts:
    - `compute_context` (inherited): data for the covariance factors A, G.
    - `corr_context`: data for the eigenvalue correction Λ.

    For deterministic collection (EMPIRICAL_FISHER, ALL_CLASSES) the caller
    should pass the same object for both.  For MCMC, two independent
    collector runs (different rng keys) should be passed so the eigenvalue
    correction isn't fit on the same samples as the covariances.

    `apply_eigenvalue_correction=False` collapses Λ to `outer(λ_A, λ_G)`
    (the non-corrected variant — i.e. KFAC / Shampoo rather than their
    E-counterparts).

    Gradient layout: `(N, O, k)` where k is the number of pseudo-target
    draws.  `probs` shape `(N, k)` carries per-draw weights (ones for
    EF/MCMC, softmax(logits) for ALL_CLASSES).
    """

    corr_context: Optional[DataActivationsGradients] = field(default=None)
    batch_size: Optional[int] = field(default=None)
    apply_eigenvalue_correction: bool = field(default=True)

    def __post_init__(self):
        if self.apply_eigenvalue_correction and self.corr_context is None:
            raise ValueError(
                f"{type(self).__name__} requires an explicit `corr_context` "
                "when `apply_eigenvalue_correction=True`.  For deterministic "
                "pseudo-target strategies (EMPIRICAL_FISHER, ALL_CLASSES) "
                "there is no sampling noise, so pass "
                "`corr_context=compute_context` to reuse the same data.  "
                "For MCMC, pass an independent collector run (different rng "
                "key) — otherwise the eigenvalue correction overfits the same "
                "samples used to estimate the eigenvectors."
            )

    def get_layer_names(self) -> List[str]:
        """Get the list of layer names from the covariance context."""
        return self.compute_context.layer_names

    # ------------------------------------------------------------------
    # _build — produce the LayerMatrix
    # ------------------------------------------------------------------

    def _build(self) -> LayerMatrix:
        """Assemble the block-diagonal LayerMatrix of KroneckerFactors blocks."""
        cov_data = self.compute_context
        assert isinstance(cov_data, DataActivationsGradients)

        (
            activation_eigvecs,
            gradient_eigvecs,
            activation_eigvals,
            gradient_eigvals,
        ) = self._compute_eigendecomposition(cov_data)

        lambdas = self._compute_lambdas(
            compute_context=self.corr_context,
            activation_eigvecs=activation_eigvecs,
            gradient_eigvecs=gradient_eigvecs,
            activation_eigvals=activation_eigvals,
            gradient_eigvals=gradient_eigvals,
        )

        layer_names = list(cov_data.layer_names)
        diag_blocks: Dict[str, KroneckerFactors] = {
            layer: KroneckerFactors.from_eigendecomposition(
                Q_A=activation_eigvecs[layer],
                Q_G=gradient_eigvecs[layer],
                Lambda=lambdas[layer],
                lambda_A=activation_eigvals[layer],
                lambda_G=gradient_eigvals[layer],
            )
            for layer in layer_names
        }
        return LayerMatrix.block_diagonal(
            diag_blocks=diag_blocks, param_groups=layer_names
        )

    # ------------------------------------------------------------------
    # Eigendecomposition
    # ------------------------------------------------------------------

    def _compute_eigendecomposition(
        self, data: DataActivationsGradients
    ) -> Tuple[
        Dict[str, Float[Array, "I I"]],
        Dict[str, Float[Array, "O O"]],
        Dict[str, Float[Array, "I"]],
        Dict[str, Float[Array, "O"]],
    ]:
        """Compute `(Q_A, Q_G, λ_A, λ_G)` per layer from the covariance data."""
        logger.info("Computing covariances for Kronecker approximation.")
        covariances = self._compute_covariances(
            activations_dict=data.activations,
            gradients_dict=data.gradients,
            probs=data.probs,
            batch_size=self.batch_size,
        )
        (
            (activation_eigvecs, gradient_eigvecs),
            (activation_eigvals, gradient_eigvals),
        ) = self.compute_eigenvectors_and_eigenvalues(
            covariances["activation_cov"], covariances["gradient_cov"]
        )
        logger.info("Computed eigenvectors for Kronecker approximation.")
        return (
            activation_eigvecs,
            gradient_eigvecs,
            activation_eigvals,
            gradient_eigvals,
        )

    @staticmethod
    def compute_eigenvectors_and_eigenvalues(
        activations_covariances: Dict[str, jnp.ndarray],
        gradients_covariances: Dict[str, jnp.ndarray],
    ) -> Tuple[
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ]:
        """Compute eigenvectors of the covariance matrices A and G for each layer."""
        activation_eigvals: Dict[str, jnp.ndarray] = {}
        activation_eigvecs: Dict[str, jnp.ndarray] = {}
        gradient_eigvals: Dict[str, jnp.ndarray] = {}
        gradient_eigvecs: Dict[str, jnp.ndarray] = {}

        for layer_name in activations_covariances.keys():
            (
                activation_eigvals_layer,
                gradient_eigvals_layer,
                activation_eigvecs_layer,
                gradient_eigvecs_layer,
            ) = KroneckerEstimator.compute_layer_eigenvectors(
                activations_covariances[layer_name],
                gradients_covariances[layer_name],
            )
            activation_eigvals[layer_name] = activation_eigvals_layer
            activation_eigvecs[layer_name] = activation_eigvecs_layer
            gradient_eigvals[layer_name] = gradient_eigvals_layer
            gradient_eigvecs[layer_name] = gradient_eigvecs_layer

        return (
            (activation_eigvecs, gradient_eigvecs),
            (activation_eigvals, gradient_eigvals),
        )

    @staticmethod
    @jax.jit
    def compute_layer_eigenvectors(
        A: Float[Array, "I I"], G: Float[Array, "O O"]
    ) -> Tuple[
        Float[Array, "I"],
        Float[Array, "O"],
        Float[Array, "I I"],
        Float[Array, "O O"],
    ]:
        """Compute eigenvectors of covariance matrices A and G for a single layer.

        eigh on near-rank-deficient covariances is unreliable in float32;
        promote to float64 for the decomposition and cast back to orig_dtype.
        """
        orig_dtype = A.dtype
        eigenvals_A, eigvecs_A = jnp.linalg.eigh(A.astype(jnp.float64))
        eigenvals_G, eigvecs_G = jnp.linalg.eigh(G.astype(jnp.float64))
        return (
            eigenvals_A.astype(orig_dtype),
            eigenvals_G.astype(orig_dtype),
            eigvecs_A.astype(orig_dtype),
            eigvecs_G.astype(orig_dtype),
        )

    # ------------------------------------------------------------------
    # Eigenvalue correction Λ (or KFAC short-circuit)
    # ------------------------------------------------------------------

    def _compute_lambdas(
        self,
        compute_context: Optional[DataActivationsGradients],
        activation_eigvecs: Dict[str, Float[Array, "I I"]],
        gradient_eigvecs: Dict[str, Float[Array, "O O"]],
        activation_eigvals: Dict[str, Float[Array, "I"]],
        gradient_eigvals: Dict[str, Float[Array, "O"]],
    ) -> Dict[str, Float[Array, "I O"]]:
        """Per-layer `Λ` used as the `KroneckerFactors.Lambda` field.

        Corrected: `Λ[i, o] = (Σ p_nk · (Q_A^T a)_{n,i}^2 (Q_G^T g)_{n,o,k}^2) / Σp`.
        Uncorrected (`apply_eigenvalue_correction=False`): `Λ = outer(λ_A, λ_G)`.
        """
        if not self.apply_eigenvalue_correction:
            return {
                layer: jnp.outer(activation_eigvals[layer], gradient_eigvals[layer])
                for layer in activation_eigvals.keys()
            }
        assert compute_context is not None
        probs = compute_context.probs
        total_prob = probs.sum()
        layer_names = list(compute_context.layer_names)
        N = probs.shape[0]

        def _chunk(sl: slice):
            return {
                layer: self._compute_layer_lambda_chunk_sum(
                    compute_context.activations[layer][sl],
                    compute_context.gradients[layer][sl],
                    probs[sl],
                    activation_eigvecs[layer],
                    gradient_eigvecs[layer],
                )
                for layer in layer_names
            }

        summed = _accumulate_chunks(N, self.batch_size, _chunk)
        logger.info("Computed eigenvalue corrections for Kronecker approximation.")
        return {layer: summed[layer] / total_prob for layer in layer_names}

    @staticmethod
    @jax.jit
    def _compute_layer_lambda_chunk_sum(
        a: Float[Array, "N I"],
        g: Float[Array, "N O k"],
        probs: Float[Array, "N k"],
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
    ) -> Float[Array, "I O"]:
        """Unnormalized chunk contribution to Λ:
            Σ_{n∈chunk} Σ_k p[n,k] · a_tilde_{n,i}^2 · g_tilde_{n,o,k}^2.

        Since ((Q_A^T a)_i * (Q_G^T g)_o)^2 = a_tilde[i]^2 * g_tilde[o]^2,
        we square first and contract with an einsum.
        """
        a_tilde_sq = (a @ Q_A) ** 2  # (N, I) — Q_A is orthogonal so a @ Q_A = (Q_A^T a)^T
        g_tilde_sq = jnp.einsum("op,npk->nok", Q_G.T, g) ** 2  # (N, O, k)
        return jnp.einsum("ni,nok,nk->io", a_tilde_sq, g_tilde_sq, probs)

    # ------------------------------------------------------------------
    # Covariance pipeline — chunked wrapper dispatching to subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    def _compute_covariances(
        cls,
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Chunked covariance computation.

        Accumulates unnormalized chunk sums from `_cov_chunk_sums` and calls
        `_finalize_covariances` to normalize and apply any variant-specific
        post-processing (trace-scaling, etc.).
        """
        layer_names = list(activations_dict.keys())
        N = activations_dict[layer_names[0]].shape[0]
        total_prob = probs.sum()

        def _chunk(sl: slice):
            return cls._cov_chunk_sums(
                {layer: activations_dict[layer][sl] for layer in layer_names},
                {layer: gradients_dict[layer][sl] for layer in layer_names},
                probs[sl],
            )

        summed = _accumulate_chunks(N, batch_size, _chunk)
        return cls._finalize_covariances(
            summed["act"], summed["grad"], N, total_prob
        )

    @staticmethod
    @abstractmethod
    def _cov_chunk_sums(
        activations_dict: Dict[str, Float[Array, "N I"]],
        gradients_dict: Dict[str, Float[Array, "N O k"]],
        probs: Float[Array, "N k"],
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Unnormalized per-chunk outer-product sums.  Returns
        `{"act": {layer: (I,I)}, "grad": {layer: (O,O)}}`.
        """

    @staticmethod
    @abstractmethod
    def _finalize_covariances(
        act_sums: Dict[str, Float[Array, "I I"]],
        grad_sums: Dict[str, Float[Array, "O O"]],
        N: int,
        total_prob: Float,
    ) -> Dict[str, Dict[str, Float[Array, "D D"]]]:
        """Normalize the accumulated chunk sums into `{activation_cov, gradient_cov}`."""

    # ------------------------------------------------------------------
    # Damping
    # ------------------------------------------------------------------

    def get_damping(
        self,
        damping_strategy: RegularizationStrategy,
        factor: float,
    ) -> float:
        """Get damping value based on strategy.

        Reads per-layer statistics from the `KroneckerFactors` blocks of the
        built `LayerMatrix`:

        * `AUTO_MEAN_EIGENVALUE` → average over layers of
          `mean(λ_A) * mean(λ_G)`, the product of the raw activation and
          gradient covariance eigenvalue means.  For the uncorrected variant
          this equals `mean(Lambda)` because `Lambda = outer(λ_A, λ_G)`, but
          for the corrected variant the eigenvalue correction diverges and
          this strategy reflects the *uncorrected* basis.
        * `AUTO_MEAN_EIGENVALUE_CORRECTION` → average over layers of
          `mean(block.Lambda)`, i.e. the eigenvalue-corrected Λ (equivalent
          to `AUTO_MEAN_EIGENVALUE` for the uncorrected variant, but the
          correction-aware version for the corrected variant).
        """
        if damping_strategy == RegularizationStrategy.FIXED:
            return factor
        if damping_strategy not in (
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE,
            RegularizationStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
        ):
            raise ValueError(f"Unsupported regularization strategy: {damping_strategy}")
        if self.layer_matrix is None:
            raise RuntimeError(
                f"{type(self).__name__} is not built — call `.build()` before `.get_damping()`."
            )
        means = []
        for layer in self.get_layer_names():
            block = self.layer_matrix.blocks[(layer, layer)]
            assert isinstance(block, KroneckerFactors)
            if damping_strategy == RegularizationStrategy.AUTO_MEAN_EIGENVALUE:
                if block.lambda_A is None or block.lambda_G is None:
                    raise RuntimeError(
                        f"Layer '{layer}' KroneckerFactors block is missing "
                        f"`lambda_A` / `lambda_G` — AUTO_MEAN_EIGENVALUE "
                        f"requires the raw covariance eigenvalues.  Rebuild "
                        f"with a Kronecker pipeline, which populates them."
                    )
                means.append(jnp.mean(block.lambda_A) * jnp.mean(block.lambda_G))
            else:  # AUTO_MEAN_EIGENVALUE_CORRECTION
                means.append(jnp.mean(block.Lambda))
        if not means:
            return 0.0
        aggregated = jnp.mean(jnp.stack(means))
        return float(aggregated) * factor
