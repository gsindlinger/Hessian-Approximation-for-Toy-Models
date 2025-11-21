from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Literal, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing_extensions import override

from config.config import Config
from config.hessian_approximation_config import (
    KFACBuildConfig,
    KFACConfig,
    KFACRunConfig,
)
from data.jax_dataloader import JAXDataLoader
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.hessian_approximations import HessianApproximation
from hessian_approximations.kfac.activation_gradient_collector import (
    ActivationGradientCollector,
)
from hessian_approximations.kfac.data import (
    KFACData,
    KFACJITData,
    KFACMeanEigenvaluesAndCorrections,
)
from hessian_approximations.kfac.layer_components import LayerComponents
from hessian_approximations.kfac.storage import KFACStorage
from metrics.full_matrix_metrics import FullMatrixMetric
from models.dataclasses.hessian_jit_data import HessianJITData
from models.train import ApproximationModel
from models.utils.loss import get_loss_name
from utils.utils import print_device_memory_stats


@dataclass
class KFAC(HessianApproximation):
    """
    Kronecker-Factored Approximate Curvature (KFAC) and Eigenvalue-Corrected KFAC (EKFAC) Hessian approximation.
    """

    storage: KFACStorage = field(init=False)
    collector: ActivationGradientCollector = field(
        default_factory=ActivationGradientCollector
    )
    kfac_data: KFACData = field(default_factory=KFACData)
    kfac_data_means: KFACMeanEigenvaluesAndCorrections = field(
        default_factory=KFACMeanEigenvaluesAndCorrections
    )
    kfac_config: KFACConfig = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if not self.full_config.hessian_approximation:
            self.full_config.hessian_approximation = KFACConfig()

        self.storage = KFACStorage(config=self.full_config)

        if not isinstance(self.full_config.hessian_approximation, KFACConfig):
            raise ValueError(
                "KFAC Hessian approximation requires KFACConfig in the config."
            )
        self.kfac_config = self.full_config.hessian_approximation

    @classmethod
    def setup_with_run_and_build_config(
        cls,
        full_config: Config,
        run_config: KFACRunConfig | None = None,
        build_config: KFACBuildConfig | None = None,
    ) -> KFAC:
        """Setup KFAC instance with given run or build configuration.
        If either is None, the one from the initial config is used."""
        # create copy of config to ensure that a new instance is created
        full_config = copy.deepcopy(full_config)

        if not full_config.hessian_approximation:
            full_config.hessian_approximation = KFACConfig()
        elif not isinstance(full_config.hessian_approximation, KFACConfig):
            raise ValueError(
                "KFAC Hessian approximation requires KFACConfig in the config."
            )

        kfac_config = full_config.hessian_approximation

        if build_config is not None:
            kfac_config.build_config = build_config
        if run_config is not None:
            kfac_config.run_config = run_config
        full_config.hessian_approximation = kfac_config
        return cls(full_config=full_config)

    def damping(self) -> Float[Array, ""]:
        """Get damping value from config.

        Returns:
        Float[Array, ""]: Damping value.
        """
        if not self.kfac_data_means.overall_mean_eigenvalues:
            self.get_ekfac_components()

        mode = self.kfac_config.run_config.damping_mode
        lambda_ = self.kfac_config.run_config.damping_lambda
        if mode == "mean_eigenvalue":
            return lambda_ * self.kfac_data_means.overall_mean_eigenvalues
        elif mode == "mean_corrections":
            return lambda_ * self.kfac_data_means.overall_mean_corrections
        else:
            raise ValueError(
                f"Unknown damping mode: {self.kfac_config.run_config.damping_mode}"
            )

    def get_sample_size(self) -> int:
        """Get number of samples used in the collected data."""
        if not self.collector.captured_data:
            raise ValueError("No captured data. Run a forward-backward pass first.")

        first_layer = next(iter(self.collector.captured_data.values()))
        activations, _ = first_layer
        return activations.shape[0]

    @override
    def compute_hessian(self) -> Float[Array, "n_params n_params"]:
        """
        Compute full Hessian approximation.

        Not practical for large models but useful for testing and comparison
        with the true Hessian.
        """

        return self.compute_hessian_or_inverse_hessian(
            method="normal",
        )

    def compute_inverse_hessian(
        self,
    ) -> Float[Array, "n_params n_params"]:
        """
        Compute inverse Hessian approximation.
        """
        return self.compute_hessian_or_inverse_hessian(
            method="inverse",
        )

    def compare_hessian(
        self, metric: FullMatrixMetric = FullMatrixMetric.FROBENIUS
    ) -> float:
        """
        Compute L2 norm difference between KFAC/EKFAC Hessian and a comparison matrix.

        Args:
            comparison_matrix_method: Callable that returns the comparison matrix. Should be a jitted method.
        """
        self.get_ekfac_components()

        activations_eigenvectors, gradients_eigenvectors, Lambdas = (
            self.collect_eigenvectors_and_lambda()
        )
        kfac_jit_data = KFACJITData(
            activations_eigenvectors, gradients_eigenvectors, Lambdas
        )
        hessian_jit_data = HessianJITData.get_data_and_params_for_hessian(
            self.model_data
        )

        print_device_memory_stats(
            "After loading data for Hessian comparison, but before calling any hessian computation."
        )

        ground_truth_hessian = Hessian.compute_hessian_jitted(
            hessian_jit_data
        ) + self.damping() * jnp.eye(hessian_jit_data.num_params)

        print_device_memory_stats(
            "After computing ground truth Hessian, before comparing with KFAC Hessian."
        )

        result = self.compare_hessians_jitted_with_matrix_input(
            kfac_jit_data,
            self.damping(),
            ground_truth_hessian,
            metric=metric.compute_fn(),
            method="normal",
        )

        print_device_memory_stats("After comparing Hessians.")

        return result

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def compare_hessians_jitted_with_matrix_input(
        kfac_data: KFACJITData,
        damping: Float[Array, ""],
        ground_truth_hessian: Float[Array, "n_params n_params"],
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """JIT-compiled helper to compute L2 norm difference."""
        kfac_hessian = KFAC.compute_hessian_or_inverse_hessian_jitted(
            kfac_data.eigenvectors_A,
            kfac_data.eigenvectors_G,
            kfac_data.Lambdas,
            damping=damping,
            method=method,
        )

        return metric(kfac_hessian, ground_truth_hessian)

    @staticmethod
    @partial(jax.jit, static_argnames=["method", "metric"])
    def compare_hessians_jitted(
        kfac_data: KFACJITData,
        damping: Float[Array, ""],
        hessian_data: HessianJITData,
        metric: Callable[[jnp.ndarray, jnp.ndarray], float],
        method: Literal["normal", "inverse"] = "normal",
    ) -> float:
        """JIT-compiled helper to compute L2 norm difference."""
        kfac_hessian = KFAC.compute_hessian_or_inverse_hessian_jitted(
            kfac_data.eigenvectors_A,
            kfac_data.eigenvectors_G,
            kfac_data.Lambdas,
            damping=damping,
            method=method,
        )

        true_hessian = Hessian.compute_hessian_jitted(hessian_data) + damping * jnp.eye(
            kfac_hessian.shape[0]
        )
        return metric(kfac_hessian, true_hessian)

    def compute_hessian_or_inverse_hessian(
        self, method: Literal["normal", "inverse"]
    ) -> Float[Array, "n_params n_params"]:
        """
        Unified helper method to compute either the full Hessian or its inverse.

        Depending on configuration, this computes either:
        - (E)KFAC Hessians with eigenvalue corrections, or
        - Standard KFAC Hessians using Kronecker-factored covariances.

        Note:
        PyTorch assumes weights shaped [d_out, d_in] with vec(∇W) = a ⊗ ∇s
        (column-major). In contrast, JAX uses [d_in, d_out], which yields
        vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.
        """

        self.get_ekfac_components()
        activations_eigenvectors, gradients_eigenvectors, Lambdas = (
            self.collect_eigenvectors_and_lambda()
        )

        return self.compute_hessian_or_inverse_hessian_jitted(
            activations_eigenvectors,
            gradients_eigenvectors,
            Lambdas,
            self.damping(),
            method,
        )

    def collect_eigenvectors_and_lambda(
        self,
    ) -> Tuple[
        List[Float[Array, "I I"]], List[Float[Array, "O O"]], List[Float[Array, "I O"]]
    ]:
        """Collect eigenvectors and lambda values for all layers."""
        activations_eigenvectors = []
        gradients_eigenvectors = []
        Lambdas = []

        for layer_name in self.model_data.params["params"].keys():
            activations_eigenvectors.append(
                self.kfac_data.eigenvectors.activations[layer_name]
            )
            gradients_eigenvectors.append(
                self.kfac_data.eigenvectors.gradients[layer_name]
            )
            if self.kfac_config.run_config.use_eigenvalue_correction:
                Lambdas.append(self.kfac_data.eigenvalue_corrections[layer_name])
            else:
                Lambdas.append(self.compute_eigenvalue_lambda_kfac(layer_name))

        return activations_eigenvectors, gradients_eigenvectors, Lambdas

    @staticmethod
    def _get_damped_lambda(
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ) -> Float[Array, "n_params n_params"]:
        if method == "inverse":
            return 1.0 / (Lambda + damping)
        else:
            return Lambda + damping

    @staticmethod
    @jax.jit
    def _compute_layer_hessian(eigenvectors_A, eigenvectors_G, Lambda):
        """JIT-compiled helper to compute layer Hessian with eigenvalue corrections."""
        return jnp.einsum(
            "ij,j,jk->ik",
            jnp.kron(eigenvectors_A, eigenvectors_G),
            Lambda.flatten(),
            jnp.kron(eigenvectors_A, eigenvectors_G).T,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_hessian_or_inverse_hessian_jitted(
        eigenvectors_activations: List[Float[Array, "I I"]],
        eigenvectors_gradients: List[Float[Array, "O O"]],
        Lambdas: List[Float[Array, "I O"]],
        damping: Float[Array, ""],
        method: Literal["normal", "inverse"],
    ):
        """
        JIT-compiled method to compute layer Hessian or its inverse for KFAC or EKFAC
        based on a layer-based lists of the required components.

        Depending on the selected method:
        - "normal" computes the standard KFAC approximation with damping:
              H ≈ (Q_G ⊗ Q_A) diag(Λ_G ⊗ Λ_A + λ) (Q_G ⊗ Q_A)ᵀ
        - "inverse" computes its damped inverse:
              H⁻¹ ≈ (Q_G ⊗ Q_A) diag(1 / (Λ_G ⊗ Λ_A + λ)) (Q_G ⊗ Q_A)ᵀ

        Note: The original KFAC formulation assumes weights shaped [d_out, d_in]
        with vec(∇W) = a ⊗ ∇s (column-major). In contrast, JAX uses [d_in, d_out],
        which yields vec(∇W') = ∇s ⊗ a due to the forward pass formulation y = xW'
        instead of y = Wx (as in PyTorch).

        Because JAX flattens arrays in row-major (C-style) order, the effective
        vectorization swaps again, giving vec_row(∇W') = a ⊗ ∇s. This matches
        the ordering used when comparing with the true Hessian or constructing
        Kronecker-factored curvature blocks.

        Since we store the eigenvalues and corrections in the shape [input_dim, output_dim],
        we can directly use them here by flattening in JAX-default row-major order without needing to transpose.
        """

        hessian_list = [
            KFAC._compute_layer_hessian(
                layer_eigv_activations,
                layer_eigv_gradients,
                KFAC._get_damped_lambda(layer_lambda, damping, method),
            )
            for layer_eigv_activations, layer_eigv_gradients, layer_lambda in zip(
                eigenvectors_activations,
                eigenvectors_gradients,
                Lambdas,
            )
        ]

        return jax.scipy.linalg.block_diag(*hessian_list)

    @override
    def compute_hvp(
        self,
        vector: Float[Array, "*batch_size n_params"],
    ) -> Float[Array, "*batch_size n_params"]:
        """Compute Hessian-vector product."""
        return self.compute_ihvp_or_hvp(
            vector,
            method="hvp",
        )

    @override
    def compute_ihvp(
        self,
        vector: Float[Array, "*batch_size n_params"],
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product using EKFAC approximation.
        """
        return self.compute_ihvp_or_hvp(
            vector,
            method="ihvp",
        )

    def compute_ihvp_or_hvp(
        self,
        vector: Float[Array, "*batch_size n_params"],
        method: Literal["ihvp", "hvp"],
    ) -> Float[Array, "*batch_size n_params"]:
        """
        Compute inverse Hessian-vector product or Hessian-vector product.

        Note, that the vector to be multiplied is reshaped in row-major order to
        match the JAX weight layout which is reflected by
        the eigenvalue corrections shape [input_dim, output_dim].
        """
        # Ensure EKFAC components are computed
        self.get_ekfac_components()

        vp_pieces = []
        offset = 0

        for layer_name in self.model_data.params["params"].keys():
            Lambda: Float[Array, "I O"] = self.kfac_data.eigenvalue_corrections[
                layer_name
            ]
            input_dim, output_dim = Lambda.shape
            size = input_dim * output_dim

            # Extract the corresponding part of the vector
            v_flat: Float[Array, "*batch_size I*O"] = vector[
                ..., offset : offset + size
            ]

            # Reshape last two dimensions to [I, O] matching JAX weights shape convention (also Lambda)
            v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                v_flat.shape[:-1] + (input_dim, output_dim)
            )

            # If running KFAC-only, we use the eigenvalues of the covariances and not the corrections
            if not self.kfac_config.run_config.use_eigenvalue_correction:
                Lambda = self.compute_eigenvalue_lambda_kfac(layer_name)

            vp_piece: Float[Array, "*batch_size I O"] = self.compute_ihvp_or_hvp_layer(
                v_layer,
                self.kfac_data.eigenvectors.activations[layer_name],
                self.kfac_data.eigenvectors.gradients[layer_name],
                Lambda,
                self.damping(),
                method=method,
            )
            vp_pieces.append(vp_piece.reshape(vp_piece.shape[:-2] + (size,)))
            offset += size

        # Concatenate all layer HVPS
        return jnp.concatenate(vp_pieces, axis=-1)

    def get_ekfac_components(self) -> None:
        """
        Compute all EKFAC components in sequence.

        Steps:
        1. Compute covariances from activations and gradients
        2. Compute eigenvectors of covariance matrices
        3. Compute eigenvalue corrections
        4. Compute mean eigenvalues and corrections for damping

        If components are already stored on disk and recalculation is not forced,
        they will be reused.
        """

        # If the components are already there we don't need to recompute them
        if (
            self.kfac_data
            and self.kfac_data_means
            and (not self.kfac_config.recalc_ekfac_components)
        ):
            return

        # check if kfac components are computed, if so reuse
        if self.storage.check_storage() and not (
            self.kfac_config.recalc_ekfac_components
        ):
            print("Loading EKFAC components from disk.")
            self.kfac_data, self.kfac_data_means = self.storage.load_kfac_data()
        else:
            print("Computing EKFAC components from scratch.")
            training_data, training_targets = self.model_data.dataset.get_train_data()
            self.compute_from_scratch(
                self.model_data.model,
                self.model_data.params,
                training_data,
                training_targets,
                self.model_data.loss,
            )

    def compute_from_scratch(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ) -> None:
        """Compute all (E)KFAC components from scratch."""
        self.compute_covariances(
            model, params, training_data, training_targets, loss_fn
        )
        print("Computing eigenvectors for each layer.")
        self.compute_eigenvectors()
        print("Computing eigenvalue corrections for each layer.")
        self.compute_eigenvalue_corrections(
            model, params, training_data, training_targets, loss_fn
        )
        print("Computing mean eigenvalues and corrections for damping.")
        self.compute_mean_eigenvalues_and_corrections()
        print("Finished computing EKFAC components from scratch.")

    def compute_covariances(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ):
        """
        Compute A and G covariance matrices for each layer.
        Performs a forward-backward pass to collect activations and gradients,
        then computes their covariance matrices.
        """

        self._process_multiple_batches_collector(
            model,
            params,
            training_data,
            targets,
            loss_fn,
            compute_method="covariance",
        )
        self.storage.save_covariances(self.kfac_data.covariances)

    @staticmethod
    @jax.jit
    def compute_layer_eigenvectors(
        A: Float[Array, "I I"], G: Float[Array, "O O"]
    ) -> Tuple[
        Float[Array, "I"], Float[Array, "O"], Float[Array, "I I"], Float[Array, "O O"]
    ]:
        """
        Compute eigenvectors of covariance matrices A and G for a single layer.

        Returns:
            Tuple containing:
            - Eigenvalues of A (Float[Array, "I"])
            - Eigenvalues of G (Float[Array, "O"])
            - Eigenvectors of A (Float[Array, "I I"])
            - Eigenvectors of G (Float[Array, "O O"])
        """
        # Ensure numerical stability by using float64 for eigen decomposition
        jax.config.update("jax_enable_x64", True)
        A = A.astype(jnp.float64)
        G = G.astype(jnp.float64)

        eigenvals_A: Float[Array, "I"]
        eigvecs_A: Float[Array, "I I"]
        eigenvals_G: Float[Array, "O"]
        eigvecs_G: Float[Array, "O O"]

        eigenvals_A, eigvecs_A = jnp.linalg.eigh(A)
        eigenvals_G, eigvecs_G = jnp.linalg.eigh(G)
        jax.config.update("jax_enable_x64", False)

        return (
            eigenvals_A.astype(jnp.float32),
            eigenvals_G.astype(jnp.float32),
            eigvecs_A.astype(jnp.float32),
            eigvecs_G.astype(jnp.float32),
        )

    def compute_eigenvectors(self) -> None:
        """Compute eigenvectors of the covariance matrices A and G for each layer."""
        if not self.kfac_data.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )

        activation_eigvals = {}
        activation_eigvecs = {}
        gradient_eigvals = {}
        gradient_eigvecs = {}

        for layer_name in self.kfac_data.covariances.activations.keys():
            (
                activation_eigvals_layer,
                gradient_eigvals_layer,
                activation_eigvecs_layer,
                gradient_eigvecs_layer,
            ) = self.compute_layer_eigenvectors(
                self.kfac_data.covariances.activations[layer_name],
                self.kfac_data.covariances.gradients[layer_name],
            )

            activation_eigvals[layer_name] = activation_eigvals_layer
            activation_eigvecs[layer_name] = activation_eigvecs_layer
            gradient_eigvals[layer_name] = gradient_eigvals_layer
            gradient_eigvecs[layer_name] = gradient_eigvecs_layer

        self.kfac_data.eigenvalues = LayerComponents(
            activations=activation_eigvals,
            gradients=gradient_eigvals,
        )
        self.kfac_data.eigenvectors = LayerComponents(
            activations=activation_eigvecs,
            gradients=gradient_eigvecs,
        )
        self.storage.save_eigenvectors(self.kfac_data.eigenvectors)
        self.storage.save_eigenvalues(self.kfac_data.eigenvalues)

    def compute_eigenvalue_corrections(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        training_targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
    ) -> None:
        """
        Compute eigenvalue corrections for each layer.

        Projects activations and gradients onto eigenbases and computes
        the empirical eigenvalue corrections as outer products.
        """
        if not self.kfac_data.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )
        if not self.kfac_data.eigenvectors:
            raise ValueError(
                "Eigenvectors not computed yet. Run compute_eigenvectors_and_eigenvalues first."
            )

        self._process_multiple_batches_collector(
            model,
            params,
            training_data,
            training_targets,
            loss_fn,
            compute_method="eigenvalue_correction",
        )
        self.storage.save_eigenvalue_corrections(self.kfac_data.eigenvalue_corrections)

    def compute_mean_eigenvalues_and_corrections(self) -> None:
        """Compute mean eigenvalues and eigenvalue corrections for damping."""
        self.kfac_data_means.eigenvalues = {}
        self.kfac_data_means.corrections = {}

        overall_mean_eigenvalues = 0.0
        overall_mean_eigenvalue_corrections = 0.0

        for layer_name in self.kfac_data.eigenvalues.activations.keys():
            mean_eigenvalue: Float[Array, ""] = jnp.mean(
                self.compute_eigenvalue_lambda_kfac(layer_name)
            )
            self.kfac_data_means.eigenvalues[layer_name] = mean_eigenvalue
            overall_mean_eigenvalues += mean_eigenvalue

        for layer_name in self.kfac_data.eigenvalue_corrections.keys():
            mean_correction: Float[Array, ""] = jnp.mean(
                self.kfac_data.eigenvalue_corrections[layer_name]
            )
            self.kfac_data_means.corrections[layer_name] = mean_correction
            overall_mean_eigenvalue_corrections += mean_correction

        # Divide overall sums by number of layers
        n_layers = len(self.kfac_data.eigenvalues.activations)
        self.kfac_data_means.overall_mean_eigenvalues = jnp.array(
            overall_mean_eigenvalues / n_layers if n_layers > 0 else 0.0
        )
        self.kfac_data_means.overall_mean_corrections = jnp.array(
            overall_mean_eigenvalue_corrections / n_layers if n_layers > 0 else 0.0
        )

        self.storage.save_mean_eigenvalues_and_corrections(self.kfac_data_means)

    @staticmethod
    @jax.jit
    def covariance(
        input: Float[Array, "n_samples features"],
    ) -> Float[Array, "features features"]:
        """
        Compute covariance matrix for given input.

        For each layer:
        - A = (1/N) * a^T @ a
        - G = (1/N) * g^T @ g
        """
        return input.T @ input

    def _process_multiple_batches_collector(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        targets: Float[Array, "n_samples targets"] | Int[Array, "n_samples"],
        loss_fn: Callable,
        compute_method: Literal["covariance", "eigenvalue_correction"],
    ):
        """
        Process data in batches to collect activations and gradients and apply some function to it.
        So far it is used to compute covariances of the activations and preactivation gradients, as well as computing eigenvalue corrections.
        """

        def loss_fn_for_grad(p, training_data, targets):
            predictions = model.apply(
                p, training_data, self.collector, method=model.kfac_apply
            )
            # Use sum reduction to avoid prematurely averaging gradients
            return loss_fn(predictions, targets, reduction="sum")

        # Optionally generate pseudo-targets for true Fisher computation
        # Ensure different RNG keys for covariance and eigenvalue correction computations
        if self.kfac_config.build_config.use_pseudo_targets:
            if compute_method == "covariance":
                prng_key = jax.random.PRNGKey(42)
            else:
                prng_key = jax.random.PRNGKey(43)
            targets = self.generate_pseudo_targets(
                model, params, training_data, loss_fn, rng_key=prng_key
            )

        n_samples = training_data.shape[0]

        dataloader_batch_size = self.kfac_config.build_config.collector_batch_size
        dataloader = JAXDataLoader(
            data=training_data,
            targets=targets,
            shuffle=False,
            batch_size=dataloader_batch_size,
            rng_key=jax.random.PRNGKey(0),
        )

        # Process batches
        for batch_data, batch_targets in dataloader:
            # JIT-compiled gradient computation
            _ = jax.value_and_grad(loss_fn_for_grad)(params, batch_data, batch_targets)

            # Process with optimized accumulation
            self._process_single_batch_collector(
                batch_data.shape[0],
                compute_method,
                model.use_bias,
            )

        if compute_method == "covariance":
            # Average covariances over all samples
            for layer_name in self.kfac_data.covariances.activations.keys():
                self.kfac_data.covariances.activations[layer_name] /= n_samples
                self.kfac_data.covariances.gradients[layer_name] /= n_samples
        if compute_method == "eigenvalue_correction":
            # Average eigenvalue corrections over all samples
            for layer_name in self.kfac_data.eigenvalue_corrections.keys():
                self.kfac_data.eigenvalue_corrections[layer_name] /= n_samples

    def _process_single_batch_collector(
        self,
        batch_size: int,
        compute_method: Literal["covariance", "eigenvalue_correction"],
        use_bias: bool = False,
    ):
        """
        Process a batch of activation and gradient data. Compute and accumulate their covariances / eigenvalue corrections.
        """
        for layer_name, (a, g) in self.collector.captured_data.items():

            @jax.jit
            def add_bias_term(
                x: Float[Array, "n_batch d_in"],
            ) -> Float[Array, "n_batch d_in+1"]:
                batch_size = x.shape[0]
                return jnp.concatenate([x, jnp.ones((batch_size, 1))], axis=1)

            if use_bias:
                a = add_bias_term(a)

            if compute_method == "covariance":
                # Compute running covariance / eigenvalue correction by accumulation
                self._accumulate_covariances(layer_name, a, g)

            elif compute_method == "eigenvalue_correction":
                self._accumulate_eigenvalue_corrections(layer_name, a, g)
            else:
                raise ValueError(f"Unknown compute method: {compute_method}")

    def _accumulate_data(
        self, store: Dict, key: str, batch_covariance: Float[Array, "..."]
    ) -> None:
        store[key] = store.get(key, 0) + batch_covariance

    def _accumulate_covariances(
        self,
        layer_name: str,
        activations: Float[Array, "n_batch d_in"],
        gradients: Float[Array, "n_batch d_out"],
    ):
        """Accumulate covariance matrices for a given layer."""
        self._accumulate_data(
            self.kfac_data.covariances.activations,
            layer_name,
            self.covariance(activations),
        )
        self._accumulate_data(
            self.kfac_data.covariances.gradients,
            layer_name,
            self.covariance(gradients),
        )

    def _accumulate_eigenvalue_corrections(
        self,
        layer_name: str,
        activations: Float[Array, "n_batch d_in"],
        gradients: Float[Array, "n_batch d_out"],
    ):
        r"""
        Compute eigenvalue correction for a given layer.

        For each sample n, we compute:
            (Q_G \otimes Q_A)^T vec(a_n g_n^T) = (Q_G \otimes Q_A)^T (g_n \otimes a_n)

        Using the Kronecker product property (A \otimes B)^T = A^T \otimes B^T and
        the mixed-product property, this simplifies to:
            (Q_G^T g_n) \otimes (Q_A^T a_n) = vec((Q_A^T a_n) (Q_G^T g_n)^T)

        where:
        - Q_A, Q_G are the eigenvector matrices of the activation and gradient covariances
        - a_n, g_n are the activation and pre-activation gradient vectors for sample n
        - \otimes denotes the Kronecker product

        Implementation steps:
        1. Transform activations to eigenbasis: a_tilde_n = Q_A^T @ a_n
        2. Transform gradients to eigenbasis: g_tilde_n = Q_G^T @ g_n
        3. Compute outer product / Kronecker product: a_tilde_n \otimes g_tilde_n
        4. Square and sum across samples (averaging is later done by the caller)

        Note:
        The paper of Grosse et al. (2023) misses the transpose of the eigenvector
        basis (Q_A \otimes Q_G) in Equation (20). Refer to George et al. (2018) for the
        correct formulation.

        In this JAX implementation, we have to swap the Kronecker order to
        (Q_G \otimes Q_A) because JAX layers store weights with shape [input_dim, output_dim],
        unlike the [output_dim, input_dim] convention used in the original paper.

        Furthermore, we don't store the vectorized version of the outer product, but rather the
        matrix form. Note, that the resulting matrix represents the row-major flattening
        of the original EKFAC formulation due to the JAX weight layout convention.
        """
        Q_A: Float[Array, "d_in d_in"] = self.kfac_data.eigenvectors.activations[
            layer_name
        ]
        Q_G: Float[Array, "d_out d_out"] = self.kfac_data.eigenvectors.gradients[
            layer_name
        ]

        correction: Float[Array, "d_in d_out"] = (
            self._compute_eigenvalue_correction_batch(Q_A, Q_G, activations, gradients)
        )

        self._accumulate_data(
            self.kfac_data.eigenvalue_corrections, layer_name, correction
        )

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "n_batch I"],
        gradients: Float[Array, "n_batch O"],
    ) -> Float[Array, "I O"]:
        """JIT-compiled eigenvalue correction computation."""
        g_tilde = jnp.einsum("op, np -> no", Q_G.T, gradients)
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)
        return (outer**2).sum(axis=0)

    @staticmethod
    @partial(jax.jit, static_argnames=["method"])
    def compute_ihvp_or_hvp_layer(
        v_layer: Float[Array, "*batch_size I O"],
        Q_A: Float[Array, "I I"],
        Q_S: Float[Array, "O O"],
        Lambda: Float[Array, "I O"],
        damping: Float[Array, ""] = jnp.array(0.0),
        method: Literal["ihvp", "hvp"] = "ihvp",
    ) -> Float[Array, "*batch_size I O"]:
        """
        Compute the EKFAC-based (inverse) Hessian-vector product for a single layer.

        Depending on the selected method, this function computes either:
            - the inverse Hessian-vector product (IHVP):
                    (H + λI)⁻¹ v ≈ Q_A (Ṽ / (Λ + λ)) Q_S^T
            - or the Hessian-vector product (HVP):
                    (H + λI) v ≈ Q_A (Ṽ * (Λ + λ)) Q_S^T
        where Ṽ = Q_A^T unvectorized_v Q_S.

        Shapes follow the JAX convention with W ∈ R^{[I, O]},
        i.e., the vector is provided as unvectorized matrix in row-major order.
        """

        # Transform to eigenbasis
        V_tilde: Float[Array, "*batch_size I O"] = Q_A.T @ v_layer @ Q_S

        # Apply eigenvalue corrections + damping
        Lambda_damped: Float[Array, "I O"] = Lambda + damping

        if method == "ihvp":
            scaled: Float[Array, "*batch_size I O"] = V_tilde / Lambda_damped
        else:
            scaled: Float[Array, "*batch_size I O"] = V_tilde * Lambda_damped

        # Transform back to original basis
        vector_product: Float[Array, "*batch_size I O"] = Q_A @ scaled @ Q_S.T

        return vector_product

    def compute_eigenvalue_lambda_kfac(self, layer_name: str) -> Float[Array, "I O"]:
        """Compute eigenvalue lambda for KFAC using the following formula:
        Λ = (Λ_G ⊗ Λ_A) = Λ_A @ Λ_G^T
        where Λ_G and Λ_A are the eigenvalues of the gradient and activation covariances.
        """
        A_eigvals: Float[Array, "I"] = self.kfac_data.eigenvalues.activations[
            layer_name
        ]
        G_eigvals: Float[Array, "O"] = self.kfac_data.eigenvalues.gradients[layer_name]
        return jnp.outer(A_eigvals, G_eigvals)

    def generate_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        loss_fn: Callable,
        rng_key: PRNGKeyArray | None = None,
    ) -> Float[Array, "n_samples targets"] | Int[Array, "n_samples"]:
        """
        Generate pseudo-targets based on the model's output distribution.

        This is used to compute the true Fisher Information Matrix rather than
        the empirical Fisher (which would use true labels).
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        loss_name = get_loss_name(loss_fn)
        if loss_name == "cross_entropy":
            return self._generate_classification_pseudo_targets(
                model, params, training_data, rng_key
            )
        elif loss_name == "mse":
            return self._generate_regression_pseudo_targets(
                model, params, training_data, rng_key
            )
        else:
            raise ValueError(f"Unsupported loss function for EKFAC: {loss_name}")

    def _generate_classification_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
    ) -> Int[Array, "n_samples"]:
        """Generate pseudo-targets by sampling from softmax probabilities."""
        logits = model.apply(params, training_data)
        if not isinstance(logits, jnp.ndarray):
            raise ValueError(
                "Model predictions must be a jnp.ndarray for classification."
            )
        probs = jax.nn.softmax(logits, axis=-1)
        return jax.random.categorical(rng_key, jnp.log(probs), axis=-1)

    def _generate_regression_pseudo_targets(
        self,
        model: ApproximationModel,
        params: Dict,
        training_data: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
    ) -> Float[Array, "n_samples"]:
        """Generate pseudo-targets by adding Gaussian noise to predictions."""
        preds = model.apply(params, training_data)

        if not isinstance(preds, jnp.ndarray):
            raise ValueError("Model predictions must be a jnp.ndarray for regression.")
        noise = (
            self.kfac_config.build_config.pseudo_target_noise_std
            * jax.random.normal(rng_key, preds.shape)
        )
        return preds + noise
