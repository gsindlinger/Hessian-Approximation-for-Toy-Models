import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...config.dataset_config import DatasetConfig
from ...config.hessian_approximation_config import (
    KFACBuildConfig,
)
from ...config.model_config import ModelConfig
from ...config.training_config import TrainingConfig
from ...data.jax_dataloader import JAXDataLoader
from ...models.base import ApproximationModel
from ...models.dataclasses.model_context import ModelContext
from ...models.utils.loss import get_loss_name
from .data.activation_gradient_collector import (
    ActivationGradientCollector,
)
from .data.data import KFACData, MeanEigenvaluesAndCorrections
from .data.layer_components import LayerComponents
from .kfac_storage import KFACStorage

logger = logging.getLogger(__name__)


@dataclass
class KFACProvider:
    model_context: ModelContext
    configs: Tuple[ModelConfig, DatasetConfig, TrainingConfig, KFACBuildConfig]
    storage: KFACStorage = field(init=False)

    data: KFACData = field(default_factory=KFACData)
    data_means: MeanEigenvaluesAndCorrections = field(
        default_factory=MeanEigenvaluesAndCorrections
    )
    collector: ActivationGradientCollector = field(
        default_factory=ActivationGradientCollector
    )

    def __post_init__(self):
        self.storage = KFACStorage(configs=self.configs)
        assert isinstance(self.configs[3], KFACBuildConfig), (
            "Hessian approximation config must be of type KFACConfig"
        )
        self.kfac_build_config = self.configs[3]

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
        # Note: There is a custom bool implementation in KFACData and KFACDataMeans
        # that checks for the presence of all required components.
        if (
            self.data
            and self.data_means
            and (not self.kfac_build_config.recalc_ekfac_components)
        ):
            return

        # check if kfac components are computed, if so reuse
        if self.storage.check_storage() and not (
            self.kfac_build_config.recalc_ekfac_components
        ):
            logger.info("Loading EKFAC components from disk.")
            self.data, self.data_means = self.storage.load_kfac_data()
        else:
            logger.info("Computing EKFAC components from scratch.")
            self.compute_from_scratch()

    def compute_from_scratch(self) -> None:
        """Compute all (E)KFAC components from scratch."""

        logger.info("Computing covariances for each layer.")
        self.compute_covariances()

        logger.info("Computing eigenvectors for each layer.")
        self.compute_eigenvectors()

        logger.info("Computing eigenvalue corrections for each layer.")
        self.compute_eigenvalue_corrections()

        logger.info("Computing mean eigenvalues and corrections for damping.")
        self.compute_mean_eigenvalues_and_corrections()

        logger.info("Finished computing EKFAC components from scratch.")

    def compute_covariances(
        self,
    ):
        """
        Compute A and G covariance matrices for each layer.
        Performs a forward-backward pass to collect activations and gradients,
        then computes their covariance matrices.
        """

        self._process_multiple_batches_collector(
            compute_method="covariance",
        )
        self.storage.save_covariances(self.data.covariances)

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
        if not self.data.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )

        activation_eigvals = {}
        activation_eigvecs = {}
        gradient_eigvals = {}
        gradient_eigvecs = {}

        for layer_name in self.data.covariances.activations.keys():
            (
                activation_eigvals_layer,
                gradient_eigvals_layer,
                activation_eigvecs_layer,
                gradient_eigvecs_layer,
            ) = self.compute_layer_eigenvectors(
                self.data.covariances.activations[layer_name],
                self.data.covariances.gradients[layer_name],
            )

            activation_eigvals[layer_name] = activation_eigvals_layer
            activation_eigvecs[layer_name] = activation_eigvecs_layer
            gradient_eigvals[layer_name] = gradient_eigvals_layer
            gradient_eigvecs[layer_name] = gradient_eigvecs_layer

        self.data.eigenvalues = LayerComponents(
            activations=activation_eigvals,
            gradients=gradient_eigvals,
        )
        self.data.eigenvectors = LayerComponents(
            activations=activation_eigvecs,
            gradients=gradient_eigvecs,
        )
        self.storage.save_eigenvectors(self.data.eigenvectors)
        self.storage.save_eigenvalues(self.data.eigenvalues)

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
        Q_A: Float[Array, "d_in d_in"] = self.data.eigenvectors.activations[layer_name]
        Q_G: Float[Array, "d_out d_out"] = self.data.eigenvectors.gradients[layer_name]

        correction: Float[Array, "d_in d_out"] = (
            self._compute_eigenvalue_correction_batch(Q_A, Q_G, activations, gradients)
        )

        self._accumulate_data(self.data.eigenvalue_corrections, layer_name, correction)

    @staticmethod
    @jax.jit
    def _compute_eigenvalue_correction_batch(
        Q_A: Float[Array, "I I"],
        Q_G: Float[Array, "O O"],
        activations: Float[Array, "n_batch I"],
        gradients: Float[Array, "n_batch O"],
    ) -> Float[Array, "I O"]:
        """Eigenvalue correction computation."""
        g_tilde = jnp.einsum("op, np -> no", Q_G.T, gradients)
        a_tilde = jnp.einsum("ij, nj -> ni", Q_A.T, activations)
        outer = jnp.einsum("ni, no -> nio", a_tilde, g_tilde)
        return (outer**2).sum(axis=0)

    def compute_eigenvalue_corrections(
        self,
    ) -> None:
        """
        Compute eigenvalue corrections for each layer.

        Projects activations and gradients onto eigenbases and computes
        the empirical eigenvalue corrections as outer products.
        """
        if not self.data.covariances:
            raise ValueError(
                "Covariances not computed yet. Run compute_covariances first."
            )
        if not self.data.eigenvectors:
            raise ValueError(
                "Eigenvectors not computed yet. Run compute_eigenvectors_and_eigenvalues first."
            )

        self._process_multiple_batches_collector(
            compute_method="eigenvalue_correction",
        )
        self.storage.save_eigenvalue_corrections(self.data.eigenvalue_corrections)

    def compute_mean_eigenvalues_and_corrections(self) -> None:
        """Compute mean eigenvalues and eigenvalue corrections for damping."""
        self.data_means.eigenvalues = {}
        self.data_means.corrections = {}

        overall_mean_eigenvalues = 0.0
        overall_mean_eigenvalue_corrections = 0.0

        for layer_name in self.data.eigenvalues.activations.keys():
            mean_eigenvalue: Float[Array, ""] = jnp.mean(
                self.compute_eigenvalue_lambda_kfac(layer_name)
            )
            self.data_means.eigenvalues[layer_name] = mean_eigenvalue
            overall_mean_eigenvalues += mean_eigenvalue

        for layer_name in self.data.eigenvalue_corrections.keys():
            mean_correction: Float[Array, ""] = jnp.mean(
                self.data.eigenvalue_corrections[layer_name]
            )
            self.data_means.corrections[layer_name] = mean_correction
            overall_mean_eigenvalue_corrections += mean_correction

        # Divide overall sums by number of layers
        n_layers = len(self.data.eigenvalues.activations)
        self.data_means.overall_mean_eigenvalues = jnp.array(
            overall_mean_eigenvalues / n_layers if n_layers > 0 else 0.0
        )
        self.data_means.overall_mean_corrections = jnp.array(
            overall_mean_eigenvalue_corrections / n_layers if n_layers > 0 else 0.0
        )

        self.storage.save_mean_eigenvalues_and_corrections(self.data_means)

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
        compute_method: Literal["covariance", "eigenvalue_correction"],
    ):
        """
        Process data in batches to collect activations and gradients and apply some function to it.
        So far it is used to compute covariances of the activations and preactivation gradients, as well as computing eigenvalue corrections.
        """

        def loss_fn_for_grad(p, training_data, targets):
            predictions = self.model_context.model.apply(
                p,
                training_data,
                self.collector,
                method=self.model_context.model.kfac_apply,
            )
            # Use sum reduction to avoid prematurely averaging gradients
            return self.model_context.loss(predictions, targets, reduction="sum")

        # Optionally generate pseudo-targets for true Fisher computation
        # Ensure different RNG keys for covariance and eigenvalue correction computations

        if self.kfac_build_config.use_pseudo_targets:
            if compute_method == "covariance":
                prng_key = jax.random.PRNGKey(42)
            else:
                prng_key = jax.random.PRNGKey(43)
            inputs = self.model_context.dataset.get_train_data()[0]
            targets = self.generate_pseudo_targets(
                model=self.model_context.model,
                params=self.model_context.params,
                inputs=inputs,
                loss_fn=self.model_context.loss,
                rng_key=prng_key,
            )
        else:
            inputs, targets = self.model_context.dataset.get_train_data()

        n_samples = inputs.shape[0]

        dataloader_batch_size = self.kfac_build_config.collector_batch_size
        dataloader = JAXDataLoader(
            data=inputs,
            targets=targets,
            shuffle=False,
            batch_size=dataloader_batch_size,
            rng_key=jax.random.PRNGKey(0),
        )

        # Process batches
        for batch_data, batch_targets in dataloader:
            # JIT-compiled gradient computation
            _ = jax.value_and_grad(loss_fn_for_grad)(
                self.model_context.params, batch_data, batch_targets
            )

            # Process with optimized accumulation
            self._process_single_batch_collector(
                batch_data.shape[0],
                compute_method,
                self.model_context.model.use_bias,
            )

        if compute_method == "covariance":
            # Average covariances over all samples
            for layer_name in self.data.covariances.activations.keys():
                self.data.covariances.activations[layer_name] /= n_samples
                self.data.covariances.gradients[layer_name] /= n_samples
        if compute_method == "eigenvalue_correction":
            # Average eigenvalue corrections over all samples
            for layer_name in self.data.eigenvalue_corrections.keys():
                self.data.eigenvalue_corrections[layer_name] /= n_samples

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
            self.data.covariances.activations,
            layer_name,
            self.covariance(activations),
        )
        self._accumulate_data(
            self.data.covariances.gradients,
            layer_name,
            self.covariance(gradients),
        )

    def collect_data(
        self,
        use_eigenvalue_correction: bool,
        vectors: Optional[Float[Array, "*batch_size n_params"]] = None,
    ) -> Tuple[
        List[Float[Array, "I I"]],
        List[Float[Array, "O O"]],
        List[Float[Array, "I O"]],
        Optional[List[Float[Array, "*batch_size I O"]]],
    ]:
        """
        Collect EKFAC components required for (inverse) Hessian-vector products.
        """
        self.get_ekfac_components()
        activations_eigenvectors = []
        gradients_eigenvectors = []
        Lambdas = []
        v_layers = []

        offset = 0
        for layer_name in self.model_context.params["params"].keys():
            activations_eigenvectors.append(
                self.data.eigenvectors.activations[layer_name]
            )
            gradients_eigenvectors.append(self.data.eigenvectors.gradients[layer_name])
            if use_eigenvalue_correction:
                Lambda = self.data.eigenvalue_corrections[layer_name]
            else:
                Lambda = self.compute_eigenvalue_lambda_kfac(layer_name)
            Lambdas.append(Lambda)

            if vectors is not None:
                input_dim, output_dim = Lambda.shape
                size = input_dim * output_dim

                v_flat: Float[Array, "*batch_size I*O"] = vectors[
                    ..., offset : offset + size
                ]
                v_layer: Float[Array, "*batch_size I O"] = v_flat.reshape(
                    v_flat.shape[:-1] + (input_dim, output_dim)
                )
                offset += size
                v_layers.append(v_layer)

        if vectors is None:
            v_layers = None

        return activations_eigenvectors, gradients_eigenvectors, Lambdas, v_layers

    @staticmethod
    def generate_pseudo_targets(
        model: ApproximationModel,
        params: Dict,
        inputs: Float[Array, "n_samples features"],
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
            return KFACProvider._generate_classification_pseudo_targets(
                model, params, inputs, rng_key
            )
        elif loss_name == "mse":
            return KFACProvider._generate_regression_pseudo_targets(
                model, params, inputs, rng_key
            )
        else:
            raise ValueError(f"Unsupported loss function for EKFAC: {loss_name}")

    @staticmethod
    def _generate_classification_pseudo_targets(
        model: ApproximationModel,
        params: Dict,
        inputs: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
    ) -> Int[Array, "n_samples"]:
        """Generate pseudo-targets by sampling from softmax probabilities."""
        logits = model.apply(params, inputs)
        if not isinstance(logits, jnp.ndarray):
            raise ValueError(
                "Model predictions must be a jnp.ndarray for classification."
            )
        probs = jax.nn.softmax(logits, axis=-1)
        return jax.random.categorical(rng_key, jnp.log(probs), axis=-1)

    @staticmethod
    def _generate_regression_pseudo_targets(
        model: ApproximationModel,
        params: Dict,
        inputs: Float[Array, "n_samples features"],
        rng_key: PRNGKeyArray,
        noise_std: float = 1.0,
    ) -> Float[Array, "n_samples"]:
        """Generate pseudo-targets by adding Gaussian noise to predictions."""
        preds = model.apply(params, inputs)

        if not isinstance(preds, jnp.ndarray):
            raise ValueError("Model predictions must be a jnp.ndarray for regression.")
        noise = noise_std * jax.random.normal(rng_key, preds.shape)
        return preds + noise

    def compute_eigenvalue_lambda_kfac(self, layer_name: str) -> Float[Array, "I O"]:
        """Compute eigenvalue lambda for KFAC using the following formula:
        Λ = (Λ_G ⊗ Λ_A) = Λ_A @ Λ_G^T
        where Λ_G and Λ_A are the eigenvalues of the gradient and activation covariances.
        """
        A_eigvals: Float[Array, "I"] = self.data.eigenvalues.activations[layer_name]
        G_eigvals: Float[Array, "O"] = self.data.eigenvalues.gradients[layer_name]
        return jnp.outer(A_eigvals, G_eigvals)

    def get_damping(
        self,
        base_damping: Float = 0.0,
        method: Literal["eigenvalues", "corrections"] = "eigenvalues",
    ) -> Float[Array, ""]:
        """Get the damping value adjusted by mean eigenvalues and corrections."""
        self.get_ekfac_components()

        if method == "eigenvalues":
            mean_value = self.data_means.overall_mean_eigenvalues
        elif method == "corrections":
            mean_value = self.data_means.overall_mean_corrections
        else:
            raise ValueError(f"Unknown damping method: {method}")

        return base_damping * mean_value
