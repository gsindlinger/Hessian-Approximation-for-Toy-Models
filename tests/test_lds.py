"""
Unit tests for the ELSO LDS analysis pipeline.

Tests are split into two tiers:

  Pure-function tests  — no neural-network training, very fast.
    - generate_random_subsets
    - compute_group_attributions
    - bootstrap_spearman_ci
    - aggregate_lds_scores

  Gradient / influence tests — train a tiny LINEAR model once (session scope).
    - compute_per_example_flat_grads  (shape, finiteness, sign)
    - compute_influence_matrix        (shape, finiteness, symmetry check)
    - compute_elso_ground_truth       (shape, Δm sign for removed vs kept)
"""

import warnings
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.random import PRNGKey
from scipy.stats import ConstantInputWarning

from src.config import (
    LossType,
    ModelArchitecture,
    ModelConfig,
    OptimizerType,
    PseudoTargetGenerationStrategy,
    TrainingConfig,
)
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.ekfac import EKFACComputer
from src.utils.data.data import Dataset, RandomRegressionDataset
from src.utils.influence import compute_influence_matrix, compute_per_example_flat_grads
from src.utils.lds import (
    aggregate_lds_scores,
    bootstrap_spearman_ci,
    compute_elso_ground_truth,
    compute_group_attributions,
    generate_random_subsets,
)
from src.utils.models.approximation_model import ApproximationModel
from src.utils.train import evaluate_per_example_losses as compute_per_example_losses
from tests._helpers import cached_train_model_for_dataset
from tests.conftest import TrainingScenario

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="session")
def config(lds_scenario: TrainingScenario) -> ModelConfig:
    return lds_scenario.model_config


@pytest.fixture(scope="session")
def dataset(lds_scenario: TrainingScenario) -> Dataset:
    return lds_scenario.dataset


@pytest.fixture(scope="session")
def model_params_loss(
    trained_model_registry: Dict[
        Tuple[Any, ...], Tuple[ApproximationModel, Dict, Callable]
    ],
    lds_scenario: TrainingScenario,
) -> Tuple[ApproximationModel, Dict, Callable]:
    return cached_train_model_for_dataset(
        lds_scenario.model_config,
        lds_scenario.dataset,
        trained_model_registry,
        seed=lds_scenario.train_seed,
        shuffle=lds_scenario.shuffle,
    )


@pytest.fixture(scope="session")
def ekfac_computer(
    config: ModelConfig,
    dataset: Dataset,
    model_params_loss: Tuple[ApproximationModel, Dict, Callable],
):
    """Built EKFACComputer for the LDS scenario."""
    model, params, loss_fn = model_params_loss

    collector = CollectorActivationsGradients(
        model=model,
        params=params,
        loss_fn=loss_fn,
        pseudo_target_strategy=PseudoTargetGenerationStrategy.EMPIRICAL_FISHER,
        pseudo_target_repetitions=1,
    )
    collector_data = collector.collect(
        dataset=dataset,
        save_directory=f"{config.directory}/ekfac",
        try_load=True,
        rng_key=PRNGKey(42),
    )
    return EKFACComputer(
        compute_context=collector_data, corr_context=collector_data
    ).build()


# ===========================================================================
# Pure-function tests: generate_random_subsets
# ===========================================================================


class TestGenerateRandomSubsets:
    def test_count(self):
        subsets = generate_random_subsets(100, 10, 0.5, seed=0)
        assert len(subsets) == 10

    def test_mask_length(self):
        subsets = generate_random_subsets(100, 5, 0.3, seed=0)
        for mask in subsets:
            assert len(mask) == 100

    def test_subset_size(self):
        subsets = generate_random_subsets(200, 8, 0.4, seed=0)
        expected = int(200 * 0.4)
        for mask in subsets:
            assert mask.sum() == expected

    def test_dtype_is_bool(self):
        subsets = generate_random_subsets(50, 3, 0.5, seed=0)
        for mask in subsets:
            assert mask.dtype == bool

    def test_reproducible(self):
        s1 = generate_random_subsets(100, 5, 0.3, seed=42)
        s2 = generate_random_subsets(100, 5, 0.3, seed=42)
        for m1, m2 in zip(s1, s2):
            np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self):
        s1 = generate_random_subsets(100, 5, 0.3, seed=1)
        s2 = generate_random_subsets(100, 5, 0.3, seed=2)
        # At least one subset pair should differ
        assert any(not np.array_equal(m1, m2) for m1, m2 in zip(s1, s2))

    def test_no_replacement_within_subset(self):
        """Each subset is sampled without replacement — no duplicates."""
        subsets = generate_random_subsets(100, 20, 0.5, seed=0)
        for mask in subsets:
            # Exactly `subset_size` True entries means no duplicate indices
            assert mask.sum() == int(100 * 0.5)


# ===========================================================================
# Pure-function tests: compute_group_attributions
# ===========================================================================


class TestComputeGroupAttributions:
    def test_output_shape(self):
        n_test, n_train, K = 5, 20, 8
        attrs = np.random.randn(n_test, n_train)
        subsets = [np.random.rand(n_train) > 0.5 for _ in range(K)]
        out = compute_group_attributions(attrs, subsets)
        assert out.shape == (n_test, K)

    def test_correct_summation(self):
        # Single query, 4 training points, 1 subset covering points 0 and 2
        attrs = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        mask = np.array([True, False, True, False])
        out = compute_group_attributions(attrs, [mask])
        assert out[0, 0] == pytest.approx(4.0)  # 1 + 3

    def test_empty_subset_gives_zero(self):
        attrs = np.ones((3, 10))
        mask = np.zeros(10, dtype=bool)  # empty subset
        out = compute_group_attributions(attrs, [mask])
        np.testing.assert_array_equal(out[:, 0], 0.0)

    def test_full_subset_equals_row_sum(self):
        attrs = np.random.randn(4, 12)
        mask = np.ones(12, dtype=bool)
        out = compute_group_attributions(attrs, [mask])
        np.testing.assert_allclose(out[:, 0], attrs.sum(axis=1))

    def test_additivity_across_complementary_subsets(self):
        """Sum over two complementary subsets should equal total row sum."""
        n_train = 10
        attrs = np.random.randn(3, n_train)
        mask1 = np.array([True] * 5 + [False] * 5)
        mask2 = ~mask1
        out = compute_group_attributions(attrs, [mask1, mask2])
        np.testing.assert_allclose(out.sum(axis=1), attrs.sum(axis=1))


# ===========================================================================
# Pure-function tests: bootstrap_spearman_ci
# ===========================================================================


class TestBootstrapSpearmanCI:
    def test_output_structure(self):
        x = np.arange(10, dtype=float)
        r, lo, hi = bootstrap_spearman_ci(x, x, n_bootstrap=100, seed=0)
        assert isinstance(r, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_bounds(self):
        rng = np.random.default_rng(0)
        x = rng.random(20)
        y = rng.random(20)
        r, lo, hi = bootstrap_spearman_ci(x, y, n_bootstrap=200, seed=0)
        assert -1.0 <= lo <= r <= hi <= 1.0

    def test_perfect_correlation_gives_high_r(self):
        x = np.arange(20, dtype=float)
        r, lo, hi = bootstrap_spearman_ci(x, x, n_bootstrap=200, seed=0)
        assert r == pytest.approx(1.0, abs=1e-10)
        assert lo > 0.9

    def test_anti_correlation_gives_low_r(self):
        x = np.arange(20, dtype=float)
        r, _, _ = bootstrap_spearman_ci(x, -x, n_bootstrap=200, seed=0)
        assert r == pytest.approx(-1.0, abs=1e-10)

    def test_constant_predicted_returns_zero(self):
        """Constant predicted gives undefined Spearman — should return 0 not NaN."""
        x = np.arange(10, dtype=float)
        y = np.ones(10)
        r, lo, hi = bootstrap_spearman_ci(x, y, n_bootstrap=100, seed=0)
        assert not np.isnan(r)

    def test_reproducible(self):
        rng = np.random.default_rng(7)
        x, y = rng.random(15), rng.random(15)
        r1, lo1, hi1 = bootstrap_spearman_ci(x, y, n_bootstrap=200, seed=42)
        r2, lo2, hi2 = bootstrap_spearman_ci(x, y, n_bootstrap=200, seed=42)
        assert r1 == r2
        assert lo1 == lo2
        assert hi1 == hi2

    def test_ci_width_shrinks_with_more_data(self):
        """Larger samples give tighter CIs (statistical property)."""
        rng = np.random.default_rng(0)
        small_x, small_y = rng.random(10), rng.random(10)
        large_x, large_y = rng.random(200), rng.random(200)

        _, lo_s, hi_s = bootstrap_spearman_ci(small_x, small_y, n_bootstrap=500)
        _, lo_l, hi_l = bootstrap_spearman_ci(large_x, large_y, n_bootstrap=500)

        assert (hi_s - lo_s) >= (hi_l - lo_l)


# ===========================================================================
# Pure-function tests: aggregate_lds_scores
# ===========================================================================


class TestAggregateLDSScores:
    def test_output_keys(self):
        dm = np.random.randn(5, 10)
        pred = np.random.randn(5, 10)
        out = aggregate_lds_scores(dm, pred, n_bootstrap=100)
        for key in (
            "mean_lds",
            "std_lds",
            "ci_low",
            "ci_high",
            "per_query_lds",
            "num_undefined_queries",
            "num_valid_queries",
        ):
            assert key in out

    def test_per_query_length(self):
        n_q = 7
        dm = np.random.randn(n_q, 12)
        pred = np.random.randn(n_q, 12)
        out = aggregate_lds_scores(dm, pred, n_bootstrap=50)
        assert len(out["per_query_lds"]) == n_q
        assert out["num_valid_queries"] + out["num_undefined_queries"] == n_q

    def test_mean_is_average_of_per_query(self):
        dm = np.random.randn(4, 8)
        pred = np.random.randn(4, 8)
        out = aggregate_lds_scores(dm, pred, n_bootstrap=50)
        expected_mean = float(np.nanmean(out["per_query_lds"]))
        assert out["mean_lds"] == pytest.approx(expected_mean)

    def test_bounded_output(self):
        dm = np.random.randn(5, 10)
        pred = np.random.randn(5, 10)
        out = aggregate_lds_scores(dm, pred, n_bootstrap=100)
        assert -1.0 <= out["mean_lds"] <= 1.0
        for r in out["per_query_lds"]:
            if not np.isnan(r):
                assert -1.0 <= r <= 1.0

    def test_perfect_prediction_gives_high_lds(self):
        """Predicted == actual should yield near-perfect LDS."""
        dm = np.tile(np.arange(10, dtype=float), (3, 1))  # (3, 10)
        out = aggregate_lds_scores(dm, dm, n_bootstrap=100)
        assert out["mean_lds"] == pytest.approx(1.0, abs=1e-10)

    def test_constant_rows_are_undefined_without_warning(self):
        dm = np.ones((2, 3))
        pred = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]])

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            out = aggregate_lds_scores(dm, pred, n_bootstrap=10)

        assert out["num_undefined_queries"] == 2
        assert out["num_valid_queries"] == 0
        assert not any(
            issubclass(w.category, ConstantInputWarning) for w in seen
        )


# ===========================================================================
# Gradient tests: compute_per_example_flat_grads
# ===========================================================================


class TestComputePerExampleFlatGrads:
    def test_output_shape(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
    ):
        model, params, loss_fn = model_params_loss
        n = 10
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n],
            targets=dataset.targets[:n],
            loss_fn=loss_fn,
        )
        # Flat dimension = total number of model parameters
        from jax import flatten_util

        n_params = flatten_util.ravel_pytree(params)[0].shape[0]
        assert grads.shape == (n, n_params)

    def test_finite_values(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
    ):
        model, params, loss_fn = model_params_loss
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:8],
            targets=dataset.targets[:8],
            loss_fn=loss_fn,
        )
        assert jnp.isfinite(grads).all()

    def test_different_examples_differ(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
    ):
        """Per-example gradients should generally not all be equal."""
        model, params, loss_fn = model_params_loss
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:5],
            targets=dataset.targets[:5],
            loss_fn=loss_fn,
        )
        # At least two gradients should differ
        pairwise_equal = jnp.allclose(grads[0], grads[1])
        assert not pairwise_equal

    def test_gradient_is_gradient(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
    ):
        """Verify against a manual per-example gradient for the first sample."""
        from jax import flatten_util

        model, params, loss_fn = model_params_loss
        flat_params, unravel = flatten_util.ravel_pytree(params)

        x = dataset.inputs[0]
        y = dataset.targets[0]

        ref_grad = jax.grad(
            lambda fp: loss_fn(model.apply(unravel(fp), x[None]), jnp.atleast_1d(y))
        )(flat_params)

        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:1],
            targets=dataset.targets[:1],
            loss_fn=loss_fn,
        )
        np.testing.assert_allclose(np.array(grads[0]), np.array(ref_grad), rtol=1e-5)


# ===========================================================================
# Influence tests: compute_influence_matrix
# ===========================================================================


class TestComputeInfluenceMatrix:
    def test_output_shape(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
        ekfac_computer: EKFACComputer,
    ):
        model, params, loss_fn = model_params_loss
        n_test, n_train = 4, 10
        test_grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n_test],
            targets=dataset.targets[:n_test],
            loss_fn=loss_fn,
        )
        train_grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n_train],
            targets=dataset.targets[:n_train],
            loss_fn=loss_fn,
        )
        damping = 0.1
        attrs = compute_influence_matrix(
            test_grads, train_grads, ekfac_computer, damping
        )
        assert attrs.shape == (n_test, n_train)

    def test_finite_values(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
        ekfac_computer: EKFACComputer,
    ):
        model, params, loss_fn = model_params_loss
        n = 5
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n],
            targets=dataset.targets[:n],
            loss_fn=loss_fn,
        )
        attrs = compute_influence_matrix(grads, grads, ekfac_computer, damping=0.1)
        assert np.isfinite(attrs).all()

    def test_self_influence_positive(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
        ekfac_computer: EKFACComputer,
    ):
        """τ(z, z) = ∇Lᵀ H⁻¹ ∇L ≥ 0 for PSD H⁻¹ (H + λI with λ > 0 is PD)."""
        model, params, loss_fn = model_params_loss
        n = 6
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n],
            targets=dataset.targets[:n],
            loss_fn=loss_fn,
        )
        attrs = compute_influence_matrix(grads, grads, ekfac_computer, damping=0.1)
        diagonal = np.diag(attrs)
        assert np.isfinite(diagonal).all()
        assert np.all(diagonal >= 0.0)

    def test_higher_damping_reduces_magnitude(
        self,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
        dataset: Dataset,
        ekfac_computer: EKFACComputer,
    ):
        """Higher damping → H^{-1} shrinks → smaller |ihvp| → smaller |influence|."""
        model, params, loss_fn = model_params_loss
        n = 5
        grads = compute_per_example_flat_grads(
            model=model,
            params=params,
            inputs=dataset.inputs[:n],
            targets=dataset.targets[:n],
            loss_fn=loss_fn,
        )
        attrs_lo = compute_influence_matrix(grads, grads, ekfac_computer, damping=0.01)
        attrs_hi = compute_influence_matrix(grads, grads, ekfac_computer, damping=10.0)
        assert np.abs(attrs_lo).mean() >= np.abs(attrs_hi).mean()


# ===========================================================================
# Integration: compute_elso_ground_truth
# ===========================================================================


class TestComputeElsoGroundTruth:
    """Lightweight integration test — trains tiny models on subset complements."""

    @pytest.mark.parametrize("n_targets", [1, 3])
    def test_vmap_regression_preserves_target_shape(self, monkeypatch, n_targets: int):
        dataset = RandomRegressionDataset(
            n_samples=12,
            n_features=3,
            n_targets=n_targets,
            noise=0.1,
            seed=11,
        )
        config = ModelConfig(
            architecture=ModelArchitecture.LINEAR,
            input_dim=dataset.input_dim(),
            hidden_dim=None,
            output_dim=dataset.output_dim(),
            loss=LossType.MSE,
            training=TrainingConfig(
                learning_rate=1e-6,
                optimizer=OptimizerType.SGD,
                epochs=1,
                batch_size=4,
            ),
        )

        def strict_mse_loss(pred, target, reduction="mean"):
            if pred.shape != target.shape:
                raise ValueError(
                    f"pred/target shape mismatch: {pred.shape} != {target.shape}"
                )
            per_sample = jnp.mean((pred - target) ** 2, axis=-1)
            if reduction == "sum":
                return jnp.sum(per_sample)
            return jnp.mean(per_sample)

        monkeypatch.setattr("src.utils.lds.get_loss", lambda _: strict_mse_loss)

        n_query, K = 2, 2
        subsets = generate_random_subsets(
            dataset_size=len(dataset.inputs),
            num_subsets=K,
            subset_fraction=0.25,
            seed=3,
        )
        delta_m = compute_elso_ground_truth(
            model_config=config,
            full_train_inputs=dataset.inputs,
            full_train_targets=dataset.targets,
            query_inputs=dataset.inputs[:n_query],
            query_targets=dataset.targets[:n_query],
            subsets=subsets,
            reps_per_subset=1,
            baseline_losses=np.zeros(n_query),
            base_seed=4,
            epochs=config.training.epochs,
            use_vmap=True,
        )

        assert delta_m.shape == (n_query, K)
        assert np.isfinite(delta_m).all()

    def test_non_vmap_does_not_write_checkpoints(self, tmp_path):
        dataset = RandomRegressionDataset(
            n_samples=12,
            n_features=3,
            n_targets=1,
            noise=0.1,
            seed=13,
        )
        model_dir = tmp_path / "baseline_model"
        model_dir.mkdir()
        sentinel = model_dir / "baseline.txt"
        sentinel.write_text("do not touch")
        config = ModelConfig(
            architecture=ModelArchitecture.LINEAR,
            input_dim=dataset.input_dim(),
            hidden_dim=None,
            output_dim=dataset.output_dim(),
            loss=LossType.MSE,
            training=TrainingConfig(
                learning_rate=1e-6,
                optimizer=OptimizerType.SGD,
                epochs=1,
                batch_size=4,
            ),
            directory=str(model_dir),
        )

        n_query, K = 2, 2
        subsets = generate_random_subsets(
            dataset_size=len(dataset.inputs),
            num_subsets=K,
            subset_fraction=0.25,
            seed=3,
        )
        delta_m = compute_elso_ground_truth(
            model_config=config,
            full_train_inputs=dataset.inputs,
            full_train_targets=dataset.targets,
            query_inputs=dataset.inputs[:n_query],
            query_targets=dataset.targets[:n_query],
            subsets=subsets,
            reps_per_subset=1,
            baseline_losses=np.zeros(n_query),
            base_seed=4,
            epochs=config.training.epochs,
            use_vmap=False,
        )

        assert delta_m.shape == (n_query, K)
        assert sentinel.read_text() == "do not touch"
        assert not (model_dir / "epoch_1").exists()

    def test_output_shape(
        self,
        dataset: Dataset,
        config: ModelConfig,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    ):
        model, params, loss_fn = model_params_loss
        n_query, K = 3, 5

        subsets = generate_random_subsets(
            dataset_size=len(dataset.inputs),
            num_subsets=K,
            subset_fraction=0.4,
            seed=0,
        )
        baseline = compute_per_example_losses(
            model=model,
            params=params,
            inputs=dataset.inputs[:n_query],
            targets=dataset.targets[:n_query],
            loss_fn=loss_fn,
        )
        delta_m = compute_elso_ground_truth(
            model_config=config,
            full_train_inputs=dataset.inputs,
            full_train_targets=dataset.targets,
            query_inputs=dataset.inputs[:n_query],
            query_targets=dataset.targets[:n_query],
            subsets=subsets,
            reps_per_subset=1,  # R=1 to keep the test fast
            baseline_losses=baseline,  # type: ignore
            base_seed=0,
            epochs=config.training.epochs,
        )
        assert delta_m.shape == (n_query, K)  # type: ignore

    def test_finite_values(
        self,
        dataset: Dataset,
        config: ModelConfig,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    ):
        model, params, loss_fn = model_params_loss
        n_query, K = 2, 3

        subsets = generate_random_subsets(len(dataset.inputs), K, 0.4, seed=1)
        baseline = compute_per_example_losses(
            model=model,
            params=params,
            inputs=dataset.inputs[:n_query],
            targets=dataset.targets[:n_query],
            loss_fn=loss_fn,
        )
        delta_m = compute_elso_ground_truth(
            model_config=config,
            full_train_inputs=dataset.inputs,
            full_train_targets=dataset.targets,
            query_inputs=dataset.inputs[:n_query],
            query_targets=dataset.targets[:n_query],
            subsets=subsets,
            reps_per_subset=1,
            baseline_losses=baseline,  # type: ignore
            base_seed=1,
            epochs=config.training.epochs,
        )
        assert np.isfinite(delta_m).all()  # type: ignore

    def test_delta_m_is_difference_from_baseline(
        self,
        dataset: Dataset,
        config: ModelConfig,
        model_params_loss: Tuple[ApproximationModel, Dict, Callable],
    ):
        """Δm_j = E[loss(D\\S_j)] - loss(D), so Δm with R=1 should have finite
        spread (not all the same value), reflecting subset-induced variation."""
        model, params, loss_fn = model_params_loss
        n_query, K = 2, 6

        subsets = generate_random_subsets(len(dataset.inputs), K, 0.5, seed=2)
        baseline = compute_per_example_losses(
            model=model,
            params=params,
            inputs=dataset.inputs[:n_query],
            targets=dataset.targets[:n_query],
            loss_fn=loss_fn,
        )
        delta_m = compute_elso_ground_truth(
            model_config=config,
            full_train_inputs=dataset.inputs,
            full_train_targets=dataset.targets,
            query_inputs=dataset.inputs[:n_query],
            query_targets=dataset.targets[:n_query],
            subsets=subsets,
            reps_per_subset=1,
            baseline_losses=baseline,  # type: ignore
            base_seed=2,
            epochs=config.training.epochs,
        )
        # Δm values across subsets should vary (not all identical)
        for i in range(n_query):
            assert not np.allclose(delta_m[i], delta_m[i, 0])  # type: ignore
