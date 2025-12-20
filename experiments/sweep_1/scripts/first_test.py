import json
import logging
import os
import time

from jax.random import PRNGKey

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import Dataset, DigitsDataset
from src.utils.loss import cross_entropy_loss
from src.utils.metrics.full_matrix_metrics import MATRIX_METRICS
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    evaluate_loss_and_classification_accuracy,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)
from src.utils.utils import get_peak_bytes_in_use, hash_data

logger = logging.getLogger(__name__)


def simple_run():
    # Define config which serves mostly as reference for paths and model metadata
    seed = 32
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Load dataset
    digits_dataset_path = "experiments/sweep_1/data/datasets/digits"
    dataset: DigitsDataset = DigitsDataset.load(
        directory=digits_dataset_path, store_on_disk=True
    )

    layer_settings = [
        [16],
        4 * [16],
        8 * [16],
        [32],
        4 * [32],
        8 * [32],
        [64],
        4 * [64],
        8 * [64],
    ]
    optimizer_name = "sgd_schedule_cosine"
    learning_rate = 0.03

    # Initialize results structure
    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "seed": seed,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "dataset_path": digits_dataset_path,
        },
        "experiments": [],
    }

    best_model_names = []

    # Loop over models
    for hidden_layers in layer_settings:
        model_mlp = MLP(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=hidden_layers,
        )
        logging.info(
            f"Created Model: {model_mlp} with {model_mlp.num_params} parameters."
        )
        model_mlp_swiglu = MLPSwiGLU(
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            hidden_dim=[(dim // 3,) * 3 for dim in hidden_layers],  # type: ignore
            activation="swiglu",
        )
        logging.info(
            f"Created Model: {model_mlp_swiglu} with {model_mlp_swiglu.num_params} parameters."
        )

        for model in [model_mlp, model_mlp_swiglu]:
            # Initialize experiment result
            experiment_result = {
                "model_type": "MLP" if isinstance(model, MLP) else "MLPSwiGLU",
                "hidden_layers": hidden_layers,
                "num_params": model.num_params,
                "fold_selection": [],
            }

            # PHASE 1: K-fold cross-validation for model selection
            train_losses = []
            test_losses = []
            best_fold_index = None
            best_test_loss = float("inf")

            for fold_index, (
                (train_inputs, train_targets),
                (test_inputs, test_targets),
            ) in enumerate(
                dataset.get_k_fold_splits(
                    n_splits=2,
                    shuffle=True,
                    seed=seed,
                )
            ):
                hashed_data = hash_data(
                    {
                        "hidden_layers": hidden_layers,
                        "optimizer": optimizer_name,
                        "learning_rate": learning_rate,
                    }
                )
                if isinstance(model, MLP):
                    model_name = f"mlp_hidden_{hashed_data}_fold_{fold_index}"
                else:
                    model_name = f"mlp_swiglu_hidden_{hashed_data}_fold_{fold_index}"

                model_directory = (
                    f"experiments/sweep_1/data/models/digits_model/{model_name}/"
                )
                metadata = {
                    "model_name": model_name,
                    "model_directory": model_directory,
                    "metadata": {
                        "model_name": model_name,
                        "hidden_layers": hidden_layers,
                        "optimizer": optimizer_name,
                        "learning_rate": learning_rate,
                        "param_count": model.num_params,
                    },
                }

                # Train or load model
                if check_saved_model(model_directory, model=model):
                    params, _, metadata = load_model_checkpoint(
                        model_directory, model=model
                    )
                    train_loss = metadata.get("train_loss", None)
                else:
                    model, params, loss_history = train_model(
                        model,
                        Dataset(train_inputs, train_targets).get_dataloader(
                            batch_size=32, seed=seed
                        ),
                        loss_fn=cross_entropy_loss,
                        optimizer=optimizer(optimizer_name, lr=learning_rate),
                        epochs=100,
                    )
                    train_loss = loss_history[-1]
                    metadata["train_loss"] = train_loss

                test_loss, test_accuracy = evaluate_loss_and_classification_accuracy(
                    model,
                    params,
                    test_inputs,
                    test_targets,
                    loss_fn=cross_entropy_loss,
                )
                metadata["test_loss"] = test_loss
                metadata["test_accuracy"] = test_accuracy

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                save_model_checkpoint(
                    model=model,
                    params=params,
                    directory=model_directory,
                    metadata=metadata,
                )

                # Track best fold
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_fold_index = fold_index
                    best_model_name = model_name

                # Store fold selection results
                experiment_result["fold_selection"].append(
                    {
                        "fold_index": fold_index,
                        "model_name": model_name,
                        "train_loss": float(train_loss),
                        "test_loss": float(test_loss),
                        "test_accuracy": float(test_accuracy),
                    }
                )

                logger.info(
                    f"Model {model_name} trained and evaluated. Test loss: {test_loss}"
                )

            # Store fold selection summary
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_test_loss = sum(test_losses) / len(test_losses)

            experiment_result["fold_selection_summary"] = {
                "avg_train_loss": float(avg_train_loss),
                "avg_test_loss": float(avg_test_loss),
                "best_fold_index": best_fold_index,
                "best_model_name": best_model_name,
                "best_test_loss": float(best_test_loss),
            }

            logger.info(
                f"Best model selected: {best_model_name} with test loss: {best_test_loss}"
            )
            best_model_names.append(best_model_name)

            # PHASE 2: Load best model and perform Hessian analysis
            logger.info(f"Starting Hessian analysis for best model: {best_model_name}")

            # Track memory at start of Hessian analysis
            memory_start = get_peak_bytes_in_use()

            # Load best model
            best_model_directory = (
                f"experiments/sweep_1/data/models/digits_model/{best_model_name}/"
            )
            params, _, _ = load_model_checkpoint(best_model_directory, model=model)

            # Get the training data for the best fold
            fold_splits = list(
                dataset.get_k_fold_splits(n_splits=2, shuffle=True, seed=seed)
            )
            (train_inputs, train_targets), (test_inputs, test_targets) = fold_splits[
                best_fold_index
            ]

            train_dataset = Dataset(train_inputs, train_targets)

            # Generate gradient samples on training data
            gradient_samples_1 = sample_gradients(
                model=model,
                params=params,
                inputs=train_inputs,
                targets=train_targets,
                loss_fn=cross_entropy_loss,
                n_vectors=500,
                rng_key=PRNGKey(seed),
            )

            gradient_samples_2 = sample_gradients(
                model=model,
                params=params,
                inputs=train_inputs,
                targets=train_targets,
                loss_fn=cross_entropy_loss,
                n_vectors=500,
                rng_key=PRNGKey(seed + 1),
            )

            # Generate pseudo-targets for EKFAC/FIM using training data
            collector_data_1 = train_dataset.replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=cross_entropy_loss,
                    rng_key=PRNGKey(seed),
                )
            )

            collector_data_2 = train_dataset.replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=cross_entropy_loss,
                    rng_key=PRNGKey(seed + 1),
                )
            )

            # Collect activations and gradients
            collector_run_dir_1 = f"experiments/sweep_1/data/activation_gradient_collector/{best_model_name}/run1/"
            collector = CollectorActivationsGradients(model=model, params=params)
            collector.collect(
                collector_data_1.inputs,
                collector_data_1.targets,
                cross_entropy_loss,
                save_directory=collector_run_dir_1,
            )

            collector_run_dir_2 = f"experiments/sweep_1/data/activation_gradient_collector/{best_model_name}/run2/"
            collector.collect(
                collector_data_2.inputs,
                collector_data_2.targets,
                cross_entropy_loss,
                save_directory=collector_run_dir_2,
            )

            # Compute KFAC components and save them
            hessian_approx_dir = (
                f"experiments/sweep_1/data/hessian_approximations/{best_model_name}/"
            )
            ekfac_approximator = EKFACApproximator(
                collected_data_path=collector_run_dir_1,
                collected_data_path_second=collector_run_dir_2,
            )
            ekfac_config = Config(
                dataset_path=digits_dataset_path,
                model=ModelConfig(
                    model_name=best_model_name,
                    directory=best_model_directory,
                    metadata={},
                ),
                hessian_approximation=HessianApproximationConfig(
                    method="ekfac",
                    directory=hessian_approx_dir,
                ),
            )
            ekfac_approximator.build(
                config=ekfac_config,
                save_directory=hessian_approx_dir,
            )

            # Load EKFAC data and compute Hessians
            ekfac_data, _ = EKFACApproximator.load_data(directory=hessian_approx_dir)
            assert isinstance(ekfac_data, EKFACData)
            logger.info("K-FAC approximation loaded successfully.")

            damping = ekfac_data.mean_eigenvalues_aggregated * 0.1

            # Memory checkpoint after EKFAC loading
            memory_after_ekfac = get_peak_bytes_in_use()

            # Compute K-FAC Hessian
            kfac_computer = KFACComputer(compute_context=ekfac_data)
            kfac_hvp = kfac_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            kfac_ihvp = kfac_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"K-FAC HVP, and IHVP computed with shapes:, {kfac_hvp.shape}, {kfac_ihvp.shape}"
            )

            # Memory checkpoint after KFAC computation
            memory_after_kfac = get_peak_bytes_in_use()

            # Compute E-KFAC Hessian
            ekfac_computer = EKFACComputer(compute_context=ekfac_data)
            ekfac_hvp = ekfac_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            ekfac_ihvp = ekfac_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"E-KFAC HVP, and IHVP computed with shapes: {ekfac_hvp.shape}, {ekfac_ihvp.shape}"
            )

            # Memory checkpoint after EKFAC computation
            memory_after_ekfac_compute = get_peak_bytes_in_use()

            # Compute FIM
            model_context = ModelContext.create(
                dataset=train_dataset,
                model=model,
                params=params,
                loss_fn=cross_entropy_loss,
            )
            fim_computer = FIMComputer(compute_context=model_context)
            fim_hvp = fim_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            fim_ihvp = fim_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"FIM HVP, and IHVP computed with shapes: {fim_hvp.shape}, {fim_ihvp.shape}"
            )

            # Memory checkpoint after FIM computation
            memory_after_fim = get_peak_bytes_in_use()

            # Compute Block FIM
            block_fim_data = CollectorActivationsGradients.load(
                directory=collector_run_dir_1
            )
            block_fim_computer = FIMBlockComputer(compute_context=block_fim_data)
            block_fim_hvp = block_fim_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            block_fim_ihvp = block_fim_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"FIM Block HVP and IHVP computed with shapes: {block_fim_hvp.shape}, {block_fim_ihvp.shape}"
            )

            # Memory checkpoint after Block FIM computation
            memory_after_block_fim = get_peak_bytes_in_use()

            # Compute GNH
            gnh_computer = GNHComputer(compute_context=model_context)
            gnh = gnh_computer.estimate_hessian(damping=damping)
            gnh_hvp = gnh_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            gnh_ihvp = gnh_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"GNH HVP, and IHVP computed with shapes: {gnh_hvp.shape}, {gnh_ihvp.shape}"
            )

            # Memory checkpoint after GNH computation
            memory_after_gnh = get_peak_bytes_in_use()

            # Compute the true Hessian
            hessian_computer = HessianComputer(compute_context=model_context)
            full_hessian = hessian_computer.compute_hessian(damping=damping)
            full_hvp = hessian_computer.compute_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            full_ihvp = hessian_computer.compute_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"Full Hessian, HVP, and IHVP computed with shapes: {full_hvp.shape}, {full_ihvp.shape}"
            )

            # Compute block Hessian for comparison
            block_hessian_computer = BlockHessianComputer(compute_context=model_context)
            block_hessian_hvp = block_hessian_computer.estimate_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            block_hessian_ihvp = block_hessian_computer.estimate_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"Block Hessian HVP and IHVP computed with shapes: {block_hessian_hvp.shape}, {block_hessian_ihvp.shape}"
            )

            # Compare matrices
            matrix_results = {}
            matrix_metrics = MATRIX_METRICS["all_matrix"]
            for metric in matrix_metrics:
                logger.info(f"Comparing Hessian estimates using metric: {metric.value}")
                kfac_vs_full = kfac_computer.compare_full_hessian_estimates(
                    comparison_matrix=full_hessian,
                    metric=metric,
                    damping=damping,
                )
                ekfac_vs_full = ekfac_computer.compare_full_hessian_estimates(
                    comparison_matrix=full_hessian,
                    metric=metric,
                    damping=damping,
                )
                fim_vs_full = fim_computer.compare_full_hessian_estimates(
                    comparison_matrix=full_hessian,
                    metric=metric,
                    damping=damping,
                )
                block_fim_vs_full = block_fim_computer.compare_full_hessian_estimates(
                    comparison_matrix=full_hessian,
                    metric=metric,
                    damping=damping,
                )
                gnh_vs_full = gnh_computer.compare_full_hessian_estimates(
                    comparison_matrix=full_hessian,
                    metric=metric,
                    damping=damping,
                )
                block_hessian_vs_full = (
                    block_hessian_computer.compare_full_hessian_estimates(
                        comparison_matrix=full_hessian,
                        metric=metric,
                        damping=damping,
                    )
                )
                kfac_vs_gnh = kfac_computer.compare_full_hessian_estimates(
                    comparison_matrix=gnh,
                    metric=metric,
                    damping=damping,
                )
                ekfac_vs_gnh = ekfac_computer.compare_full_hessian_estimates(
                    comparison_matrix=gnh,
                    metric=metric,
                    damping=damping,
                )
                fim_vs_gnh = fim_computer.compare_full_hessian_estimates(
                    comparison_matrix=gnh,
                    metric=metric,
                    damping=damping,
                )
                block_fim_vs_gnh = block_fim_computer.compare_full_hessian_estimates(
                    comparison_matrix=gnh,
                    metric=metric,
                    damping=damping,
                )

                matrix_results[metric.value] = {
                    "hessian_comparison": {
                        "kfac": float(kfac_vs_full),
                        "ekfac": float(ekfac_vs_full),
                        "fim": float(fim_vs_full),
                        "block_fim": float(block_fim_vs_full),
                        "gnh": float(gnh_vs_full),
                        "block_hessian": float(block_hessian_vs_full),
                    },
                    "gnh_comparison": {
                        "kfac": float(kfac_vs_gnh),
                        "ekfac": float(ekfac_vs_gnh),
                        "fim": float(fim_vs_gnh),
                        "block_fim": float(block_fim_vs_gnh),
                    },
                }

                logger.info(f"Finished matrix comparison for metric: {metric.value}")

            # Compare HVPs
            logger.info("Comparing Hessian-vector products (HVPs).")
            hvp_results = {}
            for metric in VectorMetric.all_metrics():
                logger.info(f"Comparing HVP estimates using metric: {metric.name}")
                kfac_vs_full_hvp = metric.compute(
                    full_hvp, kfac_hvp, gradient_samples_2
                )
                ekfac_vs_full_hvp = metric.compute(
                    full_hvp, ekfac_hvp, gradient_samples_2
                )
                fim_vs_full_hvp = metric.compute(full_hvp, fim_hvp, gradient_samples_2)
                block_fim_vs_full_hvp = metric.compute(
                    full_hvp, block_fim_hvp, gradient_samples_2
                )
                gnh_vs_full_hvp = metric.compute(full_hvp, gnh_hvp, gradient_samples_2)
                block_hessian_vs_full_hvp = metric.compute(
                    full_hvp, block_hessian_hvp, gradient_samples_2
                )
                kfac_vs_gnh_hvp = metric.compute(gnh_hvp, kfac_hvp, gradient_samples_2)
                ekfac_vs_gnh_hvp = metric.compute(
                    gnh_hvp, ekfac_hvp, gradient_samples_2
                )
                fim_vs_gnh_hvp = metric.compute(gnh_hvp, fim_hvp, gradient_samples_2)
                block_fim_vs_gnh_hvp = metric.compute(
                    gnh_hvp, block_fim_hvp, gradient_samples_2
                )

                hvp_results[metric.name] = {
                    "hessian_comparison": {
                        "kfac": float(kfac_vs_full_hvp),
                        "ekfac": float(ekfac_vs_full_hvp),
                        "fim": float(fim_vs_full_hvp),
                        "block_fim": float(block_fim_vs_full_hvp),
                        "gnh": float(gnh_vs_full_hvp),
                        "block_hessian": float(block_hessian_vs_full_hvp),
                    },
                    "gnh_comparison": {
                        "kfac": float(kfac_vs_gnh_hvp),
                        "ekfac": float(ekfac_vs_gnh_hvp),
                        "fim": float(fim_vs_gnh_hvp),
                        "block_fim": float(block_fim_vs_gnh_hvp),
                    },
                }
                logger.info(f"Finished HVP comparison for metric: {metric.name}")

            # Compare IHVPs
            logging.info("Comparing Inverse Hessian-vector products (IHVPs).")
            ihvp_results = {}
            for metric in VectorMetric.all_metrics():
                logging.info(f"Comparing IHVP estimates using metric: {metric.name}")
                kfac_vs_full_ihvp = metric.compute(
                    full_ihvp, kfac_ihvp, gradient_samples_2
                )
                ekfac_vs_full_ihvp = metric.compute(
                    full_ihvp, ekfac_ihvp, gradient_samples_2
                )
                fim_vs_full_ihvp = metric.compute(
                    full_ihvp, fim_ihvp, gradient_samples_2
                )
                block_fim_vs_full_ihvp = metric.compute(
                    full_ihvp, block_fim_ihvp, gradient_samples_2
                )
                gnh_vs_full_ihvp = metric.compute(
                    full_ihvp, gnh_ihvp, gradient_samples_2
                )
                block_hessian_vs_full_ihvp = metric.compute(
                    full_ihvp, block_hessian_ihvp, gradient_samples_2
                )
                kfac_vs_gnh_ihvp = metric.compute(
                    gnh_ihvp, kfac_ihvp, gradient_samples_2
                )
                ekfac_vs_gnh_ihvp = metric.compute(
                    gnh_ihvp, ekfac_ihvp, gradient_samples_2
                )
                fim_vs_gnh_ihvp = metric.compute(gnh_ihvp, fim_ihvp, gradient_samples_2)
                block_fim_vs_gnh_ihvp = metric.compute(
                    gnh_ihvp, block_fim_ihvp, gradient_samples_2
                )
                ihvp_results[metric.name] = {
                    "hessian_comparison": {
                        "kfac": float(kfac_vs_full_ihvp),
                        "ekfac": float(ekfac_vs_full_ihvp),
                        "fim": float(fim_vs_full_ihvp),
                        "block_fim": float(block_fim_vs_full_ihvp),
                        "gnh": float(gnh_vs_full_ihvp),
                        "block_hessian": float(block_hessian_vs_full_ihvp),
                    },
                    "gnh_comparison": {
                        "kfac": float(kfac_vs_gnh_ihvp),
                        "ekfac": float(ekfac_vs_gnh_ihvp),
                        "fim": float(fim_vs_gnh_ihvp),
                        "block_fim": float(block_fim_vs_gnh_ihvp),
                    },
                }
                logger.info(f"Finished IHVP comparison for metric: {metric.name}")

            # Store Hessian analysis results
            experiment_result["hessian_analysis"] = {
                "damping": float(damping),
                "matrix_comparisons": matrix_results,
                "hvp_comparisons": hvp_results,
                "ihvp_comparisons": ihvp_results,
            }

            logger.info(f"Hessian analysis completed for {best_model_name}.")

            # Add experiment result to all_results
            all_results["experiments"].append(experiment_result)

    # Store best models summary
    all_results["best_models"] = best_model_names
    logger.info(f"All best models: {best_model_names}")

    # Save all results to JSON
    results_dir = "experiments/sweep_1/data/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = f"{results_dir}/experiment_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All results saved to {results_file}")
    print(f"Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    simple_run()
