import json
import logging
import os
import time

from jax.random import PRNGKey

from src.config import Config, HessianApproximationConfig, ModelConfig
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import ConcreteDataset, Dataset
from src.utils.loss import mse_loss
from src.utils.metrics.full_matrix_metrics import MATRIX_METRICS
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.mlp import MLP
from src.utils.models.mlp_swiglu import MLPSwiGLU
from src.utils.optimizers import optimizer
from src.utils.train import (
    check_saved_model,
    evaluate,
    load_model_checkpoint,
    save_model_checkpoint,
    train_model,
)
from src.utils.utils import get_peak_bytes_in_use, hash_data

logger = logging.getLogger(__name__)


def run_concrete():
    # Define config which serves mostly as reference for paths and model metadata
    seed = 32
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Load dataset
    concrete_dataset_path = "experiments/sweep_1/data/datasets/concrete"
    dataset: ConcreteDataset = ConcreteDataset.load(
        directory=concrete_dataset_path, store_on_disk=True
    )
    dataset_name = "concrete"

    # Split dataset into train and validation sets

    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_dataset.inputs, train_dataset.targets
    val_inputs, val_targets = val_dataset.inputs, val_dataset.targets
    train_inputs, val_inputs = Dataset.normalize_data(
        train_data=train_inputs, test_data=val_inputs
    )
    train_targets, val_targets = Dataset.normalize_data(
        train_data=train_targets, test_data=val_targets
    )

    layer_settings = [
        [32, 32],
        # [128, 128],  # Setup of Grosse et al. (2023)
    ]
    optimizer_name = "adamw"

    # Hyperparameter grid
    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
    weight_decays = [0.0, 0.00001, 0.0001, 0.001, 0.01]

    # Initialize results structure
    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "seed": seed,
            "optimizer": optimizer_name,
            "dataset_path": concrete_dataset_path,
            "hyperparameter_grid": {
                "learning_rates": learning_rates,
                "weight_decays": weight_decays,
            },
        },
        "experiments": [],
    }

    best_model_names = []

    # Track memory at start of Hessian analysis
    logging.info(f"Peak memory usage at start: {get_peak_bytes_in_use()}")

    def split_dim_for_swiglu(x: int) -> tuple[int, int, int]:
        """
        Split a hidden dimension into (up_dim, gate_dim, down_dim) for SwiGLU,
        trying to keep the sizes as close to the regular MLP setup as possible.
        """
        base = x // 3
        rem = x % 3

        if rem == 0:
            return (base, base, base)
        elif rem == 1:
            return (base, base, base + 1)
        else:
            return (base + 1, base + 1, base)

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
            hidden_dim=[split_dim_for_swiglu(dim) for dim in hidden_layers],
            activation="swiglu",
        )

        for model in [model_mlp, model_mlp_swiglu]:
            # Initialize experiment result
            experiment_result = {
                "model_type": "MLP" if isinstance(model, MLP) else "MLPSwiGLU",
                "hidden_layers": hidden_layers,
                "num_params": model.num_params,
                "hyperparameter_search": [],
            }

            # PHASE 1: Hyperparameter grid search for model selection
            best_val_loss = float("inf")
            best_learning_rate = None
            best_weight_decay = None
            best_model_name = None

            for learning_rate in learning_rates:
                for weight_decay in weight_decays:
                    hashed_data = hash_data(
                        {
                            "hidden_layers": hidden_layers,
                            "optimizer": optimizer_name,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                        }
                    )
                    if isinstance(model, MLP):
                        model_name = f"mlp_hidden_{hashed_data}_lr_{learning_rate}_wd_{weight_decay}"
                    else:
                        model_name = f"mlp_swiglu_hidden_{hashed_data}_lr_{learning_rate}_wd_{weight_decay}"

                    model_directory = (
                        f"experiments/sweep_1/data/models/{dataset_name}/{model_name}/"
                    )
                    metadata = {
                        "model_name": model_name,
                        "model_directory": model_directory,
                        "metadata": {
                            "model_name": model_name,
                            "hidden_layers": hidden_layers,
                            "optimizer": optimizer_name,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
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
                                batch_size=128, seed=seed
                            ),
                            loss_fn=mse_loss,
                            optimizer=optimizer(
                                optimizer_name,
                                lr=learning_rate,
                                weight_decay=weight_decay,
                            ),
                            epochs=1000,
                        )
                        train_loss = loss_history[-1]
                        metadata["train_loss"] = train_loss

                    val_loss = evaluate(
                        model=model,
                        params=params,
                        inputs=val_inputs,
                        targets=val_targets,
                        loss_fn=mse_loss,
                    )
                    metadata["val_loss"] = val_loss

                    save_model_checkpoint(
                        model=model,
                        params=params,
                        directory=model_directory,
                        metadata=metadata,
                    )

                    # Track best hyperparameters
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_learning_rate = learning_rate
                        best_weight_decay = weight_decay
                        best_model_name = model_name

                    # Store hyperparameter search results
                    experiment_result["hyperparameter_search"].append(
                        {
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "model_name": model_name,
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                        }
                    )

                    logger.info(
                        f"Model {model_name} trained and evaluated. Val loss: {val_loss}"
                    )

            # Store hyperparameter search summary
            experiment_result["hyperparameter_search_summary"] = {
                "best_learning_rate": best_learning_rate,
                "best_weight_decay": best_weight_decay,
                "best_model_name": best_model_name,
                "best_val_loss": float(best_val_loss),
            }

            logger.info(
                f"Best model selected: {best_model_name} with val loss: {best_val_loss}"
            )
            logger.info(
                f"Peak memory usage during model training / loading: {get_peak_bytes_in_use()}"
            )
            best_model_names.append(best_model_name)

            # PHASE 2: Load best model and perform Hessian analysis
            logger.info(f"Starting Hessian analysis for best model: {best_model_name}")

            # Load best model
            best_model_directory = (
                f"experiments/sweep_1/data/models/{dataset_name}/{best_model_name}/"
            )
            params, _, _ = load_model_checkpoint(best_model_directory, model=model)

            train_dataset = Dataset(train_inputs, train_targets)

            # Generate gradient samples on training data
            gradient_samples_1 = sample_gradients(
                model=model,
                params=params,
                inputs=train_inputs,
                targets=train_targets,
                loss_fn=mse_loss,
                n_vectors=500,
                rng_key=PRNGKey(seed),
            )

            gradient_samples_2 = sample_gradients(
                model=model,
                params=params,
                inputs=train_inputs,
                targets=train_targets,
                loss_fn=mse_loss,
                n_vectors=500,
                rng_key=PRNGKey(seed + 1),
            )

            logger.info(
                f"Peak memory after gradient sampling: {get_peak_bytes_in_use()}"
            )

            # Generate pseudo-targets for EKFAC/FIM using training data
            collector_data_1 = train_dataset.replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=mse_loss,
                    rng_key=PRNGKey(seed),
                )
            )

            collector_data_2 = train_dataset.replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=mse_loss,
                    rng_key=PRNGKey(seed + 1),
                )
            )

            logger.info(
                f"Peak memory after pseudo-target generation: {get_peak_bytes_in_use()}"
            )

            # Collect activations and gradients
            collector_run_dir_1 = f"experiments/sweep_1/data/activation_gradient_collector/{best_model_name}/run1/"
            collector = CollectorActivationsGradients(model=model, params=params)
            collector.collect(
                collector_data_1.inputs,
                collector_data_1.targets,
                mse_loss,
                save_directory=collector_run_dir_1,
            )

            collector_run_dir_2 = f"experiments/sweep_1/data/activation_gradient_collector/{dataset_name}/{best_model_name}/run2/"
            collector.collect(
                collector_data_2.inputs,
                collector_data_2.targets,
                mse_loss,
                save_directory=collector_run_dir_2,
            )

            logger.info(
                f"Peak memory after activation and gradient collection: {get_peak_bytes_in_use()}"
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
                dataset_path=concrete_dataset_path,
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

            damping = 0.1

            # Memory checkpoint after EKFAC loading
            logger.info(
                f"Peak memory after EKFAC data loading: {get_peak_bytes_in_use()}"
            )

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
            logger.info(
                f"Peak memory after K-FAC computation: {get_peak_bytes_in_use()}"
            )

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
            logger.info(
                f"Peak memory after E-KFAC computation: {get_peak_bytes_in_use()}"
            )

            # Compute FIM
            model_context = ModelContext.create(
                dataset=train_dataset,
                model=model,
                params=params,
                loss_fn=mse_loss,
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
            logger.info(f"Peak memory after FIM computation: {get_peak_bytes_in_use()}")

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
            logger.info(
                f"Peak memory after Block FIM computation: {get_peak_bytes_in_use()}"
            )

            # Compute GNH
            gnh_computer = GNHComputer(compute_context=model_context)
            gnh = gnh_computer.estimate_hessian(damping=damping)
            logger.info(
                f"Peak memory after GNH Hessian estimation: {get_peak_bytes_in_use()}"
            )
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
            logger.info(f"Peak memory after GNH computation: {get_peak_bytes_in_use()}")

            # Compute the true Hessian
            hessian_computer = HessianComputer(compute_context=model_context)
            full_hessian = hessian_computer.compute_hessian(damping=damping)
            logger.info(
                f"Peak memory after Full Hessian estimation: {get_peak_bytes_in_use()}"
            )
            full_hvp = hessian_computer.compute_hvp(
                vectors=gradient_samples_1, damping=damping
            )
            full_ihvp = hessian_computer.compute_ihvp(
                vectors=gradient_samples_1, damping=damping
            )
            logger.info(
                f"Full Hessian, HVP, and IHVP computed with shapes: {full_hvp.shape}, {full_ihvp.shape}"
            )

            # Memory checkpoint after full Hessian computation
            logger.info(
                f"Peak memory after Full Hessian computation: {get_peak_bytes_in_use()}"
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

            # Memory checkpoint after Block Hessian computation
            logger.info(
                f"Peak memory after Block Hessian computation: {get_peak_bytes_in_use()}"
            )

            # Compare matrices
            matrix_results = {
                metric.value: {"hessian_comparison": {}, "gnh_comparison": {}}
                for metric in MATRIX_METRICS["all_matrix"]
            }
            matrix_metrics = MATRIX_METRICS["all_matrix"]

            method_computers: dict[str, HessianEstimator] = {
                "kfac": kfac_computer,
                "ekfac": ekfac_computer,
                "fim": fim_computer,
                "block_fim": block_fim_computer,
                "gnh": gnh_computer,
                "block_hessian": block_hessian_computer,
            }
            for method in ["kfac", "ekfac", "fim", "block_fim", "gnh", "block_hessian"]:
                hessian_estimate = method_computers[method].estimate_hessian(
                    damping=damping
                )

                for metric in matrix_metrics:
                    matrix_results[metric.value]["hessian_comparison"][method] = float(
                        metric.compute(full_hessian, hessian_estimate)
                    )

                if method in ["kfac", "ekfac", "fim", "block_fim"]:
                    for metric in matrix_metrics:
                        matrix_results[metric.value]["gnh_comparison"][method] = float(
                            metric.compute(gnh, hessian_estimate)
                        )

                logger.info(f"Finished matrix comparison for method: {method}")

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
                logger.info(
                    f"Peak memory during HVP comparison {metric.name}: {get_peak_bytes_in_use()}"
                )

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
                logger.info(
                    f"Peak memory during IHVP comparison {metric.name}: {get_peak_bytes_in_use()}"
                )

            # Store Hessian analysis results
            experiment_result["hessian_analysis"] = {
                "damping": float(damping),
                "matrix_comparisons": matrix_results,
                "hvp_comparisons": hvp_results,
                "ihvp_comparisons": ihvp_results,
            }

            logger.info(f"Hessian analysis completed for {best_model_name}.")
            logger.info(
                f"Peak memory at end of Hessian analysis for {best_model_name}: {get_peak_bytes_in_use()}"
            )

            # Add experiment result to all_results
            all_results["experiments"].append(experiment_result)

    # Store best models summary
    all_results["best_models"] = best_model_names
    logger.info(f"All best models: {best_model_names}")

    # Save all results to JSON
    results_dir = f"experiments/sweep_1/data/results/{dataset_name}/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file = f"{results_dir}/experiment_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All results saved to {results_file}")
    print(f"Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_concrete()
