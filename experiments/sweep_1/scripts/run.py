import gc
import json
import logging
import os
import time

import jax
from jax.random import PRNGKey
from jax.tree_util import tree_map

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
from src.hessians.utils.data import DataActivationsGradients, EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_gradients
from src.utils.data.data import Dataset, DigitsDataset
from src.utils.loss import cross_entropy_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
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

# -----------------------------------------------------------------------------
# logging / jax config
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)  # enable if you suspect recompiles


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------


def block_tree(x, name: str):
    logger.info(f"[SYNC] Blocking on {name}")
    try:
        x = tree_map(lambda y: y.block_until_ready(), x)
    except Exception:
        logger.exception(f"[SYNC] Failure while blocking on {name}")
        raise
    logger.info(f"[SYNC] Completed {name}")
    return x


def cleanup_memory(stage: str | None = None):
    gc.collect()
    jax.clear_caches()
    if stage:
        logger.info(f"[MEMORY] after {stage}: peak_bytes={get_peak_bytes_in_use()}")
    else:
        logger.info(f"[MEMORY] peak_bytes={get_peak_bytes_in_use()}")


def split_dim_for_swiglu(x: int) -> tuple[int, int, int]:
    base = x // 3
    return (2 * base, 2 * base, 2 * base)


# -----------------------------------------------------------------------------
# main experiment
# -----------------------------------------------------------------------------


def run_digits():
    seed = 124
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    logger.info("Starting run_digits")
    logger.info(f"Seed = {seed}")

    # -------------------------------------------------------------------------
    # dataset
    # -------------------------------------------------------------------------
    digits_dataset_path = "experiments/sweep_1/data/datasets/digits"
    dataset: DigitsDataset = DigitsDataset.load(
        directory=digits_dataset_path, store_on_disk=True
    )
    logger.info(f"Loaded dataset from {digits_dataset_path}")

    train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    val_inputs, val_targets = val_ds.inputs, val_ds.targets

    logger.info(f"Train samples={len(train_inputs)}, Val samples={len(val_inputs)}")

    # -------------------------------------------------------------------------
    # sweep config
    # -------------------------------------------------------------------------
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
    optimizer_name = "adamw"
    learning_rates = [1e-3, 3e-3, 1e-2]
    weight_decays = [0.0, 0.01, 0.1, 0.5]

    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "seed": seed,
            "optimizer": optimizer_name,
            "dataset_path": digits_dataset_path,
            "hyperparameter_grid": {
                "learning_rates": learning_rates,
                "weight_decays": weight_decays,
            },
        },
        "best_models": [],
        "experiments": [],
    }

    results_dir = "experiments/sweep_1/data/results/digits"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Initial peak memory: {get_peak_bytes_in_use()}")

    # -------------------------------------------------------------------------
    # sweep
    # -------------------------------------------------------------------------
    for hidden_layers in layer_settings:
        logger.info(f"Evaluating hidden_layers={hidden_layers}")

        models = [
            (
                "MLP",
                MLP(
                    input_dim=dataset.input_dim(),
                    output_dim=dataset.output_dim(),
                    hidden_dim=hidden_layers,
                    seed=seed,
                ),
                hidden_layers,
            ),
            (
                "MLPSwiGLU",
                MLPSwiGLU(
                    input_dim=dataset.input_dim(),
                    output_dim=dataset.output_dim(),
                    hidden_dim=[split_dim_for_swiglu(d) for d in hidden_layers],
                    activation="swiglu",
                    seed=seed,
                ),
                [split_dim_for_swiglu(d) for d in hidden_layers],
            ),
        ]

        for model_type, model, experiment_hidden_layers in models:
            logger.info(
                f"Created model {model_type} "
                f"(params={model.num_params}, hidden={experiment_hidden_layers})"
            )

            experiment_result = {
                "model_type": model_type,
                "hidden_layers": experiment_hidden_layers,
                "num_params": model.num_params,
                "hyperparameter_search": [],
            }

            # -----------------------------------------------------------------
            # hyperparameter search
            # -----------------------------------------------------------------
            logger.info(f"[HP SEARCH] Starting for {model_type}")
            best_val_loss = float("inf")
            best_model_name = None

            for lr in learning_rates:
                for wd in weight_decays:
                    cfg_hash = hash_data(
                        {
                            "hidden_layers": experiment_hidden_layers,
                            "optimizer": optimizer_name,
                            "lr": lr,
                            "wd": wd,
                            "seed": seed,
                        }
                    )
                    model_name = (
                        f"{model_type.lower()}_{cfg_hash}_lr{lr}_wd{wd}_seed{seed}"
                    )
                    model_dir = f"experiments/sweep_1/data/models/digits/{model_name}/"

                    if check_saved_model(model_dir, model=model):
                        params, _, meta = load_model_checkpoint(model_dir, model=model)
                        train_loss = meta["train_loss"]
                        logger.info(f"[LOAD] {model_name}")
                    else:
                        model, params, loss_hist = train_model(
                            model,
                            Dataset(train_inputs, train_targets).get_dataloader(
                                batch_size=32, seed=seed
                            ),
                            cross_entropy_loss,
                            optimizer(optimizer_name, lr=lr, weight_decay=wd),
                            epochs=500,
                        )
                        train_loss = loss_hist[-1]
                        logger.info(f"[TRAIN] {model_name} finished")

                    val_loss, val_acc = evaluate_loss_and_classification_accuracy(
                        model, params, val_inputs, val_targets, cross_entropy_loss
                    )

                    logger.info(
                        f"[EVAL] {model_name}: "
                        f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
                    )

                    save_model_checkpoint(
                        model=model,
                        params=params,
                        directory=model_dir,
                        metadata={
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "val_accuracy": float(val_acc),
                        },
                    )

                    experiment_result["hyperparameter_search"].append(
                        {
                            "learning_rate": lr,
                            "weight_decay": wd,
                            "model_name": model_name,
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "val_accuracy": float(val_acc),
                        }
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_name = model_name

            logger.info(
                f"[HP SEARCH] Best model for {model_type}: "
                f"{best_model_name} (val_loss={best_val_loss:.6f})"
            )

            experiment_result["hyperparameter_search_summary"] = {
                "best_model_name": best_model_name,
                "best_val_loss": float(best_val_loss),
            }
            all_results["best_models"].append(best_model_name)

            # -----------------------------------------------------------------
            # Hessian analysis
            # -----------------------------------------------------------------
            logger.info(f"[HESSIAN] Starting analysis for {best_model_name}")

            best_dir = f"experiments/sweep_1/data/models/digits/{best_model_name}/"
            params, _, _ = load_model_checkpoint(best_dir, model=model)

            grads_1 = sample_gradients(
                model,
                params,
                train_inputs,
                train_targets,
                cross_entropy_loss,
                1000,
                PRNGKey(seed + 12345),
            )
            grads_2 = sample_gradients(
                model,
                params,
                train_inputs,
                train_targets,
                cross_entropy_loss,
                1000,
                PRNGKey(seed + 1234),
            )
            cleanup_memory("gradient sampling")

            collector_data_1 = Dataset(train_inputs, train_targets).replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=cross_entropy_loss,
                    rng_key=PRNGKey(seed),
                )
            )
            cleanup_memory("pseudo-target generation")

            collector_dir_1 = (
                f"experiments/sweep_1/data/collector/{best_model_name}/run1/"
            )
            collector_1 = CollectorActivationsGradients(
                model=model, params=params, loss_fn=cross_entropy_loss
            )
            collected_activations_gradients_1: DataActivationsGradients = (
                collector_1.collect(
                    inputs=collector_data_1.inputs,
                    targets=collector_data_1.targets,
                    save_directory=collector_dir_1,
                    try_load=True,
                )
            )
            cleanup_memory("activation/gradient collection")

            collector_data_2 = Dataset(train_inputs, train_targets).replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=cross_entropy_loss,
                    rng_key=PRNGKey(seed + 1),
                )
            )
            collector_dir_2 = (
                f"experiments/sweep_1/data/collector/{best_model_name}/run2/"
            )
            collector_2 = CollectorActivationsGradients(
                model=model, params=params, loss_fn=cross_entropy_loss
            )
            collector_2.collect(
                inputs=collector_data_2.inputs,
                targets=collector_data_2.targets,
                save_directory=collector_dir_2,
                try_load=True,
            )
            del collector_2
            cleanup_memory("activation/gradient collection run 2")

            ekfac_dir = f"experiments/sweep_1/data/hessian/{best_model_name}/"
            ekfac = EKFACApproximator(collector_dir_1, collector_dir_1)
            ekfac.build(
                Config(
                    digits_dataset_path,
                    ModelConfig(str(best_model_name), best_dir, {}),
                    HessianApproximationConfig("ekfac", ekfac_dir),
                ),
                ekfac_dir,
            )

            ekfac_data, _ = EKFACApproximator.load_data(ekfac_dir)
            assert isinstance(ekfac_data, EKFACData)
            damping = ekfac_data.mean_eigenvalues_aggregated * 0.1

            logger.info(f"[EKFAC] Loaded (damping={damping:.6e})")
            cleanup_memory("EKFAC load")

            model_ctx = ModelContext.create(
                dataset=Dataset(train_inputs, train_targets),
                model=model,
                params=params,
                loss_fn=cross_entropy_loss,
            )

            hessian_results = {
                "damping": damping,
                "matrix_comparisons": {},
                "hvp_comparisons": {},
                "ihvp_comparisons": {},
            }

            # -----------------------------------------------------------------
            # Phase 1: exact Hessian
            # -----------------------------------------------------------------
            logger.info("[HESSIAN] Phase 1: exact Hessian reference")

            hessian_comparison_computers: dict[str, HessianEstimator] = {
                "kfac": KFACComputer(ekfac_data),
                "ekfac": EKFACComputer(ekfac_data),
                "gnh": GNHComputer(model_ctx),
                "fim": FIMComputer(collected_activations_gradients_1),
                "block_fim": FIMBlockComputer(collected_activations_gradients_1),
                "block_hessian": BlockHessianComputer(model_ctx),
            }

            exact_hessian = HessianComputer(model_ctx)

            logger.info("[HESSIAN] Computing exact Hessian")
            H = block_tree(
                exact_hessian.compute_hessian(damping=damping), "Exact Hessian"
            )

            for name_, computer_ in hessian_comparison_computers.items():
                logger.info(f"[HESSIAN] Exact vs {name_} (matrix)")
                comp: HessianEstimator = computer_
                for m in FullMatrixMetric:
                    hessian_results["matrix_comparisons"].setdefault(m.value, {})
                    hessian_results["matrix_comparisons"][m.value].setdefault(
                        "exact", {}
                    )[name_] = float(comp.compare_full_hessian_estimates(H, damping, m))

            del H
            cleanup_memory("after exact Hessian matrix comparisons")

            logger.info("[HESSIAN] Computing exact HVP")
            H_hvp = block_tree(
                exact_hessian.compute_hvp(grads_1, damping=damping), "Exact HVP"
            )

            logger.info("[HESSIAN] Computing exact IHVP")
            H_ihvp = block_tree(
                exact_hessian.compute_ihvp(grads_1, damping=damping), "Exact IHVP"
            )

            for name_, computer_ in hessian_comparison_computers.items():
                logger.info(f"[HESSIAN] Exact vs {name_} (vector)")
                comp: HessianEstimator = computer_
                hvp = block_tree(comp.estimate_hvp(grads_1, damping), f"{name_} HVP")
                ihvp = block_tree(comp.estimate_ihvp(grads_1, damping), f"{name_} IHVP")
                for m in VectorMetric.all_metrics():
                    hessian_results["hvp_comparisons"].setdefault(m.name, {})
                    hessian_results["hvp_comparisons"][m.name].setdefault("exact", {})[
                        name_
                    ] = float(m.compute(H_hvp, hvp, grads_2))
                    hessian_results["ihvp_comparisons"].setdefault(m.name, {})
                    hessian_results["ihvp_comparisons"][m.name].setdefault("exact", {})[
                        name_
                    ] = float(m.compute(H_ihvp, ihvp, grads_2))

            del exact_hessian, H_hvp, H_ihvp
            cleanup_memory("after exact Hessian")

            # -----------------------------------------------------------------
            # Phase 2: GNH
            # -----------------------------------------------------------------
            logger.info("[HESSIAN] Phase 2: GNH reference")

            gnh_comparison_computers: dict[str, HessianEstimator] = {
                "kfac": KFACComputer(ekfac_data),
                "ekfac": EKFACComputer(ekfac_data),
                "fim": FIMComputer(collected_activations_gradients_1),
                "block_fim": FIMBlockComputer(collected_activations_gradients_1),
            }

            gnh = GNHComputer(model_ctx)

            logger.info("[HESSIAN] Computing GNH Hessian")
            gnh_hessian = block_tree(gnh.estimate_hessian(damping), "GNH Hessian")

            for name_, computer_ in gnh_comparison_computers.items():
                logger.info(f"[HESSIAN] GNH vs {name_} (matrix)")
                comp: HessianEstimator = computer_
                for m in FullMatrixMetric:
                    hessian_results["matrix_comparisons"][m.value].setdefault(
                        "gnh", {}
                    )[name_] = float(
                        comp.compare_full_hessian_estimates(gnh_hessian, damping, m)
                    )

            del gnh_hessian
            cleanup_memory("after GNH matrix comparisons")

            logger.info("[HESSIAN] Computing GNH HVP")
            gnh_hvp = block_tree(gnh.estimate_hvp(grads_1, damping), "GNH HVP")

            logger.info("[HESSIAN] Computing GNH IHVP")
            gnh_ihvp = block_tree(gnh.estimate_ihvp(grads_1, damping), "GNH IHVP")

            for name_, cls in {"kfac": KFACComputer, "ekfac": EKFACComputer}.items():
                logger.info(f"[HESSIAN] GNH vs {name_} (vector)")
                comp: HessianEstimator = cls(ekfac_data)
                hvp = block_tree(comp.estimate_hvp(grads_1, damping), f"{name_} HVP")
                ihvp = block_tree(comp.estimate_ihvp(grads_1, damping), f"{name_} IHVP")
                for m in VectorMetric.all_metrics():
                    hessian_results["hvp_comparisons"][m.name].setdefault("gnh", {})[
                        name_
                    ] = float(m.compute(gnh_hvp, hvp, grads_2))
                    hessian_results["ihvp_comparisons"][m.name].setdefault("gnh", {})[
                        name_
                    ] = float(m.compute(gnh_ihvp, ihvp, grads_2))

            del gnh, gnh_hvp, gnh_ihvp
            cleanup_memory("after GNH vector comparisons")

            experiment_result["hessian_analysis"] = hessian_results
            all_results["experiments"].append(experiment_result)

            logger.info(f"[HESSIAN] Completed for {best_model_name}")

    # -------------------------------------------------------------------------
    # save results
    # -------------------------------------------------------------------------
    out_file = f"{results_dir}/experiment_full_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All results saved to {out_file}")
    print(f"Saved results to {out_file}")

    return all_results


if __name__ == "__main__":
    run_digits()
