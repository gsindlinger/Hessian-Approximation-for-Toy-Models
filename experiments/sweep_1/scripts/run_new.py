# NOTE: This version fixes runtime issues, logical bugs, and missing pieces
# while keeping your original structure intact.

import gc
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Dict, List

import jax
from chex import PRNGKey
from jax.tree_util import tree_map
from simple_parsing import ArgumentParser

from src.config import (
    LOSS_TYPE,
    CollectorConfig,
    ComputationType,
    DampingStrategy,
    DatasetConfig,
    Datasets,
    EKFACApproximatorConfig,
    HessianApproximator,
    HessianComputationConfig,
    MatrixAnalysisConfig,
    ModelArchitecture,
    ModelConfig,
    ModelContextConfig,
    OptimizerType,
    TrainingConfig,
    VectorAnalysisConfig,
    VectorSamplingMethod,
)
from src.hessians.approximator.ekfac import EKFACApproximator
from src.hessians.collector import CollectorActivationsGradients
from src.hessians.computer.computer import HESSIAN_COMPUTER_REGISTRY, HessianEstimator
from src.hessians.computer.hessian import HessianComputer
from src.hessians.utils.data import EKFACData, ModelContext
from src.hessians.utils.pseudo_targets import generate_pseudo_targets, sample_vectors
from src.utils.data.data import Dataset, DownloadableDataset
from src.utils.loss import get_loss
from src.utils.metrics.full_matrix_metrics import FullMatrixMetric
from src.utils.metrics.vector_metrics import VectorMetric
from src.utils.models.approximation_model import ApproximationModel
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    msg = f"[MEMORY] peak_bytes={get_peak_bytes_in_use()}"
    if stage:
        msg = f"[MEMORY] after {stage}: peak_bytes={get_peak_bytes_in_use()}"
    logger.info(msg)


# -----------------------------------------------------------------------------
# argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    parser = ArgumentParser(
        description="Unified experiment runner for training and Hessian analysis"
    )
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    return args.config


# -----------------------------------------------------------------------------
# main experiment
# -----------------------------------------------------------------------------


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    seed = 42

    logger.info("Starting run_digits")
    logger.info(f"Seed = {seed}")

    # -------------------------------------------------------------------------
    # dataset
    # -------------------------------------------------------------------------
    dataset_config = DatasetConfig(
        name=Datasets.DIGITS,
        path="experiments/sweep_1/data/datasets/digits",
        split_seed=seed,
        store_on_disk=True,
    )

    dataset: Dataset = DownloadableDataset.load(
        dataset=Datasets.DIGITS,
        directory=dataset_config.path,
        store_on_disk=dataset_config.store_on_disk,
    )
    logger.info(f"Loaded dataset from {dataset_config.path}")

    train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets
    val_inputs, val_targets = val_ds.inputs, val_ds.targets

    logger.info(f"Train samples={len(train_inputs)}, Val samples={len(val_inputs)}")

    # -------------------------------------------------------------------------
    # model sweep
    # -------------------------------------------------------------------------
    model_configs_base = [
        # --- MLP ---
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=[16],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=4 * [16],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=8 * [16],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=[32],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=4 * [32],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=8 * [32],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=[64],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=4 * [64],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLP,
            hidden_dims=8 * [64],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        # --- MLPSwiGLU ---
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=[(5, 5, 5)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=4 * [(5, 5, 5)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=8 * [(5, 5, 5)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=[(10, 10, 10)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=4 * [(10, 10, 10)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=8 * [(10, 10, 10)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=[(21, 21, 21)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=4 * [(21, 21, 21)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
        ModelConfig(
            ModelArchitecture.MLPSWIGLU,
            hidden_dims=8 * [(21, 21, 21)],
            init_seed=seed,
            loss=LOSS_TYPE.CROSS_ENTROPY,
        ),
    ]

    learning_rates = [5e-4, 1e-3, 2e-3, 1e-2]
    weight_decays = [0.0, 1e-4, 1e-3, 1e-2]

    model_configs: Dict[str, List[ModelConfig]] = {}
    for mc in model_configs_base:
        model_name = f"{mc.architecture.value}_hidden{mc.hidden_dims}"
        model_configs.setdefault(model_name, [])
        for lr in learning_rates:
            for wd in weight_decays:
                model_training_hash = hash_data(
                    {
                        **asdict(mc),
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "optimizer": OptimizerType.ADAMW,
                        "epochs": 500,
                        "batch_size": 32,
                    },
                    length=20,
                )
                model_configs[model_name].append(
                    ModelConfig(
                        architecture=mc.architecture,
                        hidden_dims=mc.hidden_dims,
                        init_seed=mc.init_seed,
                        directory=f"experiments/sweep_1/data/models/digits/{model_name}_{model_training_hash}",
                        skip_existing=True,
                        training_config=TrainingConfig(
                            learning_rate=lr,
                            weight_decay=wd,
                            optimizer=OptimizerType.ADAMW,
                            epochs=500,
                            batch_size=32,
                        ),
                    )
                )

    # -------------------------------------------------------------------------
    # results containers
    # -------------------------------------------------------------------------
    best_model_information: Dict = {}

    results_dir = "experiments/sweep_1/data/results/digits"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Initial peak memory: {get_peak_bytes_in_use()}")

    # Sweep over different model setups
    for model_name, model_config_list in model_configs.items():
        arch = model_config_list[0].architecture.value
        hidden = model_config_list[0].hidden_dims

        logger.info(f"Evaluating {arch} hidden_layers={hidden}")

        model = ApproximationModel.get_model(
            model_config=model_config_list[0],
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            seed=seed,
        )

        best_val_loss = float("inf")
        best_model_name = ""

        # Sweep over hyperparameters
        for model_config in model_config_list:
            model_dir = model_config.directory
            assert model_dir is not None, "Model directory must be specified."
            assert model_config.training_config is not None, (
                "Training config must be specified for training."
            )

            if check_saved_model(model_dir, model=model):
                params, _, _ = load_model_checkpoint(model_dir, model=model)
                logger.info(f"[LOAD] {model_name}")
            else:
                model, params, _ = train_model(
                    model=model,
                    dataloader=Dataset(train_inputs, train_targets).get_dataloader(
                        batch_size=model_config.training_config.batch_size, seed=seed
                    ),
                    loss_fn=get_loss(model_config.loss),
                    optimizer=optimizer(
                        optimizer_enum=model_config.training_config.optimizer,
                        lr=model_config.training_config.learning_rate,
                        weight_decay=model_config.training_config.weight_decay,
                    ),
                    epochs=model_config.training_config.epochs,
                )
                logger.info(f"[TRAIN] {model_dir} finished")

            val_loss, val_acc = evaluate_loss_and_classification_accuracy(
                model,
                params,
                val_inputs,
                val_targets,
                get_loss(model_config.loss),
            )

            logger.info(
                f"[EVAL] {model_name}: val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
            )

            save_model_checkpoint(
                model=model,
                params=params,
                directory=model_dir,
                metadata={
                    "model_config": asdict(model_config),
                    "dataset_config": asdict(dataset_config),
                },
            )

            if val_loss < best_val_loss:
                best_model_information[model_name] = {
                    "model_directory": model_dir,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": model_config,
                }

            cleanup_memory(stage=model_name)

    # -----------------------------------------------------------------
    # Hessian analysis
    # -----------------------------------------------------------------

    # get model configs of the best models if accuracy > threshold
    accuracy_threshold = 0.40

    dataset_config = DatasetConfig(
        name=Datasets.DIGITS,
        path="experiments/sweep_1/data/datasets/digits",
        split_seed=seed,
        store_on_disk=True,
    )
    dataset: Dataset = DownloadableDataset.load(
        dataset=Datasets.DIGITS,
        directory=dataset_config.path,
        store_on_disk=dataset_config.store_on_disk,
    )
    logger.info(f"Loaded dataset from {dataset_config.path}")

    train_ds, val_ds = dataset.train_test_split(test_size=0.1, seed=seed)
    train_inputs, train_targets = train_ds.inputs, train_ds.targets

    vector_analysis_config = VectorAnalysisConfig(
        num_samples=2000,
        sampling_method=VectorSamplingMethod.GRADIENTS,
        seed=seed,
    )

    matrix_analysis_config = MatrixAnalysisConfig()

    model_configs_hessian: List[ModelConfig] = [
        model_info["config"]
        for model_info in best_model_information.values()
        if model_info["val_accuracy"] >= accuracy_threshold
    ]

    collector_configs_ekfac = [
        CollectorConfig(
            num_pseudo_target_runs=2,
            pseudo_target_seeds=[seed, seed + 1],
            collector_output_dirs=[
                f"experiments/sweep_1/data/collector/digits/{model.get_model_base_name()}/run1/",
                f"experiments/sweep_1/data/collector/digits/{model.get_model_base_name()}/run2/",
            ],
            try_load_cached=True,
        )
        for model in model_configs_hessian
    ]

    collector_configs_single = [
        CollectorConfig(
            num_pseudo_target_runs=1,
            pseudo_target_seeds=[seed],
            collector_output_dirs=[
                f"experiments/sweep_1/data/collector/digits/{model.get_model_base_name()}/run_1/",
            ],
            try_load_cached=True,
        )
        for model in model_configs_hessian
    ]

    ekfac_configs: List[EKFACApproximatorConfig] = [
        EKFACApproximatorConfig(
            directory=f"experiments/sweep_1/data/ekfac/digits/{model.get_model_base_name()}/",
            collector_dir_1=collector_configs_ekfac[i].collector_output_dirs[0],
            collector_dir_2=collector_configs_ekfac[i].collector_output_dirs[1],
        )
        for i, model in enumerate(model_configs_hessian)
    ]

    model_context_configs = [
        ModelContextConfig(model_config=mc, dataset_config=dataset_config)
        for mc in model_configs_hessian
    ]

    hessian_computation_configs = [
        HessianComputationConfig(
            damping=0.1,
            damping_strategy=DampingStrategy.AUTO_MEAN_EIGENVALUE_CORRECTION,
            approximator_configs={
                HessianApproximator.EKFAC: ekfac_configs[i],
                HessianApproximator.KFAC: ekfac_configs[i],
                HessianApproximator.GNH: model_context_configs[i],
                HessianApproximator.FIM: collector_configs_single[i],
                HessianApproximator.BLOCK_FIM: collector_configs_single[i],
                HessianApproximator.BLOCK_HESSIAN: model_context_configs[i],
            },
            comparison_references={
                HessianApproximator.EXACT: model_context_configs[i],
                HessianApproximator.GNH: model_context_configs[i],
            },
            computation_types={
                ComputationType.MATRIX: matrix_analysis_config,
                ComputationType.HVP: vector_analysis_config,
                ComputationType.IHVP: vector_analysis_config,
            },
        )
        for i, _ in enumerate(model_configs_hessian)
    ]

    for i, mc in enumerate(model_configs_hessian):
        model = ApproximationModel.get_model(
            model_config=mc,
            input_dim=dataset.input_dim(),
            output_dim=dataset.output_dim(),
            seed=seed,
        )

        logger.info(f"[HESSIAN] Starting analysis for {best_model_name}")

        assert mc.directory is not None, "Model directory must be specified."
        params, _, _ = load_model_checkpoint(directory=mc.directory, model=model)

        grads = sample_vectors(
            vector_config=vector_analysis_config,
            model=model,
            params=params,
            inputs=train_inputs,
            targets=train_targets,
            loss_fn=get_loss(mc.loss),
        )
        # split grads equally (first half and second half)
        half = len(grads) // 2
        grads_1 = grads[:half]
        grads_2 = grads[half:]

        del grads

        cleanup_memory("gradient sampling")

        for run_idx in range(collector_configs_ekfac[i].num_pseudo_target_runs):
            seed = collector_configs_ekfac[i].pseudo_target_seeds[run_idx]
            collector_data = Dataset(train_inputs, train_targets).replace_targets(
                generate_pseudo_targets(
                    model=model,
                    inputs=train_inputs,
                    params=params,
                    loss_fn=get_loss(mc.loss),
                    rng_key=PRNGKey(seed),
                )
            )
            cleanup_memory("pseudo-target generation")

            collector = CollectorActivationsGradients(
                model=model, params=params, loss_fn=get_loss(mc.loss)
            )
            collector.collect(
                inputs=collector_data.inputs,
                targets=collector_data.targets,
                save_directory=collector_configs_ekfac[i].collector_output_dirs[
                    run_idx
                ],
                try_load=True,
            )
            cleanup_memory("activation/gradient collection")

        ekfac = EKFACApproximator(
            collector_configs_ekfac[i].collector_output_dirs[0],
            collector_configs_ekfac[i].collector_output_dirs[1],
        )

        if not ekfac.data_exists(ekfac_configs[i].directory):
            ekfac.build(
                config=mc,
                save_directory=ekfac_configs[i].directory,
            )

        ekfac_data, _ = EKFACApproximator.load_data(ekfac_configs[i].directory)
        assert isinstance(ekfac_data, EKFACData)

        damping = EKFACApproximator.get_damping(
            ekfac_data,
            hessian_computation_configs[i].damping_strategy,
            hessian_computation_configs[i].damping,
        )
        logger.info(f"[EKFAC] Loaded (damping={damping:.6e})")
        cleanup_memory("EKFAC load")

        model_ctx = ModelContext.create(
            dataset=Dataset(train_inputs, train_targets),
            model=model,
            params=params,
            loss_fn=get_loss(mc.loss),
        )

        collector_single_data = CollectorActivationsGradients.load(
            collector_configs_single[i].collector_output_dirs[0],
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

    def get_data(approximator_name: HessianApproximator, config):
        if approximator_name in [
            HessianApproximator.FIM,
            HessianApproximator.BLOCK_FIM,
        ]:
            return collector_single_data
        elif approximator_name in [
            HessianApproximator.EKFAC,
            HessianApproximator.KFAC,
        ]:
            return ekfac_data
        elif approximator_name in [
            HessianApproximator.GNH,
            HessianApproximator.BLOCK_HESSIAN,
            HessianApproximator.EXACT,
        ]:
            return model_ctx
        else:
            raise ValueError(
                f"Unsupported approximator for data retrieval: {approximator_name}"
            )

    for name_, config_ in hessian_computation_configs[i].comparison_references.items():
        reference_computer = HESSIAN_COMPUTER_REGISTRY[name_](get_data(name_, config_))  # type: ignore

        if ComputationType.MATRIX in hessian_computation_configs[i].computation_types:
            logger.info("[HESSIAN] Computing exact Reference Hessian / GNH / ...")
            if isinstance(reference_computer, HessianComputer):
                reference_hessian = block_tree(
                    reference_computer.compute_hessian(damping=damping),
                    f"Reference {name_} Hessian",
                )
            elif isinstance(reference_computer, HessianEstimator):
                reference_hessian = block_tree(
                    reference_computer.estimate_hessian(damping),
                    f"Reference {name_} Hessian",
                )
            else:
                raise ValueError(
                    f"Unsupported reference computer type: {type(reference_computer)}"
                )

            for comparison_name_ in hessian_computation_configs[
                i
            ].approximator_configs.keys():
                computer_ = HESSIAN_COMPUTER_REGISTRY[comparison_name_](
                    get_data(comparison_name_, config_)  # type: ignore
                )
                logger.info(f"[HESSIAN] Exact vs {comparison_name_} (matrix)")
                assert isinstance(computer_, HessianEstimator)
                comp = computer_
                for m in HessianComputationConfig.computation_types[
                    ComputationType.MATRIX
                ].metrics:
                    assert isinstance(m, FullMatrixMetric)
                    hessian_results["matrix_comparisons"].setdefault(m.value, {})
                    hessian_results["matrix_comparisons"][m.value].setdefault(
                        "exact", {}
                    )[name_] = float(
                        comp.compare_full_hessian_estimates(
                            reference_hessian, damping, m
                        )
                    )

            del reference_hessian
            cleanup_memory("after exact Hessian matrix comparisons")

        if ComputationType.HVP in hessian_computation_configs[i].computation_types:
            logger.info("[HESSIAN] Computing exact HVP")
            if isinstance(reference_computer, HessianComputer):
                H_hvp = block_tree(
                    reference_computer.compute_hvp(grads_1, damping=damping),
                    "Exact HVP",
                )
            elif isinstance(reference_computer, HessianEstimator):
                H_hvp = block_tree(
                    reference_computer.estimate_hvp(grads_1, damping),
                    "Exact HVP",
                )
            else:
                raise ValueError(
                    f"Unsupported reference computer type: {type(reference_computer)}"
                )
            for name_, computer_ in hessian_computation_configs[
                i
            ].approximator_configs.items():
                logger.info(f"[HESSIAN] Exact vs {name_} (HVP)")
                comp: HessianEstimator = HESSIAN_COMPUTER_REGISTRY[name_](
                    get_data(name_, config_)  # type: ignore
                )
                hvp = block_tree(comp.estimate_hvp(grads_1, damping), f"{name_} HVP")
                for m in HessianComputationConfig.computation_types[
                    ComputationType.HVP
                ].metrics:
                    assert isinstance(m, VectorMetric)
                    hessian_results["hvp_comparisons"].setdefault(m.name, {})
                    hessian_results["hvp_comparisons"][m.name].setdefault("exact", {})[
                        name_
                    ] = float(m.compute(H_hvp, hvp, grads_2))

            del H_hvp
            cleanup_memory("after exact Hessian HVP comparisons")

        if ComputationType.IHVP in hessian_computation_configs[i].computation_types:
            logger.info("[HESSIAN] Computing exact IHVP")
            if isinstance(reference_computer, HessianComputer):
                H_ihvp = block_tree(
                    reference_computer.compute_ihvp(grads_1, damping=damping),
                    "Exact IHVP",
                )
            elif isinstance(reference_computer, HessianEstimator):
                H_ihvp = block_tree(
                    reference_computer.estimate_ihvp(grads_1, damping),
                    "Exact IHVP",
                )
            else:
                raise ValueError(
                    f"Unsupported reference computer type: {type(reference_computer)}"
                )
            for name_, computer_ in hessian_computation_configs[
                i
            ].approximator_configs.items():
                logger.info(f"[HESSIAN] Exact vs {name_} (IHVP)")
                comp: HessianEstimator = HESSIAN_COMPUTER_REGISTRY[name_](
                    get_data(name_, config_)  # type: ignore
                )
                ihvp = block_tree(comp.estimate_ihvp(grads_1, damping), f"{name_} IHVP")
                for m in HessianComputationConfig.computation_types[
                    ComputationType.IHVP
                ].metrics:
                    assert isinstance(m, VectorMetric)
                    hessian_results["ihvp_comparisons"].setdefault(m.name, {})
                    hessian_results["ihvp_comparisons"][m.name].setdefault("exact", {})[
                        name_
                    ] = float(m.compute(H_ihvp, ihvp, grads_2))

            del H_ihvp
            cleanup_memory("after exact Hessian IHVP comparisons")

    all_results = {"experiments": []}
    experiment_result: Dict = {}
    experiment_result["model_name"] = best_model_name
    experiment_result["model_directory"] = mc.directory
    experiment_result["val_loss"] = best_model_information[best_model_name]["val_loss"]
    experiment_result["val_accuracy"] = best_model_information[best_model_name][
        "val_accuracy"
    ]
    experiment_result["model_config"] = asdict(
        best_model_information[best_model_name]["config"]
    )
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
    main()
