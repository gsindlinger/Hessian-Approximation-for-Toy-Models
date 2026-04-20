from collections.abc import Callable
from typing import Dict

from src.config import HessianApproximationMethod
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.eshampoo import EShampooComputer
from src.hessians.computer.etkfac import ETKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer
from src.hessians.computer.shampoo import ShampooComputer
from src.hessians.computer.tkfac import TKFACComputer
from src.hessians.utils.data import (
    DataActivationsGradients,
    ModelContext,
)


class HessianComputerRegistry:
    REGISTRY: Dict[
        HessianApproximationMethod, Callable[..., HessianEstimator]
    ] = {
        HessianApproximationMethod.KFAC: KFACComputer,
        HessianApproximationMethod.EKFAC: EKFACComputer,
        HessianApproximationMethod.GNH: GNHComputer,
        HessianApproximationMethod.FIM: FIMComputer,
        HessianApproximationMethod.BLOCK_FIM: FIMBlockComputer,
        HessianApproximationMethod.BLOCK_HESSIAN: BlockHessianComputer,
        HessianApproximationMethod.EXACT: HessianComputer,
        HessianApproximationMethod.TKFAC: TKFACComputer,
        HessianApproximationMethod.ETKFAC: ETKFACComputer,
        HessianApproximationMethod.SHAMPOO: ShampooComputer,
        HessianApproximationMethod.ESHAMPOO: EShampooComputer,
    }

    # Directory names used for `LayerMatrix.save/load` under a shared
    # base directory.  Keeping the old strings for backwards compatibility
    # with existing on-disk caches.
    LAYER_MATRIX_DIRECTORY: Dict[HessianApproximationMethod, str] = {
        HessianApproximationMethod.KFAC: "kfac_layer_matrix",
        HessianApproximationMethod.EKFAC: "ekfac_layer_matrix",
        HessianApproximationMethod.FIM: "fim_layer_matrix",
        HessianApproximationMethod.BLOCK_FIM: "block_fim_layer_matrix",
        HessianApproximationMethod.BLOCK_HESSIAN: "block_hessian_layer_matrix",
        HessianApproximationMethod.GNH: "gnh_layer_matrix",
        HessianApproximationMethod.EXACT: "exact_hessian_layer_matrix",
        HessianApproximationMethod.TKFAC: "tkfac_layer_matrix",
        HessianApproximationMethod.ETKFAC: "etkfac_layer_matrix",
        HessianApproximationMethod.SHAMPOO: "shampoo_layer_matrix",
        HessianApproximationMethod.ESHAMPOO: "eshampoo_layer_matrix",
    }

    @staticmethod
    def get_computer(
        approximator: HessianApproximationMethod,
        compute_context: ModelContext | DataActivationsGradients,
        corr_context: DataActivationsGradients | None = None,
    ) -> HessianEstimator:
        computer_cls = HessianComputerRegistry.REGISTRY[approximator]
        layer_matrix_directory = HessianComputerRegistry.LAYER_MATRIX_DIRECTORY.get(
            approximator
        )
        kwargs = {
            "compute_context": compute_context,
            "layer_matrix_directory": layer_matrix_directory,
        }
        if corr_context is not None and issubclass(computer_cls, EKFACComputer):
            kwargs["corr_context"] = corr_context
        return computer_cls(**kwargs)

    @staticmethod
    def get_compute_context(
        approximator: HessianApproximationMethod,
        collector_data: DataActivationsGradients,
        model_ctx: ModelContext,
    ):
        """Get the appropriate compute context for each approximator type."""
        if approximator in [
            HessianApproximationMethod.GNH,
            HessianApproximationMethod.BLOCK_HESSIAN,
            HessianApproximationMethod.EXACT,
        ]:
            return model_ctx
        return collector_data
