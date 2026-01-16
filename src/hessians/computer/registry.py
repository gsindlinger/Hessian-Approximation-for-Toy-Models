from collections.abc import Callable
from typing import Dict, Tuple

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
    ApproximationData,
    BlockHessianData,
    DataActivationsGradients,
    EKFACData,
    FIMData,
    ModelContext,
)


class HessianComputerRegistry:
    REGISTRY: Dict[
        HessianApproximationMethod, Callable[..., HessianEstimator | HessianComputer]
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

    DATA_REGISTRY: Dict[
        HessianApproximationMethod, Tuple[type[ApproximationData], str]
    ] = {
        HessianApproximationMethod.KFAC: (EKFACData, "ekfac_data"),
        HessianApproximationMethod.EKFAC: (EKFACData, "ekfac_data"),
        HessianApproximationMethod.FIM: (FIMData, "fim_data"),
        HessianApproximationMethod.SHAMPOO: (EKFACData, "eshampoo_data"),
        HessianApproximationMethod.ESHAMPOO: (EKFACData, "eshampoo_data"),
        HessianApproximationMethod.TKFAC: (EKFACData, "etkfac_data"),
        HessianApproximationMethod.ETKFAC: (EKFACData, "etkfac_data"),
        HessianApproximationMethod.BLOCK_HESSIAN: (
            BlockHessianData,
            "block_hessian_data",
        ),
    }

    @staticmethod
    def get_computer(
        approximator: HessianApproximationMethod,
        compute_context: ModelContext
        | Tuple[DataActivationsGradients, DataActivationsGradients],
    ) -> HessianEstimator | HessianComputer:
        computer_cls = HessianComputerRegistry.REGISTRY[approximator]
        precomputed_data_type, precomputed_data_dir = (
            HessianComputerRegistry.get_precomputed_data_type(approximator)
        )

        # For exact Hessian, no precomputed data is needed
        if approximator == HessianApproximationMethod.EXACT:
            return computer_cls(
                compute_context=compute_context,
            )

        # For other approximators, pass precomputed data type if available
        return computer_cls(
            compute_context=compute_context,
            precomputed_data_directory=precomputed_data_dir,
            precomputed_data=precomputed_data_type() if precomputed_data_type else None,
        )

    @staticmethod
    def get_compute_context(
        approximator: HessianApproximationMethod,
        collector_data: Tuple[DataActivationsGradients, DataActivationsGradients],
        model_ctx: ModelContext,
    ):
        """Get the appropriate data for each approximator type."""
        if approximator in [
            HessianApproximationMethod.GNH,
            HessianApproximationMethod.BLOCK_HESSIAN,
            HessianApproximationMethod.EXACT,
        ]:
            return model_ctx
        else:
            return collector_data

    @staticmethod
    def get_precomputed_data_type(
        approximator: HessianApproximationMethod,
    ) -> Tuple[type[ApproximationData], str] | Tuple[None, None]:
        return HessianComputerRegistry.DATA_REGISTRY.get(approximator, (None, None))
