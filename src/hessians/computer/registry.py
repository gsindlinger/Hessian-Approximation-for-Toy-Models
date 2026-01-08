from collections.abc import Callable
from typing import Dict

from src.config import HessianApproximator
from src.hessians.computer.computer import HessianEstimator
from src.hessians.computer.ekfac import EKFACComputer
from src.hessians.computer.fim import FIMComputer
from src.hessians.computer.fim_block import FIMBlockComputer
from src.hessians.computer.gnh import GNHComputer
from src.hessians.computer.hessian import HessianComputer
from src.hessians.computer.hessian_block import BlockHessianComputer
from src.hessians.computer.kfac import KFACComputer


class HessianComputerRegistry:
    REGISTRY: Dict[
        HessianApproximator, Callable[..., HessianEstimator | HessianComputer]
    ] = {
        HessianApproximator.KFAC: KFACComputer,
        HessianApproximator.EKFAC: EKFACComputer,
        HessianApproximator.GNH: GNHComputer,
        HessianApproximator.FIM: FIMComputer,
        HessianApproximator.BLOCK_FIM: FIMBlockComputer,
        HessianApproximator.BLOCK_HESSIAN: BlockHessianComputer,
        HessianApproximator.EXACT: HessianComputer,
    }

    @staticmethod
    def get_computer(
        approximator: HessianApproximator, *args, **kwargs
    ) -> HessianEstimator:
        computer_cls = HessianComputerRegistry.REGISTRY[approximator]
        return computer_cls(*args, **kwargs)  # type: ignore
