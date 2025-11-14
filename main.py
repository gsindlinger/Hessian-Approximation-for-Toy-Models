from config.config import Config
from config.hessian_approximation_config import KFACBuildConfig, KFACRunConfig
from hessian_approximations.kfac.kfac import KFAC


def main():
    config = Config.parse_args()
    print("Device being used:", config.device)

    kfac_build_config = KFACBuildConfig(use_pseudo_targets=True)
    kfac_run_config = KFACRunConfig(damping_lambda=0.1, use_eigenvalue_correction=True)

    hessian_method = KFAC.setup_with_run_and_build_config(
        full_config=config,
        build_config=kfac_build_config,
        run_config=kfac_run_config,
    )
    hessian_matrix = hessian_method.compute_hessian()

    print("Hessian matrix shape:", hessian_matrix.shape)


if __name__ == "__main__":
    print("Starting K-FAC Hessian computation...")
    main()
