from config.config import Config
from hessian_approximations.hessian.hessian import Hessian


def main():
    config = Config.parse_args()
    hessian_method = Hessian(full_config=config)
    hessian_matrix = hessian_method.compute_hessian()

    print("Hessian matrix shape:", hessian_matrix.shape)


if __name__ == "__main__":
    main()
