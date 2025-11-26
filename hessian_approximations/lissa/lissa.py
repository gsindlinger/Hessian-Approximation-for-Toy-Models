from __future__ import annotations

import jax
import jax.numpy as jnp
from typing_extensions import override

from config.hessian_approximation_config import LiSSAConfig
from hessian_approximations.hessian.hessian import Hessian
from hessian_approximations.hessian_approximations import HessianApproximation


class LiSSA(HessianApproximation):
    r"""
    LiSSA approximates (G + λI)^{-1} @ v using iterative stochastic updates.

    The algorithm uses the recurrence:
        r_j = v + (I - \alpha(G + \lambda*I)) r_{j-1},
    where G is a stochastic (mini-batch) estimate of the Hessian, and \alpha > 0 is a scaling factor.
    The IHVP is approximated as \alpha * r_J after J iterations.
    """

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.full_config.hessian_approximation, LiSSAConfig):
            raise ValueError(
                f"LiSSA requires LiSSAConfig, got {type(self.full_config.hessian_approximation)}"
            )
        self.config: LiSSAConfig = self.full_config.hessian_approximation

    @override
    def compute_hessian(self):
        raise NotImplementedError(
            "LiSSA approximates inverse Hessian-vector products only. "
            "Use compute_ihvp()."
        )

    @override
    def compute_hvp(self):
        raise NotImplementedError(
            "LiSSA is designed for inverse Hessian-vector products. "
            "Use compute_ihvp() instead."
        )

    @override
    def compute_ihvp(
        self,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute inverse Hessian-vector product (IHVP) using LiSSA.

        Implements:
            r_j = v + (I - \alpha(G + λI)) r_{j-1},
        with base case r_0 = v,
        and returns \alpha * r_J as the final approximation of (G + λI)^{-1} v.
        """

        training_data, training_targets = self.model_context.dataset.get_train_data()

        n_samples = training_data.shape[0]
        actual_batch_size = min(n_samples, self.config.batch_size)

        rng = jax.random.PRNGKey(self.config.seed)
        estimates = []

        # reuse true hessian approach for HVP on mini-batch
        hvp_fn = Hessian(full_config=self.full_config).compute_hvp

        for _ in range(self.config.num_samples):
            rng, subkey = jax.random.split(rng)
            r = vector  # base case: r_0 = v

            for j in range(self.config.recursion_depth):
                rng, subkey = jax.random.split(rng)

                hvp = hvp_fn(
                    vectors=r,
                )
                hvp = hvp + self.config.damping * r  # (G + λI)r

                # r_j = v + (I - \alpha(G + λI)) r_{j-1}
                #     = v + r_{j-1} - \alpha(G + λI) r_{j-1}
                r_new = vector + r - self.config.alpha * hvp

                if (
                    self.config.check_convergence_every > 0
                    and j > 0
                    and j % self.config.check_convergence_every == 0
                ):
                    update_norm = jnp.linalg.norm(r_new - r)
                    if update_norm < self.config.convergence_tol:
                        r = r_new
                        break

                r = r_new

            estimates.append(self.config.alpha * r)  # IHVP ≈ α * r_J

        inverse_hvp = jnp.stack(estimates).mean(axis=0)
        return inverse_hvp
