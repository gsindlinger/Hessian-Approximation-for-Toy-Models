# mypackage/__init__.py
import logging
import sys

import jax

# Permit float64 globally so `.astype(jnp.float64)` promotions inside
# numerically delicate ops (eigh on near-rank-deficient covariances/Hessians)
# are actually honored. Does not change the default dtype — arrays stay fp32
# unless explicitly promoted.
jax.config.update("jax_enable_x64", True)


def _setup_default_logging():
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(levelname)s:%(asctime)s:%(filename)s:"
                "%(funcName)s:%(lineno)d: %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S,%f",
            stream=sys.stdout,
        )


_setup_default_logging()
