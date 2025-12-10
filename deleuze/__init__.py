# mypackage/__init__.py
import logging
import sys


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
