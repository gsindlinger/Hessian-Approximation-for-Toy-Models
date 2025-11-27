import logging


def get_stream_logger(level=logging.INFO) -> logging.Logger:
    """Get a logger that logs to console with specified level."""

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_file_logger(file_path: str, level=logging.INFO) -> logging.Logger:
    """Get a logger that logs to a specified file with specified level."""

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
