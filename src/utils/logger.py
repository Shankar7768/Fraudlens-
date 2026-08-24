"""Standardized logging configuration for Fraud Lens."""
import logging
import sys
from typing import Optional


def get_logger(name: str = "fraudlens", level: Optional[int] = None) -> logging.Logger:
    """Configures and returns a consistent logger.

    Args:
        name: Name of the logger, typically __name__.
        level: Logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if level is None:
        level = logging.INFO

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
