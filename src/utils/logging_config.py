"""Logging configuration module for consistent output formatting."""

import logging

from src.constants import DEFAULT_LOG_DATE_FORMAT, DEFAULT_LOG_FORMAT


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with standard formatting and severity level.

    Args:
        level: Logging level integer (defaults to logging.INFO).

    Returns:
        None.

    Raises:
        None.
    """
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_LOG_DATE_FORMAT,
        force=True,
    )
