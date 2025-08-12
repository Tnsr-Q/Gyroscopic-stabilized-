"""Structured logging configuration."""

import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO", format_type: str = "structured") -> logging.Logger:
    """Setup structured logging for the quantum core engine."""
    
    # Convert string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Setup formatters
    if format_type == "structured":
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
    
    # Setup handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Setup quantum core logger
    logger = logging.getLogger('quantum_core')
    logger.setLevel(log_level)
    
    return logger