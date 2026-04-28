#!/usr/bin/env python3
"""
Logging utilities for the Blender YOLO Dataset Generator.
Provides consistent logging across all modules with color-coded output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        verbose: If True, set level to DEBUG
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else level)
    
    console_format = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (no colors)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context message
    """
    import traceback
    
    if context:
        logger.error(f"{context}: {str(exc)}")
    else:
        logger.error(f"Exception occurred: {str(exc)}")
    
    # Log full traceback at debug level
    tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.debug(f"Full traceback:\n{tb_str}")


def log_progress(logger: logging.Logger, current: int, total: int, item_name: str = "items"):
    """
    Log progress with percentage.
    
    Args:
        logger: Logger instance
        current: Current progress
        total: Total items
        item_name: Name of items being processed
    """
    if total == 0:
        return
    
    percentage = (current / total) * 100
    logger.info(f"Progress: {current}/{total} {item_name} ({percentage:.1f}%)")


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


# Global logger instance
_default_logger: Optional[logging.Logger] = None


def get_logger(name: str = "blender_yolo") -> logging.Logger:
    """
    Get or create the default logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logger(name)
    
    return _default_logger


if __name__ == '__main__':
    # Test logging
    logger = setup_logger('test', verbose=True)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test progress logging
    for i in range(1, 11):
        log_progress(logger, i, 10, "scenes")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_exception(logger, e, "Testing exception logging")
