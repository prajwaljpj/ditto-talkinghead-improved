"""
Logging configuration with Rich console output and JSON file logging.

This module provides:
- Rich console handler for beautiful, colored terminal output
- JSON file handler for structured production logs
- Correlation ID tracking across threads
- Per-component logger creation
- Context managers for request tracking
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from rich.logging import RichHandler
from rich.console import Console
from pythonjsonlogger import jsonlogger


# Thread-local storage for correlation IDs
_thread_local = threading.local()

# Global configuration
LOG_DIR = Path("logs")
LOG_LEVEL = logging.INFO
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG


class CorrelationIdFilter(logging.Filter):
    """
    Add correlation ID to log records for tracking requests across threads.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional fields.
    """
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        # Add log level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

        # Add component name
        log_record['component'] = record.name

        # Add correlation ID
        log_record['correlation_id'] = getattr(record, 'correlation_id', 'N/A')

        # Add thread info for debugging
        log_record['thread_id'] = record.thread
        log_record['thread_name'] = record.threadName


def get_correlation_id() -> str:
    """
    Get the current correlation ID from thread-local storage.

    Returns:
        Current correlation ID or 'N/A' if not set
    """
    return getattr(_thread_local, 'correlation_id', 'N/A')


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current thread.

    Args:
        correlation_id: Unique identifier for tracking requests
    """
    _thread_local.correlation_id = correlation_id


def clear_correlation_id() -> None:
    """
    Clear the correlation ID for the current thread.
    """
    if hasattr(_thread_local, 'correlation_id'):
        delattr(_thread_local, 'correlation_id')


class CorrelationIdContext:
    """
    Context manager for setting correlation ID within a scope.

    Usage:
        with CorrelationIdContext('session-123'):
            logger.info("This log will have correlation_id='session-123'")
    """
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.previous_id = None

    def __enter__(self):
        self.previous_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_id and self.previous_id != 'N/A':
            set_correlation_id(self.previous_id)
        else:
            clear_correlation_id()


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: int = CONSOLE_LOG_LEVEL,
    file_level: int = FILE_LOG_LEVEL,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Setup logging with Rich console and JSON file handlers.

    Args:
        log_dir: Directory for log files (default: ./logs)
        console_level: Log level for console output
        file_level: Log level for file output
        enable_console: Enable Rich console handler
        enable_file: Enable JSON file handler
    """
    # Create log directory
    if log_dir is None:
        log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()

    # Setup Rich console handler
    if enable_console:
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            show_level=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True
        )
        rich_handler.setLevel(console_level)
        rich_handler.addFilter(correlation_filter)

        # Format: [timestamp] LEVEL | correlation_id | component | message
        console_format = "%(message)s"
        rich_handler.setFormatter(logging.Formatter(console_format))
        root_logger.addHandler(rich_handler)

    # Setup JSON file handler
    if enable_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ditto_{timestamp}.json"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.addFilter(correlation_filter)

        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

        # Also create a symlink to latest log
        latest_log = log_dir / "latest.json"
        if latest_log.exists() or latest_log.is_symlink():
            latest_log.unlink()
        latest_log.symlink_to(log_file.name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Component name (e.g., 'core.audio2motion', 'api.inference')

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **kwargs) -> None:
    """
    Log performance metrics in a structured way.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **kwargs: Additional metadata
    """
    metadata = {
        'operation': operation,
        'duration_ms': round(duration_ms, 2),
        **kwargs
    }
    logger.info(f"Performance: {operation}", extra={'metadata': metadata})


def log_stage_progress(logger: logging.Logger, stage: str, current: int, total: int, **kwargs) -> None:
    """
    Log pipeline stage progress.

    Args:
        logger: Logger instance
        stage: Stage name
        current: Current item number
        total: Total items
        **kwargs: Additional metadata
    """
    percentage = (current / total * 100) if total > 0 else 0
    metadata = {
        'stage': stage,
        'current': current,
        'total': total,
        'percentage': round(percentage, 1),
        **kwargs
    }
    logger.info(f"{stage}: {current}/{total} ({percentage:.1f}%)", extra={'metadata': metadata})


def log_queue_status(logger: logging.Logger, queue_name: str, size: int, max_size: int) -> None:
    """
    Log queue status for monitoring pipeline health.

    Args:
        logger: Logger instance
        queue_name: Name of the queue
        size: Current queue size
        max_size: Maximum queue size
    """
    percentage = (size / max_size * 100) if max_size > 0 else 0
    metadata = {
        'queue': queue_name,
        'size': size,
        'max_size': max_size,
        'percentage': round(percentage, 1)
    }

    # Warn if queue is getting full
    if percentage > 80:
        logger.warning(f"Queue {queue_name} is {percentage:.1f}% full", extra={'metadata': metadata})
    else:
        logger.debug(f"Queue {queue_name}: {size}/{max_size}", extra={'metadata': metadata})


# Utility function to replace print statements
def log_separator(logger: logging.Logger, message: str = "", char: str = "=", width: int = 20) -> None:
    """
    Log a separator line (replacement for print("="*20, message, "="*20)).

    Args:
        logger: Logger instance
        message: Message to display
        char: Character for separator
        width: Width of separator on each side
    """
    if message:
        logger.info(f"{char * width} {message} {char * width}")
    else:
        logger.info(char * (width * 2 + 2))


# Initialize logging on module import (can be reconfigured later)
def initialize_logging():
    """
    Initialize logging with default configuration.
    Call this at application startup.
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Logging system initialized", extra={
        'metadata': {
            'console_enabled': True,
            'file_enabled': True,
            'log_dir': str(LOG_DIR.absolute())
        }
    })


# Export public API
__all__ = [
    'setup_logging',
    'get_logger',
    'get_correlation_id',
    'set_correlation_id',
    'clear_correlation_id',
    'CorrelationIdContext',
    'log_performance',
    'log_stage_progress',
    'log_queue_status',
    'log_separator',
    'initialize_logging',
]
