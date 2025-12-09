# Class for Logging 
# DO NOT TOUCH :(
import logging
import os
from datetime import datetime
from pathlib import Path
from functools import wraps


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Base format string
    format_string = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_string + reset,
        logging.INFO: blue + format_string + reset,
        logging.WARNING: yellow + format_string + reset,
        logging.ERROR: red + format_string + reset,
        logging.CRITICAL: bold_red + format_string + reset,
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_string)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    """Formatter for file output (no colors)"""
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
        )


def setup_logger(
    name: str = __name__,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__).
        log_dir: Directory to store log files.
        console_level: Logging level for console.
        file_level: Logging level for log files.
        enable_console: If False, disables console output.
    
    Returns:
        Configured logger instance.
    """
    
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console Handler with colors (optional)
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    # File Handler - Daily log file
    log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)
    
    # Error File Handler - Separate file for errors
    error_filename = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
    error_handler = logging.FileHandler(error_filename, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(FileFormatter())
    logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str | None = None, **kwargs) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (defaults to __name__).
        **kwargs: Passed through to setup_logger (e.g., enable_console=False).
    
    Returns:
        Logger instance.
    """
    if name is None:
        name = __name__
    return setup_logger(name=name, **kwargs)


def log_function_call(logger: logging.Logger):
    """Decorator to log function entry and exit."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Calling function {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Function {func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} raised exception: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


class LogContext:
    """Context manager for logging code blocks."""
    def __init__(self, logger: logging.Logger, message: str):
        self.logger = logger
        self.message = message
        self.start_time: datetime | None = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed: {self.message} (took {duration:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.message} (took {duration:.2f}s)",
                exc_info=True,
            )
        # Do not suppress exceptions
        return False
