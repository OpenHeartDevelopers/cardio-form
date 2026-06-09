import logging
import os

def configure_logging(log_name: str, log_level=logging.INFO, log_format='[%(funcName)s] %(message)s'):
    """
    Configure a logger with a stream handler (console output).

    Args:
        log_name (str): Name of the logger.
        log_level (int): Logging level (e.g., logging.INFO).
        log_format (str): Format for log messages.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(log_name)

    # Check if logger already has handlers to avoid duplicates
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    return logger

def get_unique_parts(file_paths):
    # Strip directory paths and file extensions
    base_names = [os.path.splitext(os.path.dirname(path))[0] for path in file_paths]
    
    # Find the common prefix
    common_prefix = os.path.commonprefix(base_names)
    return common_prefix

def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
