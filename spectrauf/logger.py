import logging
import os
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Sets up the logger to save to the execution directory."""
    logger = logging.getLogger("mcio")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Determine the file path
        log_path = os.path.join(Path.cwd(), "mcio.log")

        # File Handler
        try:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except PermissionError:
            # Fallback if the script is run in a directory without write permissions
            print(f"Warning: No write access to {log_path}. Logging to console only.")

        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def get_logger():
    """Returns the existing logger instance without re-configuring it."""
    return logging.getLogger("mcio")


# set up logger level look up
# store logging levels to allow user to set verbosity
class LoggerLevels:
    def __init__(self):
        self.debug = logging.DEBUG
        self.info = logging.INFO
        self.warning = logging.WARNING
        self.error = logging.ERROR
        self.critical = logging.CRITICAL

        self.LEVEL_LOOKUP = {
            10: self.debug,
            20: self.info,
            30: self.warning,
            40: self.error,
            50: self.critical,
        }


LOGGER_LEVELS = LoggerLevels()
