# Import the logging module.
import logging

# Import Path for directory creation.
from pathlib import Path

# Import the log directory path from config.
from src.config import LOGS_DIR


# Define a function to configure and return a logger.
def get_logger(name: str) -> logging.Logger:
    # Create the logs directory if it does not exist.
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Create or get a logger with the given name.
    logger = logging.getLogger(name)

    # Return early if handlers already exist to avoid duplicates.
    if logger.handlers:
        # Return the existing configured logger.
        return logger

    # Set the logging level to INFO.
    logger.setLevel(logging.INFO)

    # Create a formatter for log messages.
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Create a file handler for persistent logs.
    file_handler = logging.FileHandler(LOGS_DIR / "project.log", encoding="utf-8")

    # Attach the formatter to the file handler.
    file_handler.setFormatter(formatter)

    # Create a console handler for terminal output.
    stream_handler = logging.StreamHandler()

    # Attach the formatter to the console handler.
    stream_handler.setFormatter(formatter)

    # Add the file handler to the logger.
    logger.addHandler(file_handler)

    # Add the console handler to the logger.
    logger.addHandler(stream_handler)

    # Return the configured logger.
    return logger