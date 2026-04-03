# Import the logging module for application logs.
import logging

# Import Path to work with directories safely.
from pathlib import Path

# Import the train function from the training module.
from src.train import train_and_save_pipeline


# Define the base directory of the project.
BASE_DIR = Path(__file__).resolve().parent

# Define the logs directory path.
LOGS_DIR = BASE_DIR / "logs"

# Create the logs directory if it does not exist.
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging settings for the project.
logging.basicConfig(
    filename=LOGS_DIR / "project.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Create a logger object for this file.
logger = logging.getLogger(__name__)


# Create the main entry point function.
def main():
    # Start a try block to handle training errors.
    try:
        # Log the start of model training.
        logger.info("Starting Real Estate model training pipeline.")

        # Run the full training pipeline.
        train_and_save_pipeline()

        # Log successful completion.
        logger.info("Training pipeline completed successfully.")
    # Catch any exception during execution.
    except Exception as error:
        # Log the full exception details.
        logger.exception("Training pipeline failed: %s", error)
        # Re raise the exception so failure is visible to the user.
        raise


# Run the main function only when this file is executed directly.
if __name__ == "__main__":
    # Call the main function.
    main()