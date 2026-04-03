# Import pandas for data loading and processing.
import pandas as pd

# Import the dataset path from config.
from src.config import DATA_PATH

# Import the project logger helper.
from src.logger import get_logger


# Create a logger for this module.
logger = get_logger(__name__)


# Define a function to load the raw dataset.
def load_data() -> pd.DataFrame:
    # Start a try block to handle file and parsing errors.
    try:
        # Log the data loading start.
        logger.info("Loading dataset from %s", DATA_PATH)

        # Read the CSV file into a DataFrame.
        df = pd.read_csv(DATA_PATH)

        # Log the loaded dataset shape.
        logger.info("Dataset loaded successfully with shape %s", df.shape)

        # Return the loaded DataFrame.
        return df
    # Catch any exception during loading.
    except Exception as error:
        # Log the exception details.
        logger.exception("Failed to load dataset: %s", error)
        # Re raise the error for upstream handling.
        raise