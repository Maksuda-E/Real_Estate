# Import pandas for DataFrame operations.
import pandas as pd

# Import the logger helper.
from src.logger import get_logger


# Create a logger for this module.
logger = get_logger(__name__)


# Define a function to prepare features and target.
def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Start a try block to handle preprocessing errors.
    try:
        # Log preprocessing start.
        logger.info("Preparing features and target.")

        # Create a copy of the dataset to avoid mutating the original DataFrame.
        data = df.copy()

        # Check whether the target column exists.
        if "price" not in data.columns:
            # Raise a clear error if the target is missing.
            raise ValueError("Target column 'price' is missing from the dataset.")

        # Drop the target column to create feature matrix X.
        x = data.drop("price", axis=1)

        # Select the target column to create target vector y.
        y = data["price"]

        # Log the shapes of features and target.
        logger.info("Prepared X shape %s and y shape %s", x.shape, y.shape)

        # Return features and target.
        return x, y
    # Catch any exception during preprocessing.
    except Exception as error:
        # Log the exception details.
        logger.exception("Feature preparation failed: %s", error)
        # Re raise the exception for calling code.
        raise