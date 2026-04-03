# Import json for writing configuration files.
import json

# Import joblib to save trained models.
import joblib

# Import artifact paths from config.
from src.config import ARTIFACTS_DIR, FEATURES_PATH, METRICS_PATH, MODEL_PATH

# Import the logger helper.
from src.logger import get_logger


# Create a logger for this module.
logger = get_logger(__name__)


# Define a function to save the trained model and metadata.
def save_artifacts(model, feature_columns, metrics):
    # Start a try block for safe file writing.
    try:
        # Create the artifacts directory if it does not exist.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save the trained model using joblib.
        joblib.dump(model, MODEL_PATH)

        # Open the feature columns file for writing.
        with open(FEATURES_PATH, "w", encoding="utf-8") as file:
            # Write the feature columns list as JSON.
            json.dump(list(feature_columns), file, indent=4)

        # Open the metrics file for writing.
        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            # Write the metrics dictionary as JSON.
            json.dump(metrics, file, indent=4)

        # Log successful artifact saving.
        logger.info("Artifacts saved successfully in %s", ARTIFACTS_DIR)
    # Catch any exception during artifact saving.
    except Exception as error:
        # Log the save failure.
        logger.exception("Saving artifacts failed: %s", error)
        # Re raise the exception.
        raise