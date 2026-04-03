# Import the data loading function.
from src.data_loader import load_data

# Import the preprocessing function.
from src.preprocessing import prepare_features_and_target

# Import split and training functions.
from src.modeling import split_data, train_models

# Import artifact saving helper.
from src.utils import save_artifacts

# Import logger helper.
from src.logger import get_logger


# Create a logger for this module.
logger = get_logger(__name__)


# Define the full training pipeline.
def train_and_save_pipeline():
    # Start a try block to handle pipeline failures.
    try:
        # Log pipeline start.
        logger.info("Training pipeline started.")

        # Load the dataset from disk.
        df = load_data()

        # Prepare features and target from the raw data.
        x, y = prepare_features_and_target(df)

        # Split the dataset into train and test partitions.
        x_train, x_test, y_train, y_test = split_data(x, y)

        # Train the models and select the best one.
        best_model, best_model_name, metrics = train_models(x_train, x_test, y_train, y_test)

        # Add the selected model name to the metrics dictionary.
        metrics["best_model"] = best_model_name

        # Save the selected model and metadata files.
        save_artifacts(best_model, x.columns.tolist(), metrics)

        # Log pipeline completion.
        logger.info("Training pipeline finished successfully.")

        # Return the metrics for optional use.
        return metrics
    # Catch any exception during the pipeline.
    except Exception as error:
        # Log the full pipeline failure.
        logger.exception("Training pipeline error: %s", error)
        # Re raise the exception.
        raise