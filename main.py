# This line imports the dataset file path
from src.config import DATA_FILE_PATH

# This line imports the data loading function
from src.data_loader import load_data

# This line imports preprocessing functions
from src.preprocess import clean_data, split_features_target, split_train_test

# This line imports training functions
from src.train import train_model, save_artifacts

# This line imports the logger
from src.logger import get_logger

# This line creates a logger for this file
logger = get_logger(__name__)

# This function controls the full training pipeline
def main():
    # This line starts the try block for the full pipeline
    try:
        # This line logs that the main pipeline has started
        logger.info("Main pipeline started")

        # This line loads the raw dataset
        df = load_data(DATA_FILE_PATH)

        # This line cleans and preprocesses the dataset
        df_clean = clean_data(df)

        # This line splits the cleaned data into features and target
        x, y = split_features_target(df_clean)

        # This line creates training and testing datasets
        x_train, x_test, y_train, y_test = split_train_test(x, y)

        # This line trains the models and selects the best one
        model, metrics = train_model(x_train, y_train, x_test, y_test)

        # This line saves the trained model, feature columns, and metrics
        save_artifacts(model, list(x.columns), metrics)

        # This line prints a success message
        print("Training completed successfully")

        # This line prints each metric heading
        print("Model evaluation results")

        # This line starts a loop through each metric
        for key, value in metrics.items():
            # This line prints the metric name and value
            print(f"{key}: {value}")

        # This line logs that the main pipeline finished
        logger.info("Main pipeline completed successfully")

    # This block handles errors in the full pipeline
    except Exception as exc:
        # This line logs the pipeline failure
        logger.error(f"Main pipeline failed: {exc}")

        # This line raises the error again
        raise

# This line checks whether this file is being run directly
if __name__ == "__main__":
    # This line calls the main function
    main()