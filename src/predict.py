# This line imports pickle for loading saved objects
import pickle

# This line imports pandas for creating input DataFrame
import pandas as pd

# This line imports saved model paths
from src.config import MODEL_FILE_PATH, FEATURE_COLUMNS_FILE_PATH

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function loads the saved model and feature columns
def load_model_and_features():
    # This line starts the try block
    try:
        # This line opens the model file in read binary mode
        with open(MODEL_FILE_PATH, "rb") as model_file:
            # This line loads the trained model
            model = pickle.load(model_file)

        # This line opens the feature columns file in read binary mode
        with open(FEATURE_COLUMNS_FILE_PATH, "rb") as feature_file:
            # This line loads the feature columns
            feature_columns = pickle.load(feature_file)

        # This line logs successful loading
        logger.info("Model and feature columns loaded successfully")

        # This line returns the loaded model and feature columns
        return model, feature_columns

    # This block handles loading errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while loading model or feature columns")

        # This line raises a custom exception
        raise ProjectException(f"Failed to load model artifacts: {exc}")

# This function converts user input into the format needed by the model
def prepare_input_data(user_input: dict, feature_columns: list):
    # This line starts the try block
    try:
        # This line creates a DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # This line matches input columns to training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # This line logs successful input preparation
        logger.info("Input data prepared successfully for prediction")

        # This line returns the prepared DataFrame
        return input_df

    # This block handles input preparation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while preparing input data")

        # This line raises a custom exception
        raise ProjectException(f"Failed to prepare input data: {exc}")

# This function predicts the real estate price
def predict_real_estate_price(user_input: dict):
    # This line starts the try block
    try:
        # This line loads the model and feature list
        model, feature_columns = load_model_and_features()

        # This line prepares user input
        input_df = prepare_input_data(user_input, feature_columns)

        # This line predicts the house price
        prediction = model.predict(input_df)[0]

        # This line logs successful prediction
        logger.info("Prediction completed successfully")

        # This line returns the predicted price as float
        return float(prediction)

    # This block handles prediction errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during prediction")

        # This line raises a custom exception
        raise ProjectException(f"Failed to predict real estate price: {exc}")