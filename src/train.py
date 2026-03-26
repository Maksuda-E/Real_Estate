# This line imports os for folder creation
import os

# This line imports json for saving metrics
import json

# This line imports pickle for saving the model
import pickle

# This line imports LinearRegression for regression modeling
from sklearn.linear_model import LinearRegression

# This line imports RandomForestRegressor for regression modeling
from sklearn.ensemble import RandomForestRegressor

# This line imports mean_absolute_error for model evaluation
from sklearn.metrics import mean_absolute_error

# This line imports config values
from src.config import (
    ARTIFACTS_DIR,
    MODEL_FILE_PATH,
    FEATURE_COLUMNS_FILE_PATH,
    METRICS_FILE_PATH,
    RANDOM_STATE
)

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function trains the linear regression model
def train_linear_regression(x_train, y_train):
    # This line starts the try block
    try:
        # This line logs the start of linear regression training
        logger.info("Linear Regression training started")

        # This line creates the linear regression model
        model = LinearRegression()

        # This line trains the model on the training data
        model.fit(x_train, y_train)

        # This line logs that linear regression training completed
        logger.info("Linear Regression training completed successfully")

        # This line returns the trained model
        return model

    # This block handles training errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during Linear Regression training")

        # This line raises a custom exception
        raise ProjectException(f"Failed to train Linear Regression model: {exc}")

# This function trains the random forest regressor model
def train_random_forest(x_train, y_train):
    # This line starts the try block
    try:
        # This line logs the start of random forest training
        logger.info("Random Forest training started")

        # This line creates the random forest regressor model
        model = RandomForestRegressor(random_state=RANDOM_STATE)

        # This line trains the model on the training data
        model.fit(x_train, y_train)

        # This line logs that random forest training completed
        logger.info("Random Forest training completed successfully")

        # This line returns the trained model
        return model

    # This block handles training errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during Random Forest training")

        # This line raises a custom exception
        raise ProjectException(f"Failed to train Random Forest model: {exc}")

# This function evaluates a trained model using mean absolute error
def evaluate_model(model, x_test, y_test):
    # This line starts the try block
    try:
        # This line logs the start of model evaluation
        logger.info("Model evaluation started")

        # This line predicts the target values for test data
        predictions = model.predict(x_test)

        # This line calculates the mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        # This line logs that evaluation completed successfully
        logger.info("Model evaluation completed successfully")

        # This line returns the mean absolute error
        return mae

    # This block handles evaluation errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model evaluation")

        # This line raises a custom exception
        raise ProjectException(f"Failed to evaluate model: {exc}")

# This function trains all models and selects the best one
def train_model(x_train, y_train, x_test, y_test):
    # This line starts the try block
    try:
        # This line logs the start of full model training
        logger.info("Training and comparison of models started")

        # This line trains the linear regression model
        linear_model = train_linear_regression(x_train, y_train)

        # This line evaluates the linear regression model
        linear_mae = evaluate_model(linear_model, x_test, y_test)

        # This line trains the random forest model
        random_forest_model = train_random_forest(x_train, y_train)

        # This line evaluates the random forest model
        random_forest_mae = evaluate_model(random_forest_model, x_test, y_test)

        # This line creates a dictionary of model metrics
        metrics = {
            "LinearRegression_MAE": linear_mae,
            "RandomForestRegressor_MAE": random_forest_mae
        }

        # This line checks whether linear regression performed better
        if linear_mae <= random_forest_mae:
            # This line stores the best model name
            best_model_name = "LinearRegression"

            # This line stores the best trained model
            best_model = linear_model

            # This line stores the best MAE score
            best_mae = linear_mae
        else:
            # This line stores the best model name
            best_model_name = "RandomForestRegressor"

            # This line stores the best trained model
            best_model = random_forest_model

            # This line stores the best MAE score
            best_mae = random_forest_mae

        # This line adds the best model name to the metrics dictionary
        metrics["Best_Model"] = best_model_name

        # This line adds the best model MAE to the metrics dictionary
        metrics["Best_Model_MAE"] = best_mae

        # This line logs the best model selection
        logger.info(f"Best model selected: {best_model_name}")

        # This line returns the best model and metrics
        return best_model, metrics

    # This block handles training and selection errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during model training and selection")

        # This line raises a custom exception
        raise ProjectException(f"Failed to train and select best model: {exc}")

# This function saves the model and feature columns
def save_artifacts(model, feature_columns, metrics):
    # This line starts the try block
    try:
        # This line creates the artifacts folder if it does not exist
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # This line opens the model file in write binary mode
        with open(MODEL_FILE_PATH, "wb") as model_file:
            # This line saves the trained model
            pickle.dump(model, model_file)

        # This line opens the feature columns file in write binary mode
        with open(FEATURE_COLUMNS_FILE_PATH, "wb") as feature_file:
            # This line saves the feature columns
            pickle.dump(feature_columns, feature_file)

        # This line opens the metrics file in write mode
        with open(METRICS_FILE_PATH, "w", encoding="utf-8") as metrics_file:
            # This line saves the model metrics as json
            json.dump(metrics, metrics_file, indent=4)

        # This line logs that artifacts were saved successfully
        logger.info("Artifacts saved successfully")

    # This block handles save errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while saving artifacts")

        # This line raises a custom exception
        raise ProjectException(f"Failed to save artifacts: {exc}")