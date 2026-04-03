# Import mean absolute error metric.
from sklearn.metrics import mean_absolute_error

# Import train test split utility.
from sklearn.model_selection import train_test_split

# Import linear regression model.
from sklearn.linear_model import LinearRegression

# Import random forest regressor.
from sklearn.ensemble import RandomForestRegressor

# Import the logger helper.
from src.logger import get_logger


# Create a logger for this module.
logger = get_logger(__name__)


# Define a function to split the data like the notebook flow.
def split_data(x, y):
    # Start a try block for safe split handling.
    try:
        # Log split start.
        logger.info("Splitting data into train and test sets.")

        # Use stratified split based on property_type_Condo when available.
        if "property_type_Condo" in x.columns:
            # Perform the train test split with stratification.
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42,
                stratify=x["property_type_Condo"],
            )
        else:
            # Perform a normal train test split if stratification column is missing.
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42,
            )

        # Log the resulting shapes.
        logger.info(
            "Split completed. X_train=%s X_test=%s y_train=%s y_test=%s",
            x_train.shape,
            x_test.shape,
            y_train.shape,
            y_test.shape,
        )

        # Return the split datasets.
        return x_train, x_test, y_train, y_test
    # Catch any exception during splitting.
    except Exception as error:
        # Log split failure.
        logger.exception("Data splitting failed: %s", error)
        # Re raise the error.
        raise


# Define a function to train both notebook models and compare them.
def train_models(x_train, x_test, y_train, y_test):
    # Start a try block for safe model training.
    try:
        # Log training start.
        logger.info("Training Linear Regression model.")

        # Create the linear regression model instance.
        linear_model = LinearRegression()

        # Fit linear regression on the training data.
        linear_model.fit(x_train, y_train)

        # Predict on training data using linear regression.
        linear_train_pred = linear_model.predict(x_train)

        # Predict on test data using linear regression.
        linear_test_pred = linear_model.predict(x_test)

        # Calculate linear regression train MAE.
        linear_train_mae = mean_absolute_error(y_train, linear_train_pred)

        # Calculate linear regression test MAE.
        linear_test_mae = mean_absolute_error(y_test, linear_test_pred)

        # Log linear regression results.
        logger.info(
            "Linear Regression completed. Train MAE=%.4f Test MAE=%.4f",
            linear_train_mae,
            linear_test_mae,
        )

        # Log random forest training start.
        logger.info("Training Random Forest Regressor model.")

        # Create the random forest model aligned with notebook flow.
        random_forest_model = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42,
            n_jobs=-1,
        )

        # Fit the random forest model on the training data.
        random_forest_model.fit(x_train, y_train)

        # Predict on training data using random forest.
        rf_train_pred = random_forest_model.predict(x_train)

        # Predict on test data using random forest.
        rf_test_pred = random_forest_model.predict(x_test)

        # Calculate random forest train MAE.
        rf_train_mae = mean_absolute_error(y_train, rf_train_pred)

        # Calculate random forest test MAE.
        rf_test_mae = mean_absolute_error(y_test, rf_test_pred)

        # Log random forest results.
        logger.info(
            "Random Forest completed. Train MAE=%.4f Test MAE=%.4f",
            rf_train_mae,
            rf_test_mae,
        )

        # Store metrics in a dictionary.
        metrics = {
            "linear_regression_train_mae": round(float(linear_train_mae), 4),
            "linear_regression_test_mae": round(float(linear_test_mae), 4),
            "random_forest_train_mae": round(float(rf_train_mae), 4),
            "random_forest_test_mae": round(float(rf_test_mae), 4),
        }

        # Choose the best model based on the lowest test MAE.
        best_model = random_forest_model if rf_test_mae <= linear_test_mae else linear_model

        # Determine the best model name.
        best_model_name = "RandomForestRegressor" if rf_test_mae <= linear_test_mae else "LinearRegression"

        # Log the selected best model.
        logger.info("Best model selected: %s", best_model_name)

        # Return the best model and metrics.
        return best_model, best_model_name, metrics
    # Catch any exception during model training.
    except Exception as error:
        # Log training failure.
        logger.exception("Model training failed: %s", error)
        # Re raise the error.
        raise