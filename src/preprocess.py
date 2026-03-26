# This line imports pandas for data processing
import pandas as pd

# This line imports train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# This line imports configuration values
from src.config import TEST_SIZE, RANDOM_STATE

# This line imports the logger
from src.logger import get_logger

# This line imports the custom exception
from src.custom_exception import ProjectException

# This line creates a logger for this file
logger = get_logger(__name__)

# This function cleans the dataset
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # This line starts the try block
    try:
        # This line logs the start of data cleaning
        logger.info("Starting data cleaning")

        # This line creates a copy of the dataset to avoid changing the original
        df = df.copy()

        # This line removes leading and trailing spaces from column names
        df.columns = df.columns.str.strip()

        # This line removes duplicate rows if any are present
        df = df.drop_duplicates()

        # This line checks if the target column exists in the dataset
        if "price" not in df.columns:
            # This line raises a custom exception if the target column is missing
            raise ProjectException("Target column 'price' not found in dataset")

        # This line gets all numeric columns from the dataset
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        # This line removes the target column from numeric columns if present
        if "price" in numeric_columns:
            numeric_columns.remove("price")

        # This line gets all categorical columns from the dataset
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # This line fills missing values in numeric columns with the median
        for column in numeric_columns:
            # This line checks whether the column has missing values
            if df[column].isnull().sum() > 0:
                # This line fills missing values with the median of the column
                df[column] = df[column].fillna(df[column].median())

        # This line fills missing values in categorical columns with the mode
        for column in categorical_columns:
            # This line checks whether the column has missing values
            if df[column].isnull().sum() > 0:
                # This line fills missing values with the most frequent value
                df[column] = df[column].fillna(df[column].mode()[0])

        # This line converts categorical columns into numeric dummy variables only if categorical columns exist
        if len(categorical_columns) > 0:
            # This line applies one hot encoding to categorical columns
            df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

        # This line logs that data cleaning is complete
        logger.info("Data cleaning completed successfully")

        # This line returns the cleaned dataset
        return df

    # This block handles errors during cleaning
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during data cleaning")

        # This line raises a custom error
        raise ProjectException(f"Failed to clean data: {exc}")

# This function separates features and target
def split_features_target(df: pd.DataFrame):
    # This line starts the try block
    try:
        # This line logs the start of feature and target split
        logger.info("Splitting features and target")

        # This line stores all columns except price in x
        x = df.drop("price", axis=1)

        # This line stores the price column in y
        y = df["price"]

        # This line logs successful feature and target split
        logger.info("Feature and target split completed successfully")

        # This line returns x and y
        return x, y

    # This block handles split errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred while splitting features and target")

        # This line raises a custom exception
        raise ProjectException(f"Failed to split features and target: {exc}")

# This function splits the data into training and testing sets
def split_train_test(x, y):
    # This line starts the try block
    try:
        # This line logs the start of train and test split
        logger.info("Starting train test split")

        # This line checks whether the stratify column from the notebook exists
        if "property_type_Condo" in x.columns:
            # This line splits x and y into training and test sets using stratification
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=x["property_type_Condo"]
            )
        else:
            # This line splits x and y into training and test sets without stratification
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

        # This line logs that splitting is complete
        logger.info("Train test split completed successfully")

        # This line returns the split data
        return x_train, x_test, y_train, y_test

    # This block handles splitting errors
    except Exception as exc:
        # This line logs the error
        logger.error("Error occurred during train test split")

        # This line raises a custom exception
        raise ProjectException(f"Failed to split train and test data: {exc}")