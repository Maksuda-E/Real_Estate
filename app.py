# Import the json module for reading feature configuration files.
import json

# Import the logging module for application logging.
import logging

# Import Path to work with file paths safely.
from pathlib import Path

# Import joblib to load the trained model.
import joblib

# Import pandas for tabular data handling.
import pandas as pd

# Import streamlit to build the web app.
import streamlit as st


# Create a logger object for this file.
logger = logging.getLogger(__name__)

# Define the base directory of the project.
BASE_DIR = Path(__file__).resolve().parent

# Define the artifacts directory where model files are stored.
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Define the expected path for the trained model file.
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.joblib"

# Define the expected path for the feature list file.
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Define the expected path for the metrics file.
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# Cache the model loading so Streamlit does not reload on every interaction.
@st.cache_resource
def load_model():
    # Start a try block to catch file loading errors.
    try:
        # Check whether the model file exists before loading.
        if not MODEL_PATH.exists():
            # Raise a file not found error if the model is missing.
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

        # Load the trained model from disk.
        model = joblib.load(MODEL_PATH)

        # Return the loaded model.
        return model

    # Catch any exception during model loading.
    except Exception as error:
        # Log the model loading error.
        logger.exception("Failed to load model: %s", error)

        # Raise a Streamlit visible error.
        raise RuntimeError("Model file could not be loaded. Make sure the trained artifacts exist.")


# Cache the feature list loading for performance.
@st.cache_data
def load_feature_columns():
    # Start a try block to catch JSON reading errors.
    try:
        # Check whether the feature file exists before opening.
        if not FEATURES_PATH.exists():
            # Raise a file not found error if the feature file is missing.
            raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

        # Open the feature JSON file in read mode.
        with open(FEATURES_PATH, "r", encoding="utf-8") as file:
            # Load and return the feature list from JSON.
            return json.load(file)

    # Catch any exception during feature file loading.
    except Exception as error:
        # Log the feature loading error.
        logger.exception("Failed to load feature columns: %s", error)

        # Raise a user friendly runtime error.
        raise RuntimeError("Feature configuration could not be loaded. Make sure the trained artifacts exist.")


# Cache metrics loading for performance.
@st.cache_data
def load_metrics():
    # Start a try block to catch JSON reading errors.
    try:
        # Check if the metrics file exists.
        if METRICS_PATH.exists():
            # Open the metrics file.
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                # Return the metrics dictionary.
                return json.load(file)

        # Return an empty dictionary if metrics file is missing.
        return {}

    # Catch any exception during metrics loading.
    except Exception as error:
        # Log the metrics loading error.
        logger.exception("Failed to load metrics: %s", error)

        # Return an empty dictionary to keep the app running.
        return {}


# Build the input row in the exact feature order used in training.
def build_input_dataframe(user_inputs: dict, feature_columns: list[str]) -> pd.DataFrame:
    # Create a dictionary with all required features initialized to zero.
    row = {feature: 0 for feature in feature_columns}

    # Fill year sold from user input.
    row["year_sold"] = user_inputs["year_sold"]

    # Fill property tax from user input.
    row["property_tax"] = user_inputs["property_tax"]

    # Fill insurance from user input.
    row["insurance"] = user_inputs["insurance"]

    # Fill beds from user input.
    row["beds"] = user_inputs["beds"]

    # Fill baths from user input.
    row["baths"] = user_inputs["baths"]

    # Fill square footage from user input.
    row["sqft"] = user_inputs["sqft"]

    # Fill build year from user input.
    row["year_built"] = user_inputs["year_built"]

    # Fill lot size from user input.
    row["lot_size"] = user_inputs["lot_size"]

    # Fill basement as numeric binary value.
    row["basement"] = 1 if user_inputs["basement"] else 0

    # Fill popular flag as numeric binary value.
    row["popular"] = 1 if user_inputs["popular"] else 0

    # Fill recession flag as numeric binary value.
    row["recession"] = 1 if user_inputs["recession"] else 0

    # Calculate property age based on year sold and year built.
    row["property_age"] = max(user_inputs["year_sold"] - user_inputs["year_built"], 0)

    # Encode condo type if the project uses this feature.
    if "property_type_Condo" in row:
        # Set condo feature to one when selected.
        row["property_type_Condo"] = 1 if user_inputs["property_type"] == "Condo" else 0

    # Create a DataFrame with one row in the correct feature order.
    input_df = pd.DataFrame([row], columns=feature_columns)

    # Return the final input DataFrame.
    return input_df


# Configure the Streamlit page before any other UI content.
st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")

# Add custom CSS styling to improve the full application design.
# Simple clean blue theme styling.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9fafb;
    }

    /* Titles */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }

    h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }

    p, label {
        color: #374151;
    }

    /* Form container */
    div[data-testid="stForm"] {
        background: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    /* Input fields */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #f3f4f6;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e5e7eb;
    }

    /* Info box */
    div[data-testid="stInfo"] {
        background: #e0f2fe;
        color: #1e3a8a;
        border-radius: 10px;
        border: 1px solid #bae6fd;
    }

    /* Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #E0B8B8;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
        border: none;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1d4ed8;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# Display a styled hero header section.
st.markdown(
    """
    <div style="
        background: #8A3727;
        padding: 32px;
        border-radius: 26px;
        margin-bottom: 24px;
        box-shadow: 0 18px 36px rgba(124, 45, 18, 0.18);
    ">
        <h1 style="color: white; margin: 0; font-size: 3rem; font-weight: 800;">
            Real Estate Price Predictor
        </h1>
        <p style="color: #ffedd5; margin-top: 12px; font-size: 1.1rem;">
            Predict property price using the trained machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the model into memory.
model = load_model()

# Load feature columns into memory.
feature_columns = load_feature_columns()

# Load model metrics if available.
metrics = load_metrics()

# Create two main layout columns.
left_col, right_col = st.columns([2.2, 1])

# Start the left column where the form will appear.
with left_col:
    # Add a section heading.
    st.subheader("Enter Property Details")

    # Create a form so all values submit together.
    with st.form("prediction_form"):
        # Create two inner columns for a clean layout.
        col1, col2 = st.columns(2)

        # Put the first set of inputs in the first column.
        with col1:
            # Add year sold input.
            year_sold = st.number_input("Year Sold", min_value=1990, max_value=2035, value=2015, step=1)

            # Add property tax input.
            property_tax = st.number_input("Property Tax", min_value=0, value=200, step=1)

            # Add insurance input.
            insurance = st.number_input("Insurance", min_value=0, value=75, step=1)

            # Add number of bedrooms input.
            beds = st.number_input("Beds", min_value=0, max_value=20, value=3, step=1)

            # Add number of bathrooms input.
            baths = st.number_input("Baths", min_value=0, max_value=20, value=2, step=1)

            # Add square footage input.
            sqft = st.number_input("Square Feet", min_value=100, value=1500, step=50)

        # Put the second set of inputs in the second column.
        with col2:
            # Add year built input.
            year_built = st.number_input("Year Built", min_value=1800, max_value=2035, value=2005, step=1)

            # Add lot size input.
            lot_size = st.number_input("Lot Size", min_value=0, value=3000, step=100)

            # Add basement selection.
            basement = st.selectbox("Basement", options=[False, True], format_func=lambda x: "Yes" if x else "No")

            # Add popular selection.
            popular = st.selectbox("Popular Area", options=[False, True], format_func=lambda x: "Yes" if x else "No")

            # Add recession selection.
            recession = st.selectbox("Recession Period", options=[False, True], format_func=lambda x: "Yes" if x else "No")

            # Add property type selection.
            property_type = st.selectbox("Property Type", options=["House", "Condo"])

        # Add the submit button at the end of the form.
        submitted = st.form_submit_button("Predict Property Price")

# Start the right column for model info and result display.
with right_col:
    # Add a section heading.
    st.subheader("Model Information")

    # Show selected model name.
    st.info("Model used: Random Forest Regressor")

    # Show metrics if available.
    if metrics:
        # Display test MAE metric if it exists.
        st.metric("Test MAE", f"{metrics.get('random_forest_test_mae', 'N/A')}")

        # Display train MAE metric if it exists.
        st.metric("Train MAE", f"{metrics.get('random_forest_train_mae', 'N/A')}")

# Process prediction only after form submission.
if submitted:
    # Start a try block to handle prediction errors safely.
    try:
        # Store form values in a dictionary.
        user_inputs = {
            "year_sold": int(year_sold),
            "property_tax": float(property_tax),
            "insurance": float(insurance),
            "beds": int(beds),
            "baths": int(baths),
            "sqft": float(sqft),
            "year_built": int(year_built),
            "lot_size": float(lot_size),
            "basement": bool(basement),
            "popular": bool(popular),
            "recession": bool(recession),
            "property_type": property_type,
        }

        # Convert user inputs to a DataFrame matching training features.
        input_df = build_input_dataframe(user_inputs, feature_columns)

        # Make the price prediction using the loaded model.
        prediction = model.predict(input_df)[0]

        # Create a separator for clean display.
        st.markdown("---")

        # Show result heading.
        st.subheader("Prediction Result")

        # Display the predicted price in a styled result card.
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
                padding: 24px;
                border-radius: 20px;
                text-align: center;
                color: white;
                font-size: 28px;
                font-weight: 800;
                box-shadow: 0 16px 30px rgba(34, 197, 94, 0.22);
                margin-top: 8px;
                margin-bottom: 18px;
            ">
                Estimated Price: ${prediction:,.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show the actual features sent to the model.
        st.subheader("Processed Input Data")

        # Display the processed one row DataFrame.
        st.dataframe(input_df, use_container_width=True)

    # Catch any error that happens during prediction.
    except Exception as error:
        # Log the prediction error for debugging.
        logger.exception("Prediction failed: %s", error)

        # Show a user friendly error message.
        st.error("Prediction failed. Please verify the inputs and model files.")