# This line imports streamlit for the web app
import streamlit as st

# This line imports the prediction function and model loader
from src.predict import predict_real_estate_price, load_model_and_features

# This line sets the page title and layout
st.set_page_config(page_title="Real Estate Price Prediction", layout="centered")

# This line shows the app title
st.title("Real Estate Price Prediction App")

# This line shows a short description
st.write("Enter property details to predict the estimated house price.")

# This line loads the saved model and feature columns
model, feature_columns = load_model_and_features()

# This line defines the binary columns in the dataset
binary_columns = ["basement", "popular", "recession", "property_type_Condo"]

# This line creates an empty dictionary for user input
user_input = {}

# This line checks whether year_sold exists in the feature columns
if "year_sold" in feature_columns:
    # This line creates a number input for year sold
    user_input["year_sold"] = st.number_input("Year Sold", min_value=1900, max_value=2100, value=2010, step=1)

# This line checks whether property_tax exists in the feature columns
if "property_tax" in feature_columns:
    # This line creates a number input for property tax
    user_input["property_tax"] = st.number_input("Property Tax", min_value=0.0, value=3000.0)

# This line checks whether insurance exists in the feature columns
if "insurance" in feature_columns:
    # This line creates a number input for insurance
    user_input["insurance"] = st.number_input("Insurance", min_value=0.0, value=1200.0)

# This line checks whether beds exists in the feature columns
if "beds" in feature_columns:
    # This line creates a number input for number of beds
    user_input["beds"] = st.number_input("Beds", min_value=0.0, value=3.0)

# This line checks whether baths exists in the feature columns
if "baths" in feature_columns:
    # This line creates a number input for number of baths
    user_input["baths"] = st.number_input("Baths", min_value=0.0, value=2.0)

# This line checks whether sqft exists in the feature columns
if "sqft" in feature_columns:
    # This line creates a number input for square feet
    user_input["sqft"] = st.number_input("Square Feet", min_value=0.0, value=1800.0)

# This line checks whether year_built exists in the feature columns
if "year_built" in feature_columns:
    # This line creates a number input for year built
    user_input["year_built"] = st.number_input("Year Built", min_value=1800, max_value=2100, value=2000, step=1)

# This line checks whether lot_size exists in the feature columns
if "lot_size" in feature_columns:
    # This line creates a number input for lot size
    user_input["lot_size"] = st.number_input("Lot Size", min_value=0.0, value=4000.0)

# This line checks whether basement exists in the feature columns
if "basement" in feature_columns:
    # This line creates a dropdown for basement
    user_input["basement"] = st.selectbox("Basement", [0, 1])

# This line checks whether popular exists in the feature columns
if "popular" in feature_columns:
    # This line creates a dropdown for popular location
    user_input["popular"] = st.selectbox("Popular Area", [0, 1])

# This line checks whether recession exists in the feature columns
if "recession" in feature_columns:
    # This line creates a dropdown for recession period
    user_input["recession"] = st.selectbox("Recession", [0, 1])

# This line checks whether property_age exists in the feature columns
if "property_age" in feature_columns:
    # This line calculates the property age from year sold and year built
    calculated_property_age = user_input.get("year_sold", 0) - user_input.get("year_built", 0)

    # This line prevents negative property age values
    if calculated_property_age < 0:
        # This line sets negative age to zero
        calculated_property_age = 0

    # This line shows the calculated property age in the app
    st.number_input("Property Age", value=float(calculated_property_age), disabled=True)

    # This line stores the calculated property age in user input
    user_input["property_age"] = float(calculated_property_age)

# This line checks whether property_type_Condo exists in the feature columns
if "property_type_Condo" in feature_columns:
    # This line creates a dropdown for condo property type
    user_input["property_type_Condo"] = st.selectbox("Property Type Condo", [0, 1])

# This line adds any missing feature columns with default value zero
for column in feature_columns:
    # This line checks whether the feature is not already collected
    if column not in user_input:
        # This line checks whether the missing feature is binary
        if column in binary_columns:
            # This line sets default binary value
            user_input[column] = 0
        else:
            # This line sets default numeric value
            user_input[column] = 0.0

# This line checks if the Predict button is clicked
if st.button("Predict"):
    # This line starts a try block for safe prediction
    try:
        # This line gets the predicted price
        result = predict_real_estate_price(user_input)

        # This line displays the prediction result
        st.success(f"Estimated House Price: ${result:,.2f}")

    # This block shows error messages in the app
    except Exception as exc:
        # This line displays the error
        st.error(f"Prediction failed: {exc}")