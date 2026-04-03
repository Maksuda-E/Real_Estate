#  imports streamlit for the web app
import streamlit as st

#  imports the prediction function and model loader
from src.predict import predict_real_estate_price, load_model_and_features

#  sets the page title and layout
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide"
)

#  adds custom CSS for a fresh design and color theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0fdf4, #ecfeff, #f8fafc);
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        text-align: center;
        color: #14532d;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #475569;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #166534;
        margin-bottom: 1rem;
    }

    .side-card {
        background: linear-gradient(135deg, #166534, #0f766e);
        color: white;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 14px 30px rgba(15, 118, 110, 0.20);
        margin-bottom: 1rem;
    }

    .side-card-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .side-card-text {
        font-size: 1rem;
        line-height: 1.75;
        opacity: 0.97;
    }

    .result-card {
        background: linear-gradient(135deg, #ea580c, #f59e0b);
        color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 12px 28px rgba(245, 158, 11, 0.24);
        margin-top: 1rem;
    }

    .result-title {
        font-size: 0.95rem;
        opacity: 0.95;
        margin-bottom: 0.45rem;
    }

    .result-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .result-text {
        font-size: 1rem;
        opacity: 0.96;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #15803d, #0f766e);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(21, 128, 61, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #166534, #115e59);
    }
    </style>
    """,
    unsafe_allow_html=True
)

#  displays the main title
st.markdown(
    '<div class="main-title">Real Estate Price Prediction App</div>',
    unsafe_allow_html=True
)

#  displays the subtitle
st.markdown(
    '<div class="sub-title">Enter property details to estimate the house price</div>',
    unsafe_allow_html=True
)

#  loads the saved model and feature columns
model, feature_columns = load_model_and_features()

#  defines binary columns expected by the model
binary_columns = ["basement", "popular", "recession", "property_type_Condo"]

#  initializes a dictionary to store user inputs
user_input = {}

#  creates a centered page layout
left_space, center_col, right_space = st.columns([0.4, 4, 0.4])

# This block contains the full page content
with center_col:

    #  creates two columns with a different layout style
    form_col, side_col = st.columns([2.3, 1], gap="large")

    # This block creates the main form section
    with form_col:

        #  shows the form section heading
        st.markdown('<div class="section-title">Property Details</div>', unsafe_allow_html=True)

        #  creates the first row of numeric inputs
        row1_col1, row1_col2, row1_col3 = st.columns(3)

        # This block creates the year sold input if it exists in features
        with row1_col1:
            if "year_sold" in feature_columns:
                user_input["year_sold"] = st.number_input(
                    "Year Sold",
                    min_value=1900,
                    max_value=2100,
                    value=2010,
                    step=1
                )

        # This block creates the property tax input if it exists in features
        with row1_col2:
            if "property_tax" in feature_columns:
                user_input["property_tax"] = st.number_input(
                    "Property Tax",
                    min_value=0.0,
                    value=3000.0
                )

        # This block creates the insurance input if it exists in features
        with row1_col3:
            if "insurance" in feature_columns:
                user_input["insurance"] = st.number_input(
                    "Insurance",
                    min_value=0.0,
                    value=1200.0
                )

        #  creates the second row of numeric inputs
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # This block creates the beds input if it exists in features
        with row2_col1:
            if "beds" in feature_columns:
                user_input["beds"] = st.number_input(
                    "Beds",
                    min_value=0.0,
                    value=3.0
                )

        # This block creates the baths input if it exists in features
        with row2_col2:
            if "baths" in feature_columns:
                user_input["baths"] = st.number_input(
                    "Baths",
                    min_value=0.0,
                    value=2.0
                )

        # This block creates the square feet input if it exists in features
        with row2_col3:
            if "sqft" in feature_columns:
                user_input["sqft"] = st.number_input(
                    "Square Feet",
                    min_value=0.0,
                    value=1800.0
                )

        #  creates the third row of numeric inputs
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        # This block creates the year built input if it exists in features
        with row3_col1:
            if "year_built" in feature_columns:
                user_input["year_built"] = st.number_input(
                    "Year Built",
                    min_value=1800,
                    max_value=2100,
                    value=2000,
                    step=1
                )

        # This block creates the lot size input if it exists in features
        with row3_col2:
            if "lot_size" in feature_columns:
                user_input["lot_size"] = st.number_input(
                    "Lot Size",
                    min_value=0.0,
                    value=4000.0
                )

        # This block shows calculated property age if it exists in features
        with row3_col3:
            if "property_age" in feature_columns:
                calculated_property_age = user_input.get("year_sold", 0) - user_input.get("year_built", 0)

                if calculated_property_age < 0:
                    calculated_property_age = 0

                st.number_input(
                    "Property Age",
                    value=float(calculated_property_age),
                    disabled=True
                )

                user_input["property_age"] = float(calculated_property_age)

        #  creates the fourth row of binary inputs
        row4_col1, row4_col2, row4_col3, row4_col4 = st.columns(4)

        # This block creates the basement input if it exists in features
        with row4_col1:
            if "basement" in feature_columns:
                user_input["basement"] = st.selectbox("Basement", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # This block creates the popular input if it exists in features
        with row4_col2:
            if "popular" in feature_columns:
                user_input["popular"] = st.selectbox("Popular Area", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # This block creates the recession input if it exists in features
        with row4_col3:
            if "recession" in feature_columns:
                user_input["recession"] = st.selectbox("Recession Period", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        # This block creates the condo input if it exists in features
        with row4_col4:
            if "property_type_Condo" in feature_columns:
                user_input["property_type_Condo"] = st.selectbox("Condo", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        #  fills any missing expected feature columns with default values
        for column in feature_columns:
            if column not in user_input:
                if column in binary_columns:
                    user_input[column] = 0
                else:
                    user_input[column] = 0.0

        #  adds a little space before the button
        st.markdown("<br>", unsafe_allow_html=True)

        #  creates the prediction button
        predict_button = st.button("Predict House Price")

    # This block creates the side overview section
    with side_col:

        #  reads values for a cleaner overview message
        year_sold_val = user_input.get("year_sold", 0)
        year_built_val = user_input.get("year_built", 0)
        sqft_val = user_input.get("sqft", 0)
        beds_val = user_input.get("beds", 0)
        baths_val = user_input.get("baths", 0)
        lot_size_val = user_input.get("lot_size", 0)
        property_age_val = user_input.get("property_age", 0)
        condo_val = user_input.get("property_type_Condo", 0)

        #  displays the overview card
        st.markdown(
            f"""
            <div class="side-card">
                <div class="side-card-title">Property Overview</div>
                <div class="side-card-text">
                    This property includes {beds_val:.0f} bedrooms and {baths_val:.0f} bathrooms with about {sqft_val:.0f} square feet of living space.
                    It was built in {year_built_val:.0f} and evaluated for sale in {year_sold_val:.0f}, giving it an estimated age of {property_age_val:.0f} years.
                    The lot size is {lot_size_val:.0f}, and the property type is marked as {"condo" if condo_val == 1 else "non condo"}.
                    These details are sent to the saved trained model to estimate the market price.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        #  checks if a previous prediction exists
        if "real_estate_result" in st.session_state:

            #  gets the saved prediction
            result = st.session_state["real_estate_result"]

            #  displays the result card below the overview
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Estimated Price</div>
                    <div class="result-value">${result:,.2f}</div>
                    <div class="result-text">
                        This estimate is generated from the trained regression model saved by your pipeline.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    #  checks if the Predict button is clicked
    if predict_button:

        #  starts a try block for safe prediction
        try:
            #  gets the predicted price
            result = predict_real_estate_price(user_input)

            #  stores the result in session state
            st.session_state["real_estate_result"] = result

            #  reruns the app so the result appears immediately
            st.rerun()

        # This block handles prediction errors
        except Exception as exc:
            #  displays the prediction error
            st.error(f"Prediction failed: {exc}")
