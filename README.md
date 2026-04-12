# Real Estate Price Prediction using Machine Learning

## Overview
This project converts the Real Estate notebook into a modular Python project and an interactive Streamlit web application.

The goal is to predict property prices using a supervised machine learning model.

The project follows the notebook workflow:

- Load dataset  
- Perform preprocessing and feature engineering  
- Train multiple models  
- Evaluate performance using MAE  
- Select best model  
- Save artifacts  
- Deploy using Streamlit  


## Problem Statement
Real estate pricing depends on multiple factors such as size, structure, and market indicators.

This project helps:
- Buyers estimate property value  
- Sellers price homes correctly  
- Analysts make data-driven decisions  


## Dataset
Features used:

- year_sold  
- property_tax  
- insurance  
- beds  
- baths  
- sqft  
- year_built  
- lot_size  
- basement  
- popular  
- recession  
- property_age  
- property_type_Condo  

Target:
- price  

Dataset file:  
`data/final.csv`


## Model Workflow
- Load dataset  
- Preprocess data  
- Feature engineering  
- Train-test split  
- Train Linear Regression  
- Train Random Forest  
- Evaluate using MAE  
- Select best model  
- Save model and artifacts  


## Model Performance
- Linear Regression MAE: ~86,000  
- Random Forest MAE: ~46,000  

Random Forest is selected as the final model.


## Prediction
The model predicts:
- Estimated house price based on input features  


## How to Run

# Install dependencies:
pip install -r requirements.txt

# Run pipeline:
python main.py

# Run app:
streamlit run app.py

# Streamlit Link
https://realestate-maksuda.streamlit.app/


# Key Features
Modular code structure
Logging and error handling
Model comparison
Clean preprocessing pipeline
Interactive Streamlit app
Styled UI

# Project Structure
```text
REAL_ESTATE/
│
├── app.py  
├── main.py  
├── requirements.txt  
├── runtime.txt  
├── README.md  
│
├── artifacts/  
│   ├── random_forest_model.joblib  
│   ├── feature_columns.json  
│   └── metrics.json  
│
├── data/  
│   └── final.csv  
│
├── logs/  
│   └── project.log  
│
├── notebooks/  
│   └── Real_Estate_Solution.ipynb  
│
├── src/  
│   ├── __init__.py  
│   ├── config.py  
│   ├── custom_exception.py  
│   ├── data_loader.py  
│   ├── evaluate.py  
│   ├── logger.py  
│   ├── modeling.py  
│   ├── predict.py  
│   ├── preprocessing.py  
│   ├── train.py  
│   └── utils.py  
