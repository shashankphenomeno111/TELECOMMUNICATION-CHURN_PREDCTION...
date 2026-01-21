# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 01:02:39 2025

@author: Rajiv Anatwar
"""
import streamlit as st
import pandas as pd
import joblib

# --- This feature list MUST EXACTLY MATCH the training script ---
FEATURES_FOR_MODEL = [
    'total_charge', 'customer_service_calls', 'international_plan', 'day_mins', 
    'day_charge', 'voice_mail_messages', 'international_calls', 
    'voice_mail_plan', 'international_mins', 'evening_charge'
]

# --- 1. App Title ---
st.title("Customer Churn Predictor")
st.write("This app uses your custom XGBoost model to predict churn.")

# --- 2. Load Your Model ---
try:
    model = joblib.load('xgb_churn_model.joblib')
except FileNotFoundError:
    st.error("Error: 'xgb_churn_model.joblib' not found. Please run the training script first.")
    st.stop()

# --- 3. User Input Interface ---
# Create input fields ONLY for the 10 features your model needs
st.header("Enter Customer Details")

inputs = {}
inputs['day_mins'] = st.number_input('Total Day Minutes', min_value=0.0, format="%.2f")
inputs['day_charge'] = st.number_input('Total Day Charge ($)', min_value=0.0, format="%.2f")
inputs['evening_charge'] = st.number_input('Total Evening Charge ($)', min_value=0.0, format="%.2f")
inputs['international_mins'] = st.number_input('Total International Minutes', min_value=0.0, format="%.2f")
inputs['international_calls'] = st.number_input('Total International Calls', min_value=0)
inputs['total_charge'] = st.number_input('Total Charge ($)', min_value=0.0, format="%.2f")
inputs['customer_service_calls'] = st.number_input('Calls to Customer Service', min_value=0)
inputs['voice_mail_messages'] = st.number_input('Number of Voice Mail Messages', min_value=0)
international_plan_str = st.selectbox('Has International Plan?', ['no', 'yes'])
voice_mail_plan_str = st.selectbox('Has Voice Mail Plan?', ['no', 'yes'])

# --- 4. Prediction Logic ---
if st.button('Predict Churn'):
    # Convert string 'yes'/'no' to 1/0
    inputs['international_plan'] = 1 if international_plan_str == 'yes' else 0
    inputs['voice_mail_plan'] = 1 if voice_mail_plan_str == 'yes' else 0

    # Create the DataFrame and re-order it to match the training order
    input_df = pd.DataFrame([inputs])[FEATURES_FOR_MODEL]
    
    # Make the prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # --- 5. Display the Prediction ---
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error(f"Prediction: LIKELY to Churn (Confidence: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"Prediction: UNLIKELY to Churn (Confidence: {prediction_proba[0][0]:.2%})")
