import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('aluminum_wire_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Aluminum Wire Properties Predictor')

st.write("""
This app predicts the Ultimate Tensile Strength (UTS), Elongation, and Conductivity 
of aluminum wire based on casting parameters.
""")

# Input fields
casting_temp = st.number_input('Casting Temperature (°C)', min_value=600.0, max_value=800.0, value=700.0)
rolling_speed = st.number_input('Rolling Speed (m/min)', min_value=100.0, max_value=200.0, value=150.0)
cooling_rate = st.number_input('Cooling Rate (°C/s)', min_value=20.0, max_value=50.0, value=35.0)

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'Casting_Temperature_C': [casting_temp],
    'Rolling_Speed_m_min': [rolling_speed],
    'Cooling_Rate_C_s': [cooling_rate],
    'Temp_Speed_Interaction': [casting_temp * rolling_speed],
    'Temp_Cooling_Interaction': [casting_temp * cooling_rate]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display results
    st.subheader('Predicted Properties:')
    st.write(f'Ultimate Tensile Strength (UTS): {prediction[0][0]:.2f} MPa')
    st.write(f'Elongation: {prediction[0][1]:.2f} %')
    st.write(f'Conductivity: {prediction[0][2]:.2f} % IACS')

st.write("""
Note: This model is based on synthetic data and should not be used for real-world applications 
without proper validation.
""")