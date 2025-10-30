import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('logistic_model.pkl', 'rb'))

# Streamlit App
st.title("🩺 Diabetes Detection App (Balanced Logistic Regression)")
st.markdown("This model was trained without the **Pregnancies** column and balanced using SMOTE for better accuracy.")

# Input fields (7 features)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.3)
Age = st.number_input("Age", min_value=1, max_value=120, value=35)

# Prepare input data
input_data = pd.DataFrame(
    [[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
    columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
)

# Predict button
if st.button("Predict"):
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        if prediction[0] == 1:
            st.error("⚠️ The person is **likely diabetic.**")
        else:
            st.success("✅ The person is **not diabetic.**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
