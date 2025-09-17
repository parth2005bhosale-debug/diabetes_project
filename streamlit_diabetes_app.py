import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.joblib")

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction Dashboard")
st.write("Enter patient details below to check diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47, format="%.2f")
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Collect inputs into dataframe
features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
input_df = pd.DataFrame(features, columns=columns)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error("⚠️ High risk of Diabetes detected.")
    else:
        st.success("✅ Low risk of Diabetes detected.")

    if prob is not None:
        st.write(f"**Probability of Diabetes:** {prob*100:.2f}%")
