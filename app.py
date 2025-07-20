import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

model = tf.keras.models.load_model('diabetes_model.keras')
diabetes_data = pd.read_csv('diabetes.csv')

# Prepare the data
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the details below to predict if a person is diabetic:")


preg = st.number_input('Pregnancies', min_value=0, value=1)
glucose = st.number_input('Glucose', min_value=0, value=120)
bp = st.number_input('Blood Pressure', min_value=0, value=70)
skin = st.number_input('Skin Thickness', min_value=0, value=20)
insulin = st.number_input('Insulin', min_value=0, value=79)
bmi = st.number_input('BMI', min_value=0.0, format="%.1f", value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f", value=0.5)
age = st.number_input('Age', min_value=1, value=33)

if st.button('Predict Your Diabetic Status : '):
    input_data = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
    input_ = scaler.transform(input_data)
    pred = model.predict(input_)
    result = "Diabetic" if pred[0][0] > 0.5 else "Non-Diabetic"
    st.success(f"\nPrediction: {result} (Probability: {pred[0][0]:.2f})")







