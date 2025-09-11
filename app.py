import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_predictor.joblib")

st.title("Heart Disease Predictor")

age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [1, 0])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1/0)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0=Unknown, 1=Fixed, 2=Normal, 3=Reversible)", [0, 1, 2, 3])

if st.button("Predict"):
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    pred = model.predict(sample)[0]
    st.success("Prediction: Heart Disease" if pred == 1 else "Prediction: No Heart Disease")
