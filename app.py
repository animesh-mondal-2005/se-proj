import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="💓 Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Title and Description ---
st.title("💓 Heart Disease Predictor")
st.markdown("""
Predict your risk of heart disease based on health parameters.  
Adjust the inputs below and click **Predict** to see the result.
""")

# Load the model
model = joblib.load("heart_predictor.joblib")

# --- Layout Inputs in Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

with col3:
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Unknown", "Fixed Defect", "Normal", "Reversible Defect"])

# --- Convert Text Inputs to Numeric ---
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Unknown", "Fixed Defect", "Normal", "Reversible Defect"].index(thal)

# --- Prediction Button ---
st.markdown("---")
if st.button("💡 Predict"):
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    pred = model.predict(sample)[0]

    if pred == 1:
        st.error("⚠️ Prediction: Heart Disease Detected")
        st.warning("Please consult a medical professional for further evaluation.")
    else:
        st.success("✅ Prediction: No Heart Disease")
        st.info("Keep maintaining a healthy lifestyle!")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
