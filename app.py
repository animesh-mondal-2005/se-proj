import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_predictor.joblib")

model = load_model()

# ----------------------------
# Load dataset for SHAP explainer reference
# ----------------------------
@st.cache_resource
def load_data():
    return pd.read_csv("heart.csv").drop("target", axis=1)

X_reference = load_data()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Predictor with Explainable AI")
st.markdown("""
This system predicts the risk of **heart disease** and explains **why** the model made its prediction using SHAP values.  
Please provide patient details below:
""")

# ----------------------------
# Input fields
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0=Normal, 1=ST-T abnormality, 2=LVH)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0=Upsloping, 1=Flat, 2=Downsloping)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1=Fixed, 2=Normal, 3=Reversible)", [1, 2, 3])

# ----------------------------
# Prediction + SHAP Explanation
# ----------------------------
if st.button("🔍 Predict with Explanation"):
    # Convert categorical values
    sex_val = 1 if sex == "Male" else 0

    # Prepare input dataframe
    sample = pd.DataFrame([{
        "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    # Predict
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({1-probability:.2%})")

    # SHAP Explanation
    st.subheader("🧠 Why this Prediction? (Explainable AI)")
    shap_values = explainer.shap_values(sample)

    # Plot SHAP bar chart
    shap.initjs()
    plt.title("Feature Impact on Prediction")
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

    # Optional: Local force plot
    st.subheader("📊 Detailed Local Explanation (Force Plot)")
    shap.force_plot(explainer.expected_value[1], shap_values[1], sample, matplotlib=True, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed as part of a Software Engineering project using ML + Explainable AI (SHAP).")
