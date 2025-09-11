import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_predictor.joblib")

model = load_model()

# ----------------------------
# Conversational State
# ----------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# Ordered list of questions (map feature → question)
questions = [
    ("age", "👤 What is your age?"),
    ("sex", "⚧️ What is your gender? (Male/Female)"),
    ("cp", "💢 Chest Pain Type? (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)"),
    ("trestbps", "🩺 Resting Blood Pressure (mm Hg)?"),
    ("chol", "🥓 Serum Cholesterol (mg/dl)?"),
    ("fbs", "🧪 Fasting Blood Sugar > 120 mg/dl? (0=No, 1=Yes)"),
    ("restecg", "🫀 Resting ECG (0=Normal, 1=ST-T abnormality, 2=LVH)?"),
    ("thalach", "🏃 Max Heart Rate Achieved?"),
    ("exang", "😰 Exercise Induced Angina? (0=No, 1=Yes)"),
    ("oldpeak", "📉 ST Depression (Oldpeak value)?"),
    ("slope", "📈 Slope (0=Upsloping, 1=Flat, 2=Downsloping)?"),
    ("ca", "🔍 Number of major vessels (0–3)?"),
    ("thal", "🧬 Thal (1=Fixed, 2=Normal, 3=Reversible)?")
]

# ----------------------------
# Personalized Advice
# ----------------------------
def give_advice(risk, age, sex):
    advice = []
    if risk > 0.7:
        advice.append("⚠️ High risk detected! Please consult a cardiologist immediately.")
    elif risk > 0.4:
        advice.append("⚠️ Moderate risk. Regular check-ups and lifestyle changes recommended.")
    else:
        advice.append("✅ Low risk. Maintain a healthy lifestyle.")

    # Age-based
    if age > 50:
        advice.append("🔹 Since you are above 50, focus on regular cardiac checkups.")
    else:
        advice.append("🔹 Maintain physical activity and avoid junk food.")

    # Gender-based
    if sex.lower().startswith("m"):
        advice.append("🔹 Reduce smoking/alcohol, and monitor cholesterol closely.")
    else:
        advice.append("🔹 Manage stress, maintain balanced diet, and regular walks are beneficial.")

    return " ".join(advice)

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Heart Disease Chatbot", layout="centered")
st.title("❤️ Heart Disease Predictor - Chatbot Mode")

st.markdown("Answer the following step by step. After the last question, the prediction will be shown.")

# Current question
if st.session_state.step < len(questions):
    feature, q_text = questions[st.session_state.step]
    st.subheader(q_text)

    user_input = st.text_input("Your answer:", key=f"q_{feature}")

    if st.button("➡️ Next"):
        if user_input.strip() != "":
            st.session_state.inputs[feature] = user_input
            st.session_state.step += 1
            st.experimental_rerun()
        else:
            st.warning("Please provide an answer before continuing.")

# ----------------------------
# Once All Inputs Collected → Prediction
# ----------------------------
else:
    st.success("✅ All inputs collected! Running prediction...")

    try:
        # Convert inputs
        inputs = st.session_state.inputs
        sex_val = 1 if str(inputs["sex"]).lower().startswith("m") else 0

        sample = pd.DataFrame([{
            "age": int(inputs["age"]),
            "sex": sex_val,
            "cp": int(inputs["cp"]),
            "trestbps": int(inputs["trestbps"]),
            "chol": int(inputs["chol"]),
            "fbs": int(inputs["fbs"]),
            "restecg": int(inputs["restecg"]),
            "thalach": int(inputs["thalach"]),
            "exang": int(inputs["exang"]),
            "oldpeak": float(inputs["oldpeak"]),
            "slope": int(inputs["slope"]),
            "ca": int(inputs["ca"]),
            "thal": int(inputs["thal"])
        }])

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The model predicts **Heart Disease Risk** with probability {probability:.2%}")
        else:
            st.success(f"✅ The model predicts **No Heart Disease** with probability {1-probability:.2%}")

        # Advice
        st.markdown("### 📌 Personalized Advice")
        st.info(give_advice(probability, int(inputs["age"]), inputs["sex"]))

    except Exception as e:
        st.error(f"Error in prediction: {e}")
