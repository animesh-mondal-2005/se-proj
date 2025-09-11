import streamlit as st
import pandas as pd
import joblib
import speech_recognition as sr
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("heart_predictor.joblib")

model = load_model()

st.markdown("""
    <style>
    .main-title {
        font-size:32px;
        color:#d32f2f;
        text-align:center;
        font-weight:bold;
    }
    .chat-bubble {
        background:#fff3f3;
        padding:10px;
        border-radius:10px;
        margin:5px 0;
        border:1px solid #ffcccc;
    }
    .user-bubble {
        background:#e3f2fd;
        padding:10px;
        border-radius:10px;
        margin:5px 0;
        border:1px solid #90caf9;
        text-align:right;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">❤️ Heart Health Chatbot with Explainable AI</div>', unsafe_allow_html=True)

def listen_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("🎤 Listening... please speak now")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"✅ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio, please type instead.")
        return None
    except sr.RequestError:
        st.error("⚠️ Speech recognition service unavailable.")
        return None

questions = [
    ("age", "What is your age?"),
    ("sex", "What is your sex? (Male/Female)"),
    ("cp", "Chest pain type? (0=Typical,1=Atypical,2=Non-anginal,3=Asymptomatic)"),
    ("trestbps", "Resting blood pressure (mm Hg)?"),
    ("chol", "Cholesterol level (mg/dl)?"),
    ("fbs", "Is fasting blood sugar > 120 mg/dl? (0=No, 1=Yes)"),
    ("restecg", "Resting ECG? (0=Normal,1=ST-T abnormality,2=LVH)"),
    ("thalach", "Max heart rate achieved?"),
    ("exang", "Exercise induced angina? (0=No, 1=Yes)"),
    ("oldpeak", "ST depression (oldpeak)?"),
    ("slope", "Slope of ST segment? (0=Upsloping,1=Flat,2=Downsloping)"),
    ("ca", "Number of major vessels (0-3)?"),
    ("thal", "Thalassemia? (1=Fixed, 2=Normal, 3=Reversible)")
]

responses = {}

for key, question in questions:
    st.markdown(f'<div class="chat-bubble">{question}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        ans = st.text_input(f"Answer for {key}", key=key)
    with col2:
        if st.button("🎤 Speak", key=f"btn_{key}"):
            voice_text = listen_voice()
            if voice_text:
                st.session_state[key] = voice_text
                ans = voice_text
    if ans:
        responses[key] = ans
        st.markdown(f'<div class="user-bubble">{ans}</div>', unsafe_allow_html=True)

if len(responses) == len(questions) and st.button("🔍 Predict"):
    try:
        # Preprocess inputs
        sex_val = 1 if str(responses["sex"]).lower() == "male" else 0
        data = pd.DataFrame([{
            "age": int(responses["age"]),
            "sex": sex_val,
            "cp": int(responses["cp"]),
            "trestbps": int(responses["trestbps"]),
            "chol": int(responses["chol"]),
            "fbs": int(responses["fbs"]),
            "restecg": int(responses["restecg"]),
            "thalach": int(responses["thalach"]),
            "exang": int(responses["exang"]),
            "oldpeak": float(responses["oldpeak"]),
            "slope": int(responses["slope"]),
            "ca": int(responses["ca"]),
            "thal": int(responses["thal"])
        }])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        if prediction == 1:
            result_text = f"⚠️ High Risk of Heart Disease ({probability:.2%})"
            st.error(result_text)
        else:
            result_text = f"✅ Low Risk of Heart Disease ({1-probability:.2%})"
            st.success(result_text)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        st.subheader("🔎 Why this prediction?")
        fig, ax = plt.subplots()
        shap.bar_plot(shap_values[1][0], feature_names=data.columns, max_display=5, show=False)
        st.pyplot(fig)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Heart Disease Prediction Report", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, "Patient Inputs:", ln=True)
        for k, v in responses.items():
            pdf.cell(200, 8, f"{k}: {v}", ln=True)

        pdf.cell(200, 10, f"Prediction Result: {result_text}", ln=True)
        pdf.cell(200, 10, "Key Factors Influencing Prediction:", ln=True)

        feature_importance = pd.Series(shap_values[1][0], index=data.columns).sort_values(ascending=False)
        for feat, val in feature_importance.head(5).items():
            pdf.cell(200, 8, f"{feat}: {val:.4f}", ln=True)

        pdf.output("Heart_Report.pdf")
        with open("Heart_Report.pdf", "rb") as file:
            st.download_button("📄 Download Report", file, file_name="Heart_Report.pdf")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
