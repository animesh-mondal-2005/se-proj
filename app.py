# import streamlit as st
# import pandas as pd
# import re
# import joblib
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# st.set_page_config(
#     page_title="Heart Disease Predictor",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #e74c3c;
#         text-align: center;
#         margin-bottom: 2rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .section-header {
#         font-size: 1.5rem;
#         color: #2c3e50;
#         margin: 1rem 0;
#         border-bottom: 2px solid #3498db;
#         padding-bottom: 0.5rem;
#     }
#     .info-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 4px solid #3498db;
#         margin: 1rem 0;
#     }
#     .prediction-positive {
#         background-color: #ffebee;
#         color: #c62828;
#         padding: 1rem;
#         border-radius: 10px;
#         border: 2px solid #ef5350;
#         text-align: center;
#         font-size: 1.2rem;
#         font-weight: bold;
#     }
#     .prediction-negative {
#         background-color: #e8f5e8;
#         color: #2e7d32;
#         padding: 1rem;
#         border-radius: 10px;
#         border: 2px solid #66bb6a;
#         text-align: center;
#         font-size: 1.2rem;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_resource
# def load_model():
#     try:
#         return joblib.load("heart_predictor_rf.joblib")
#     except FileNotFoundError:
#         st.error("⚠️ Model file 'heart_predictor_rf.joblib' not found. Please ensure the model file is in the same directory.")
#         st.stop()

# model = load_model()

# st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)

# st.markdown("""
# <div class="info-box" >
#     <h3 style="color:red;" >🩺 About This Tool</h3>
#     <p style="color:red;">This AI-powered tool helps assess the risk of heart disease based on various medical parameters. 
#     Please consult with a healthcare professional for proper medical diagnosis and treatment.</p>
# </div>
# """, unsafe_allow_html=True)

# with st.sidebar:
#     st.markdown("### 📋 Parameter Guide")
#     st.markdown("""
#     **Chest Pain Types:**
#     - 0: Typical Angina
#     - 1: Atypical Angina  
#     - 2: Non-Anginal Pain
#     - 3: Asymptomatic
    
#     **Rest ECG:**
#     - 0: Normal
#     - 1: ST-T Wave Abnormality
#     - 2: Left Ventricular Hypertrophy
    
#     **Slope:**
#     - 0: Upsloping
#     - 1: Flat
#     - 2: Downsloping

#     **Thalassemia:**
#     - 0: Unknown
#     - 1: Fixed Defect
#     - 2: Normal Flow
#     - 3: Reversible Defect
#     """)

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown('<h2 class="section-header">📊 Patient Information</h2>', unsafe_allow_html=True)

#     demo_col1, demo_col2 = st.columns(2)
#     with demo_col1:
#         age = st.number_input("👤 Age", min_value=18, max_value=100, help="Patient's age in years")
#         sex = st.selectbox("⚧ Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
#     with demo_col2:
#         cp = st.selectbox("💔 Chest Pain Type", options=[0, 1, 2, 3], 
#                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
#         fbs = st.selectbox("🍯 Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
#                           format_func=lambda x: "Yes" if x == 1 else "No")

#     st.markdown('<h3 class="section-header">🩺 Vital Signs & Lab Results</h3>', unsafe_allow_html=True)
    
#     vital_col1, vital_col2 = st.columns(2)
#     with vital_col1:
#         trestbps = st.number_input("🩸 Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
#         chol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
#         thalach = st.number_input("💓 Maximum Heart Rate", min_value=60, max_value=220, value=150)
    
#     with vital_col2:
#         restecg = st.selectbox("📈 Resting ECG", options=[0, 1, 2], 
#                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
#         exang = st.selectbox("🏃 Exercise Induced Angina", options=[0, 1], 
#                             format_func=lambda x: "Yes" if x == 1 else "No")
#         oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

#     st.markdown('<h3 class="section-header">🔬 Advanced Parameters</h3>', unsafe_allow_html=True)
    
#     adv_col1, adv_col2 = st.columns(2)
#     with adv_col1:
#         slope = st.selectbox("📊 ST Slope", options=[0, 1, 2], 
#                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
#         ca = st.selectbox("💓 Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    
#     with adv_col2:
#         thal = st.selectbox("🫀 Thalassemia", options=[0, 1, 2, 3], 
#                            format_func=lambda x: ["Unknown/Normal", "Fixed Defect", "Normal Flow", "Reversible Defect"][x])

# with col2:
#     st.markdown('<h2 class="section-header">📈 Risk Assessment</h2>', unsafe_allow_html=True)

#     st.markdown("""
#     ### 💡 Heart Health Tips
#     - 🥗 Maintain a balanced diet
#     - 🏃‍♂️ Regular exercise
#     - 🚭 Avoid smoking
#     - 😌 Manage stress
#     - 💊 Take medications as prescribed
#     """)

# st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

# predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

# with predict_col2:
#     if st.button("🔍 Analyze Heart Disease Risk", type="primary", use_container_width=True):

#         sample = pd.DataFrame([{
#             "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
#             "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
#             "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
#         }])

#         try:
#             pred = model.predict(sample)[0]

#             if pred == 1:
#                 st.markdown("""
#                 <div class="prediction-positive">
#                     🚨 HIGH RISK: Heart Disease Detected<br>
#                     <small>Please consult a cardiologist immediately</small>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.error("⚠️ **Important:** This is a screening tool only. Please seek immediate medical attention for proper diagnosis and treatment.")
                
#             else:
#                 st.markdown("""
#                 <div class="prediction-negative">
#                     ✅ LOW RISK: No Heart Disease Detected<br>
#                     <small>Continue maintaining a healthy lifestyle</small>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.success("✅ **Good News:** Low risk detected. Continue regular check-ups and maintain a healthy lifestyle.")
                
#         except Exception as e:
#             st.error(f"❌ Prediction Error: {str(e)}")

# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
#     <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.</p>
#     <p>Always consult with qualified healthcare providers regarding medical conditions.</p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Try to import SHAP and handle gracefully if missing
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------
# Styles (kept similar)
# --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .prediction-positive {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ef5350;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #66bb6a;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .explanation-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .factor-positive { color: #e74c3c; font-weight: bold; }
    .factor-negative { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Feature labels (for human-readable output)
FEATURE_DESCRIPTIONS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol Level",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia",
}

FEATURE_VALUE_DESCRIPTIONS = {
    "cp": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "restecg": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal": ["Unknown/Normal", "Fixed Defect", "Normal Flow", "Reversible Defect"],
    "sex": ["Female", "Male"],
    "fbs": ["No", "Yes"],
    "exang": ["No", "Yes"],
}

def describe_value(feature: str, value):
    if feature in FEATURE_VALUE_DESCRIPTIONS:
        opts = FEATURE_VALUE_DESCRIPTIONS[feature]
        if isinstance(value, (int, float)) and 0 <= int(value) < len(opts):
            return opts[int(value)]
    return str(value)

@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_predictor_rf.joblib")
    except FileNotFoundError:
        st.error("⚠️ Model file 'heart_predictor_rf.joblib' not found. Place it next to this script.")
        st.stop()

model = load_model()

def create_synthetic_background(n=100):
    # small background for SHAP to stay responsive
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(30, 80, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(90, 180, n),
        "chol": rng.integers(150, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(100, 180, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 4, n),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n),
    })

def unify_shap_values(explainer, sample_df: pd.DataFrame):
    """
    Returns a 1D numpy array of SHAP values for the positive class (if classifier),
    robust to different SHAP APIs, and avoids 2D column issues.
    """
    # First try the classic .shap_values API
    try:
        sv = explainer.shap_values(sample_df)
        if isinstance(sv, list):
            # assume binary classifier: class index 1 is "disease"
            arr = np.asarray(sv[1]).reshape(sample_df.shape[0], -1)[0]
        else:
            arr = np.asarray(sv).reshape(sample_df.shape[0], -1)[0]
        return np.ravel(arr)
    except Exception:
        pass

    # Fallback to new callable API returning shap.Explanation
    exp = explainer(sample_df)
    # exp.values might be shape (n_samples, n_features)
    arr = np.asarray(getattr(exp, "values", exp)).reshape(sample_df.shape[0], -1)[0]
    return np.ravel(arr)

def build_contrib_df(sample_df: pd.DataFrame, shap_1d: np.ndarray) -> pd.DataFrame:
    # Ensure 1D and correct length
    shap_1d = np.ravel(shap_1d)
    if shap_1d.shape[0] != sample_df.shape[1]:
        raise ValueError(
            f"SHAP length {shap_1d.shape[0]} != number of features {sample_df.shape[1]}"
        )

    df = pd.DataFrame({
        "feature": list(sample_df.columns),
        "value": sample_df.iloc[0].tolist(),
        "shap": shap_1d.tolist(),
    })
    df["abs_shap"] = np.abs(df["shap"])
    return df.sort_values("abs_shap", ascending=False)

def render_factor_list(pred: int, prob: float, contrib_df: pd.DataFrame):
    inc = contrib_df[contrib_df["shap"] > 0]
    dec = contrib_df[contrib_df["shap"] < 0]

    if pred == 1:
        st.markdown(f"## 🔍 Why Heart Disease Was Predicted (Risk: {prob:.1%})")
        st.markdown("The model predicted high risk primarily due to these factors:")
        if not inc.empty:
            for _, row in inc.head(5).iterrows():
                fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
                fval = describe_value(row["feature"], row["value"])
                st.markdown(f"- **{fname}** = {fval} "
                            f"(<span class='factor-positive'>increased risk by {row['shap']:.3f}</span>)",
                            unsafe_allow_html=True)
        else:
            st.markdown("- No strong risk-increasing factors were identified.")
        if not dec.empty:
            st.markdown(" ")
            st.markdown("**Factors that reduced risk (but weren't enough):**")
            for _, row in dec.head(3).iterrows():
                fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
                fval = describe_value(row["feature"], row["value"])
                st.markdown(f"- **{fname}** = {fval} "
                            f"(<span class='factor-negative'>decreased risk by {abs(row['shap']):.3f}</span>)",
                            unsafe_allow_html=True)
    else:
        st.markdown(f"## 🔍 Why No Heart Disease Was Predicted (Risk: {prob:.1%})")
        st.markdown("The model predicted low risk primarily due to these factors:")
        if not dec.empty:
            for _, row in dec.head(5).iterrows():
                fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
                fval = describe_value(row["feature"], row["value"])
                st.markdown(f"- **{fname}** = {fval} "
                            f"(<span class='factor-negative'>decreased risk by {abs(row['shap']):.3f}</span>)",
                            unsafe_allow_html=True)
        else:
            st.markdown("- No strong risk-decreasing factors were identified.")
        if not inc.empty:
            st.markdown(" ")
            st.markdown("**Factors that increased risk (but were outweighed):**")
            for _, row in inc.head(3).iterrows():
                fname = FEATURE_DESCRIPTIONS.get(row["feature"], row["feature"])
                fval = describe_value(row["feature"], row["value"])
                st.markdown(f"- **{fname}** = {fval} "
                            f"(<span class='factor-positive'>increased risk by {row['shap']:.3f}</span>)",
                            unsafe_allow_html=True)

    if not contrib_df.empty:
        top = contrib_df.iloc[0]
        fname = FEATURE_DESCRIPTIONS.get(top["feature"], top["feature"])
        fval = describe_value(top["feature"], top["value"])
        st.markdown("### 💡 Key Insight:")
        if top["shap"] > 0:
            st.markdown(f"The most influential factor was **{fname}** ({fval}), which increased the risk.")
        else:
            st.markdown(f"The most influential factor was **{fname}** ({fval}), which decreased the risk.")

def plot_top_factors(contrib_df: pd.DataFrame, top_n: int = 8):
    df = contrib_df.copy().head(top_n)
    colors = ["#e74c3c" if x > 0 else "#27ae60" for x in df["shap"]]

    fig = go.Figure()
    fig.add_bar(
        x=df["shap"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
    )
    fig.update_layout(
        title="Top Factors Influencing Prediction",
        xaxis_title="Impact on Risk (SHAP Value)",
        yaxis_title="Features",
        height=420,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# UI
# --------------------
st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  <h3 style="color:red;">🩺 About This Tool</h3>
  <p style="color:red;">This AI-powered tool helps assess the risk of heart disease based on various medical parameters.
  Please consult with a healthcare professional for proper medical diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📋 Parameter Guide")
    st.markdown("""
**Chest Pain Types:**
- 0: Typical Angina
- 1: Atypical Angina
- 2: Non-Anginal Pain
- 3: Asymptomatic

**Rest ECG:**
- 0: Normal
- 1: ST-T Wave Abnormality
- 2: Left Ventricular Hypertrophy

**Slope:**
- 0: Upsloping
- 1: Flat
- 2: Downsloping

**Thalassemia:**
- 0: Unknown
- 1: Fixed Defect
- 2: Normal Flow
- 3: Reversible Defect
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Patient Information</h2>', unsafe_allow_html=True)

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        age = st.number_input("👤 Age", min_value=18, max_value=100, value=50)
        sex = st.selectbox("⚧ Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    with demo_col2:
        cp = st.selectbox("💔 Chest Pain Type", options=[0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
        fbs = st.selectbox("🍯 Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown('<h3 class="section-header">🩺 Vital Signs & Lab Results</h3>', unsafe_allow_html=True)
    vital_col1, vital_col2 = st.columns(2)
    with vital_col1:
        trestbps = st.number_input("🩸 Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("💓 Maximum Heart Rate", min_value=60, max_value=220, value=150)
    with vital_col2:
        restecg = st.selectbox("📈 Resting ECG", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        exang = st.selectbox("🏃 Exercise Induced Angina", options=[0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.markdown('<h3 class="section-header">🔬 Advanced Parameters</h3>', unsafe_allow_html=True)
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        slope = st.selectbox("📊 ST Slope", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("💓 Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    with adv_col2:
        thal = st.selectbox("🫀 Thalassemia", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Unknown/Normal", "Fixed Defect", "Normal Flow", "Reversible Defect"][x])

with col2:
    st.markdown('<h2 class="section-header">📈 Risk Assessment</h2>', unsafe_allow_html=True)
    st.markdown("""
### 💡 Heart Health Tips
- 🥗 Maintain a balanced diet
- 🏃‍♂️ Regular exercise
- 🚭 Avoid smoking
- 😌 Manage stress
- 💊 Take medications as prescribed
""")

st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

if st.button("🔍 Analyze Heart Disease Risk", type="primary", use_container_width=True):
    # Build input row
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    # Reorder columns to match training order if available
    if hasattr(model, "feature_names_in_"):
        sample = sample.reindex(columns=list(model.feature_names_in_), fill_value=0)

    try:
        pred = int(model.predict(sample)[0])
        try:
            proba = model.predict_proba(sample)[0]
            risk_prob = float(proba[1])  # assume class 1 = disease
        except Exception:
            risk_prob = 0.85 if pred == 1 else 0.15

        if pred == 1:
            st.markdown("""
            <div class="prediction-positive">
                🚨 HIGH RISK: Heart Disease Detected<br>
                <small>Please consult a cardiologist immediately</small>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ This is a screening tool only. Seek medical attention for diagnosis and treatment.")
        else:
            st.markdown("""
            <div class="prediction-negative">
                ✅ LOW RISK: No Heart Disease Detected<br>
                <small>Continue maintaining a healthy lifestyle</small>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ Low risk detected. Continue regular check-ups and healthy habits.")

        st.markdown(f"**Estimated risk probability:** {risk_prob:.1%}")

        # -----------------------------
        # SHAP Explanation (robust fix)
        # -----------------------------
        st.markdown('<h3 class="section-header">🔍 Explanation of Prediction</h3>', unsafe_allow_html=True)

        if not SHAP_AVAILABLE:
            st.info("ℹ️ SHAP is not installed. Install to enable explanations: `pip install shap`")
        else:
            try:
                background = create_synthetic_background(n=128)

                # Prefer TreeExplainer; fallback to generic Explainer
                explainer = None
                try:
                    explainer = shap.TreeExplainer(model, background)
                except Exception:
                    explainer = shap.Explainer(model, background)

                shap_1d = unify_shap_values(explainer, sample)

                contrib_df = build_contrib_df(sample, shap_1d)

                # Human-readable explanation
                st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                render_factor_list(pred, risk_prob, contrib_df)
                st.markdown('</div>', unsafe_allow_html=True)

                # Plot top contributors
                plot_top_factors(contrib_df, top_n=8)

                # Optional table for transparency
                st.markdown("#### Detailed Contributions")
                pretty = contrib_df.assign(
                    Feature=lambda d: d["feature"].map(lambda f: FEATURE_DESCRIPTIONS.get(f, f)),
                    Value=lambda d: [describe_value(f, v) for f, v in zip(d["feature"], d["value"])],
                    Impact=lambda d: d["shap"].round(4),
                )[["Feature", "Value", "Impact"]]
                st.dataframe(pretty, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"⚠️ Could not generate detailed explanation: {e}")
                st.info("The prediction was successful. If this persists, ensure dependencies match requirements.txt.")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
  <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.</p>
  <p>Always consult with qualified healthcare providers regarding medical conditions.</p>
</div>
""", unsafe_allow_html=True)
