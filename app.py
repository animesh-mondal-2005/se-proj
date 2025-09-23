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
from plotly.subplots import make_subplots

# Try to import SHAP, show a gentle notice if missing
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles stay consistent with the original design
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
    .factor-positive {
        color: #e74c3c;
        font-weight: bold;
    }
    .factor-negative {
        color: #27ae60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_predictor_rf.joblib")
    except FileNotFoundError:
        st.error("⚠️ Model file 'heart_predictor_rf.joblib' not found. Please ensure the model file is in the same directory.")
        st.stop()

model = load_model()

# Feature descriptions for better explanations
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
    "thal": "Thalassemia"
}

FEATURE_VALUE_DESCRIPTIONS = {
    "cp": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "restecg": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal": ["Unknown/Normal", "Fixed Defect", "Normal Flow", "Reversible Defect"],
    "sex": ["Female", "Male"],
    "fbs": ["No", "Yes"],
    "exang": ["No", "Yes"]
}

def get_feature_value_description(feature, value):
    """Get human-readable description of feature values"""
    if feature in FEATURE_VALUE_DESCRIPTIONS:
        options = FEATURE_VALUE_DESCRIPTIONS[feature]
        if isinstance(value, (int, float)) and 0 <= value < len(options):
            return options[int(value)]
    return str(value)

def generate_explanation(prediction, probability, shap_values, feature_values, feature_names):
    """Generate human-readable explanation based on SHAP values"""
    
    # Create DataFrame with feature contributions
    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'value': feature_values,
        'shap': shap_values
    })
    
    # Sort by absolute SHAP value (most influential first)
    contrib_df['abs_shap'] = np.abs(contrib_df['shap'])
    contrib_df = contrib_df.sort_values('abs_shap', ascending=False)
    
    # Separate increasing and decreasing factors
    increasing_factors = contrib_df[contrib_df['shap'] > 0]
    decreasing_factors = contrib_df[contrib_df['shap'] < 0]
    
    explanation_parts = []
    
    if prediction == 1:  # High risk prediction
        explanation_parts.append(f"## 🔍 Why Heart Disease Was Predicted (Risk: {probability:.1%})")
        explanation_parts.append("The model predicted **high risk of heart disease** primarily due to these factors:")
        
        if not increasing_factors.empty:
            for _, factor in increasing_factors.head(5).iterrows():
                feature_desc = FEATURE_DESCRIPTIONS.get(factor['feature'], factor['feature'])
                value_desc = get_feature_value_description(factor['feature'], factor['value'])
                explanation_parts.append(
                    f"- **{feature_desc}** = {value_desc} "
                    f"(<span class='factor-positive'>increased risk by {factor['shap']:.3f}</span>)"
                )
        else:
            explanation_parts.append("- No strong risk-increasing factors were identified.")
            
        if not decreasing_factors.empty:
            explanation_parts.append("")
            explanation_parts.append("**Factors that reduced risk (but weren't enough):**")
            for _, factor in decreasing_factors.head(3).iterrows():
                feature_desc = FEATURE_DESCRIPTIONS.get(factor['feature'], factor['feature'])
                value_desc = get_feature_value_description(factor['feature'], factor['value'])
                explanation_parts.append(
                    f"- **{feature_desc}** = {value_desc} "
                    f"(<span class='factor-negative'>decreased risk by {abs(factor['shap']):.3f}</span>)"
                )
    
    else:  # Low risk prediction
        explanation_parts.append(f"## 🔍 Why No Heart Disease Was Predicted (Risk: {probability:.1%})")
        explanation_parts.append("The model predicted **low risk of heart disease** primarily due to these factors:")
        
        if not decreasing_factors.empty:
            for _, factor in decreasing_factors.head(5).iterrows():
                feature_desc = FEATURE_DESCRIPTIONS.get(factor['feature'], factor['feature'])
                value_desc = get_feature_value_description(factor['feature'], factor['value'])
                explanation_parts.append(
                    f"- **{feature_desc}** = {value_desc} "
                    f"(<span class='factor-negative'>decreased risk by {abs(factor['shap']):.3f}</span>)"
                )
        else:
            explanation_parts.append("- No strong risk-decreasing factors were identified.")
            
        if not increasing_factors.empty:
            explanation_parts.append("")
            explanation_parts.append("**Factors that increased risk (but were outweighed):**")
            for _, factor in increasing_factors.head(3).iterrows():
                feature_desc = FEATURE_DESCRIPTIONS.get(factor['feature'], factor['feature'])
                value_desc = get_feature_value_description(factor['feature'], factor['value'])
                explanation_parts.append(
                    f"- **{feature_desc}** = {value_desc} "
                    f"(<span class='factor-positive'>increased risk by {factor['shap']:.3f}</span>)"
                )
    
    # Add key influential factors summary
    if not contrib_df.empty:
        top_factor = contrib_df.iloc[0]
        feature_desc = FEATURE_DESCRIPTIONS.get(top_factor['feature'], top_factor['feature'])
        value_desc = get_feature_value_description(top_factor['feature'], top_factor['value'])
        
        explanation_parts.append("")
        explanation_parts.append("### 💡 Key Insight:")
        if top_factor['shap'] > 0:
            explanation_parts.append(
                f"The most influential factor was **{feature_desc}** ({value_desc}), which significantly increased the risk prediction."
            )
        else:
            explanation_parts.append(
                f"The most influential factor was **{feature_desc}** ({value_desc}), which significantly decreased the risk prediction."
            )
    
    return "\n\n".join(explanation_parts)

def create_synthetic_background(n=100):
    """Create a simple synthetic background dataset for SHAP"""
    return pd.DataFrame({
        "age": np.random.randint(30, 80, n),
        "sex": np.random.randint(0, 2, n),
        "cp": np.random.randint(0, 4, n),
        "trestbps": np.random.randint(90, 180, n),
        "chol": np.random.randint(150, 400, n),
        "fbs": np.random.randint(0, 2, n),
        "restecg": np.random.randint(0, 3, n),
        "thalach": np.random.randint(100, 180, n),
        "exang": np.random.randint(0, 2, n),
        "oldpeak": np.random.uniform(0, 4, n),
        "slope": np.random.randint(0, 3, n),
        "ca": np.random.randint(0, 4, n),
        "thal": np.random.randint(0, 4, n),
    })

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
        age = st.number_input("👤 Age", min_value=18, max_value=100, value=50, help="Patient's age in years")
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
        slope = st.selectbox("📊 ST Slope", options=[0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
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
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    try:
        pred = model.predict(sample)[0]
        prob = None
        try:
            proba = model.predict_proba(sample)[0]
            # assume class 1 is "disease"
            prob = float(proba[1])
        except Exception:
            # If predict_proba fails, use a default probability based on prediction
            prob = 0.85 if pred == 1 else 0.15

        if pred == 1:
            st.markdown("""
            <div class="prediction-positive">
                🚨 HIGH RISK: Heart Disease Detected<br>
                <small>Please consult a cardiologist immediately</small>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ **Important:** This is a screening tool only. Please seek immediate medical attention for proper diagnosis and treatment.")
        else:
            st.markdown("""
            <div class="prediction-negative">
                ✅ LOW RISK: No Heart Disease Detected<br>
                <small>Continue maintaining a healthy lifestyle</small>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ **Good News:** Low risk detected. Continue regular check-ups and maintain a healthy lifestyle.")

        st.markdown(f"**Estimated risk probability:** {prob:.1%}")

        # =========================
        # SHAP EXPLANATION SECTION (SIMPLIFIED)
        # =========================
        st.markdown('<h3 class="section-header">🔍 Explanation of Prediction</h3>', unsafe_allow_html=True)

        if not SHAP_AVAILABLE:
            st.info("ℹ️ SHAP is not installed. To see detailed explanations, install it: `pip install shap`")
            
            # Fallback simple explanation without SHAP
            st.markdown("""
            <div class="explanation-box">
            <h4>Why this prediction?</h4>
            <p>With SHAP installed, you would see a detailed breakdown of which factors contributed most to this prediction, 
            showing which features increased or decreased the risk of heart disease.</p>
            <p><em>Install SHAP to see feature importance analysis.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                # Create a simple background dataset for SHAP
                background_data = create_synthetic_background(n=100)
                
                # Create SHAP explainer (without caching to avoid the error)
                explainer = shap.TreeExplainer(model, background_data)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(sample)
                
                # Handle different SHAP return formats
                if isinstance(shap_values, list):
                    # For classifiers, get values for positive class (index 1)
                    shap_vals = shap_values[1][0]
                else:
                    shap_vals = shap_values[0]
                
                # Generate and display explanation
                explanation = generate_explanation(
                    prediction=pred,
                    probability=prob,
                    shap_values=shap_vals,
                    feature_values=sample.iloc[0].values,
                    feature_names=sample.columns
                )
                
                st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                st.markdown(explanation, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Simple visualization of top factors
                contrib_df = pd.DataFrame({
                    'Feature': sample.columns,
                    'Impact on Risk': shap_vals
                })
                contrib_df['Absolute Impact'] = np.abs(contrib_df['Impact on Risk'])
                contrib_df = contrib_df.sort_values('Absolute Impact', ascending=False).head(8)
                
                # Create a simple bar chart
                fig = go.Figure()
                colors = ['#e74c3c' if x > 0 else '#27ae60' for x in contrib_df['Impact on Risk']]
                
                fig.add_trace(go.Bar(
                    x=contrib_df['Impact on Risk'],
                    y=contrib_df['Feature'],
                    orientation='h',
                    marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Top Factors Influencing Prediction",
                    xaxis_title="Impact on Risk (SHAP Value)",
                    yaxis_title="Features",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate detailed explanation: {str(e)}")
                st.info("The prediction was still made successfully. The explanation feature requires additional dependencies.")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.</p>
    <p>Always consult with qualified healthcare providers regarding medical conditions.</p>
</div>
""", unsafe_allow_html=True)
