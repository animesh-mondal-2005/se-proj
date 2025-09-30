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

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variable to store the model
model = None

def load_model():
    """Load the machine learning model from disk"""
    global model
    try:
        model = joblib.load("heart_predictor_rf.joblib")
        print("✅ Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("❌ Error: Model file 'heart_predictor_rf.joblib' not found.")
        print("Please ensure the model file is in the same directory as app.py")
        model = None
        return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model = None
        return False

# Load model when app starts
load_model()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts POST request with JSON data containing patient information
    Returns prediction result
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure heart_predictor_rf.joblib is in the directory.',
            'success': False
        }), 500
    
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate that all required fields are present
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Create DataFrame with the input data in the correct order
        sample = pd.DataFrame([{
            "age": int(data['age']),
            "sex": int(data['sex']),
            "cp": int(data['cp']),
            "trestbps": int(data['trestbps']),
            "chol": int(data['chol']),
            "fbs": int(data['fbs']),
            "restecg": int(data['restecg']),
            "thalach": int(data['thalach']),
            "exang": int(data['exang']),
            "oldpeak": float(data['oldpeak']),
            "slope": int(data['slope']),
            "ca": int(data['ca']),
            "thal": int(data['thal'])
        }])
        
        # Make prediction
        prediction = model.predict(sample)[0]
        prediction = int(prediction)
        
        # Try to get probability if model supports it
        probability = None
        try:
            prob = model.predict_proba(sample)[0]
            probability = {
                'no_disease': float(prob[0]),
                'disease': float(prob[1])
            }
        except AttributeError:
            # Model doesn't support predict_proba
            pass
        
        # Prepare response
        response = {
            'prediction': prediction,
            'risk_level': 'high' if prediction == 1 else 'low',
            'success': True
        }
        
        if probability:
            response['probability'] = probability
            response['confidence'] = float(max(prob[0], prob[1]))
        
        print(f"Prediction made: {prediction} (Risk: {response['risk_level']})")
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid data type: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify API and model status"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'api_version': '1.0'
    }), 200

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Endpoint to reload the model without restarting the server"""
    success = load_model()
    if success:
        return jsonify({
            'message': 'Model reloaded successfully',
            'success': True
        }), 200
    else:
        return jsonify({
            'message': 'Failed to reload model',
            'success': False
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting Heart Disease Predictor API")
    print("=" * 50)
    
    if model is None:
        print("\n⚠️  WARNING: Model not loaded!")
        print("The application will start but predictions will fail.")
        print("Please ensure 'heart_predictor_rf.joblib' is in the directory.\n")
    
    print("📍 Server running at: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)