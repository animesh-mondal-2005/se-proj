from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

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
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure heart_predictor_rf.joblib is in the directory.',
            'success': False
        }), 500
    
    try:
        data = request.json
        
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
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
        
        prediction = model.predict(sample)[0]
        prediction = int(prediction)
        
        probability = None
        try:
            prob = model.predict_proba(sample)[0]
            probability = {
                'no_disease': float(prob[0]),
                'disease': float(prob[1])
            }
        except AttributeError:
            pass
        
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