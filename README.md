# ❤️ Heart Disease Predictor

**AI-powered web application for heart disease risk assessment**

</div>

---

## 📖 About

Heart Disease Predictor is a machine learning-powered web application that assesses cardiovascular disease risk based on medical parameters. Built with Flask and Random Forest classifier, it provides instant risk predictions through a modern, intuitive interface.

> ⚠️ **Disclaimer**: For educational purposes only. Not a substitute for professional medical advice.

---

## ✨ Features

- 🎨 **Modern Dark UI** - Professional interface with responsive design
- 🧠 **AI-Powered Predictions** - Random Forest model analyzing 13 cardiac indicators
- ⚡ **Real-Time Analysis** - Instant risk assessment with confidence scores
- 📊 **Comprehensive Evaluation** - Analyzes demographics, vitals, and advanced cardiac parameters
- 🚀 **RESTful API** - Clean endpoints with JSON responses
- 💊 **Health Recommendations** - Actionable tips based on risk level
- 📱 **Mobile Responsive** - Seamless experience across all devices

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask 3.0.0** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **Joblib** - Model serialization
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with dark theme
- **JavaScript (ES6+)** - Dynamic interactions
- **Fetch API** - Asynchronous requests

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Features**: 13 cardiac indicators
- **Output**: Binary classification (Disease/No Disease)

---

## 📊 Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `age` | int | 18-100 | Patient's age in years |
| `sex` | int | 0-1 | Gender (0: Female, 1: Male) |
| `cp` | int | 0-3 | Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic) |
| `trestbps` | int | 80-200 | Resting blood pressure (mmHg) |
| `chol` | int | 100-600 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0-1 | Fasting blood sugar > 120 mg/dl (0: No, 1: Yes) |
| `restecg` | int | 0-2 | Resting ECG results (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy) |
| `thalach` | int | 60-220 | Maximum heart rate achieved |
| `exang` | int | 0-1 | Exercise induced angina (0: No, 1: Yes) |
| `oldpeak` | float | 0-10 | ST depression induced by exercise |
| `slope` | int | 0-2 | Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping) |
| `ca` | int | 0-3 | Number of major vessels colored by fluoroscopy |
| `thal` | int | 0-3 | Thalassemia (0: Unknown, 1: Fixed Defect, 2: Normal Flow, 3: Reversible Defect) |

---

## 🎯 Use Cases

- 🏥 **Medical Screening** - Preliminary cardiovascular risk assessment in healthcare settings
- 📚 **Educational Tool** - Teaching students about heart disease risk factors and ML applications
- 🔬 **Research** - Studying cardiac risk prediction models and feature importance
- 💡 **Health Awareness** - Public health campaigns for early disease detection
- 📊 **Data Analysis** - Understanding correlations between medical parameters and heart disease

---

</div>
