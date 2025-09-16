# ❤️ Heart Disease Predictor

An **AI-powered web application** built with **Streamlit** that predicts the risk of heart disease based on medical parameters such as age, cholesterol, blood pressure, chest pain type, and more.  

The system uses **machine learning models (Random Forest / XGBoost)** trained on the **UCI Heart Disease dataset** with **data balancing (SMOTE + Tomek)** and **feature engineering**.  

---

## 🚀 Features
- User-friendly **web interface** built with **Streamlit**.  
- Input patient details through dropdowns, number fields, or chatbot-like flow.  
- Predicts whether a patient is at **high risk** or **low risk** of heart disease.  
- Provides **lifestyle tips** and risk interpretation.  
- Secured **model training pipeline** with data preprocessing, normalization, and feature selection.  
- Model saved with **joblib** for reuse.  

---

## 🛠️ Technologies Used
- **Python 3.9+**  
- **Streamlit** (web app)  
- **Scikit-learn** (ML pipeline)  
- **XGBoost** / **RandomForestClassifier** (prediction model)  
- **Imbalanced-learn (SMOTE + Tomek)**  
- **Pandas, NumPy** (data handling)  
- **Matplotlib / Plotly** (visualizations)  
- **Joblib** (model persistence)  

---

## 📂 Project Structure
heart-disease-predictor/
│── app.py                  # Main Streamlit application
│── heart.csv               # Dataset (UCI Heart Disease Dataset)
│── heart_predictor.joblib  # Trained ML model
│── requirements.txt        # Dependencies
│── README.md               # Documentation
