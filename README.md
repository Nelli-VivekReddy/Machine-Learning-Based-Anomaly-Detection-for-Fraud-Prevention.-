Machine Learning-Based Anomaly Detection for Fraud Prevention

A machine learning–powered system designed to detect fraudulent financial transactions in real time.
It serves as a cybersecurity measure by identifying anomalous behaviors and potential fraud in digital payment systems.

🚀 Overview

This project applies Machine Learning (ML) techniques to detect financial fraud from transactional data.
It uses Random Forest and XGBoost models to classify transactions as fraudulent or legitimate based on behavioral patterns.

🧩 Features

✅ Detects fraudulent vs. legitimate transactions

✅ Real-time prediction via Streamlit

✅ Supports both CSV batch and single-transaction prediction

✅ Interactive analytics dashboard (charts + metrics)

✅ Cybersecurity-aligned anomaly detection system

🧠 Dataset

Source: Kaggle – Fraud Detection Dataset by Aman Ali Siddiqui

Size: ~150,000+ transaction records

Column	Description
step	Time step of the transaction
type	Transaction type (CASH_IN, TRANSFER, CASH_OUT, PAYMENT, etc.)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Receiver account ID
oldbalanceDest	Receiver balance before transaction
newbalanceDest	Receiver balance after transaction
isFlaggedFraud	Flagged by rule-based system (0 or 1)
isFraud	Ground truth label (1 = Fraud, 0 = Legitimate)
⚙️ Data Preprocessing

Encoded categorical columns: type, nameOrig, nameDest

Scaled numerical features using StandardScaler

Balanced dataset using SMOTE (Synthetic Minority Oversampling Technique)

Split into training/testing datasets

Saved reusable artifacts (model.pkl, scaler.pkl, encoders.pkl)

📈 Model Performance
Model	Accuracy	F1-score	ROC-AUC
Random Forest	99.95%	99.96%	99.96%
XGBoost	99.92%	99.93%	99.93%

✅ Random Forest was chosen for deployment (best overall balance of accuracy and interpretability).

🧮 Project Structure
fraud_detection/
│
├── app/
│   ├── app.py                   # Streamlit dashboard (real-time + batch prediction)
│   └── __init__.py
│
├── data/
│   ├── fraud_data.csv
│   └── fraud_test_sample_1000.csv
│
├── model/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── src/
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── utils.py
│   └── train_all_models.py
│
├── create_test_sample.py
├── requirements.txt
└── README.md

🧰 Tech Stack
Category	Technology
Language	Python
ML Libraries	Scikit-learn, XGBoost, imbalanced-learn
Data Processing	pandas, numpy
Visualization	Plotly, Matplotlib, Seaborn
Frontend	Streamlit
Deployment	AWS EC2 / Streamlit Cloud
⚡ How It Works

1️⃣ Upload CSV / Enter Transaction → via Streamlit UI
2️⃣ Data Processing → Encoding + Scaling + Validation
3️⃣ Model Prediction → Random Forest classifies as Fraud or Not Fraud
4️⃣ Visualization → Dashboard displays analytics (charts + summary metrics)

📊 Dashboard Features

📈 Fraud vs Legitimate Pie Chart

📊 Transaction Type vs Fraud Bar Chart

💰 Fraud Amount Distribution Histogram

🧾 Metrics summary (Total, Fraudulent, Legitimate, Avg. Fraud Amount)

🧾 Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/Nelli-VivekReddy/Machine-Learning-Based-Anomaly-Detection-for-Fraud-Prevention.git
cd fraud_detection

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app/app.py

🧠 Future Enhancements

Integrate with live transaction APIs for real-time fraud streams

Deploy via Docker / AWS Lambda for production

Add Explainability (SHAP values)

Build alert and notification system for anomalies
