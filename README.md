Machine Learning–Based Fraud Detection System
🔐 Real-Time Anomaly Detection for Financial Cybersecurity

A Streamlit-powered web application that uses machine learning to detect fraudulent financial transactions in real time.

🧠 Overview

This project demonstrates how machine learning can be used as a cybersecurity tool to detect fraudulent vs. legitimate transactions in financial systems.
It trains models on real-world transaction data to identify anomalous patterns indicative of cyber fraud, money laundering, or system abuse.

🚀 Features

✅ Two Operating Modes

📂 Batch Mode: Upload a CSV of transactions for instant fraud analysis

⚡ Real-Time Mode: Input transaction details and get instant prediction

✅ Interactive Dashboard

Pie chart of fraud vs legitimate transactions

Bar chart by transaction type

Amount distribution histogram

Fraud probability per transaction

✅ Machine Learning Pipeline

Preprocessing (scaling, encoding, SMOTE balancing)

Model training (Random Forest, XGBoost)

Evaluation (Accuracy, F1, ROC-AUC)

Model deployment via Streamlit

✅ Cybersecurity Integration

Detects anomaly-based financial frauds

Works as an early alert system for suspicious activities

🧩 Dataset

Source: Kaggle – Fraud Detection Dataset by Aman Ali Siddiqui

Size: ~150,000+ transactions

Column	Description	Example
step	Time step of the transaction	43
type	Transaction type (CASH_IN, TRANSFER, etc.)	TRANSFER
amount	Amount transferred	85000
nameOrig	Sender account ID	C12345
oldbalanceOrg	Sender balance before transaction	90000
newbalanceOrig	Sender balance after transaction	5000
nameDest	Receiver account ID	M23456
oldbalanceDest	Receiver balance before transaction	0
newbalanceDest	Receiver balance after transaction	0
isFlaggedFraud	Flagged by rules (0/1)	0
isFraud	True label (1 = fraud, 0 = legitimate)	1
🧠 How It Works

Data Preprocessing

Encodes categorical columns (type, nameOrig, nameDest)

Scales numeric features

Handles class imbalance using SMOTE

Model Training

Trains both Random Forest and XGBoost models

Evaluates using metrics like F1, ROC-AUC, and Confusion Matrix

Model Deployment

Saves trained models as .pkl files (using joblib)

Streamlit app loads the model for interactive predictions

🧮 Model Performance
Model	Accuracy	F1-score	ROC-AUC
Random Forest	99.95%	99.96%	99.96%
XGBoost	99.92%	99.93%	99.93%

✅ Random Forest chosen as the final model for deployment (best overall balance of accuracy and interpretability).

🧱 Project Structure
## 🧮 Project Structure

```text
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


⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/fraud-detection-ml-app.git
cd fraud-detection-ml-app

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app/app.py

🧠 Future Enhancements

Then open your browser at 👉 http://localhost:8501

🧩 Example Prediction Flow

Upload a CSV with transaction data (no isFraud column)

Model preprocesses and predicts fraud probabilities

Dashboard displays:

Fraud vs Legit count

Average fraud amount

Distribution charts

Downloadable CSV with predictions

🧠 Key Learnings

Handling highly imbalanced data using SMOTE

Building modular ML pipelines (train → save → deploy)

Deploying interactive ML dashboards with Streamlit

Using ML for cybersecurity anomaly detection

💡 Future Enhancements

🚀 Add model comparison toggle (Random Forest vs XGBoost in UI)
📈 Feature importance visualizations (SHAP)
📬 Email/SMS alert system for high fraud risk
☁️ Docker / AWS deployment
🗃️ Database logging for user feedback and retraining