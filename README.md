# 💳 Machine Learning-Based Anomaly Detection for Fraud Prevention

A machine learning–powered system designed to detect **fraudulent financial transactions** in real time.  
It acts as a **cybersecurity tool** by identifying **anomalous behaviors** and potential fraud in digital payment systems.

---

## 🚀 Overview

This project uses **Machine Learning (ML)** techniques to classify transactions as *fraudulent* or *legitimate*.  
It leverages **Random Forest** and **XGBoost** models trained on financial data, and integrates with a Streamlit dashboard for real-time predictions.

---

## 🧩 Features

- ✅ Detects fraudulent vs. legitimate transactions  
- ✅ Real-time prediction through a Streamlit web app  
- ✅ Two modes: CSV batch upload and single transaction prediction  
- ✅ Interactive dashboard with fraud insights  
- ✅ Designed for financial cybersecurity and fraud prevention  

---

## 🧠 Dataset

**Source:** [Kaggle – Fraud Detection Dataset by Aman Ali Siddiqui](https://www.kaggle.com/datasets/amanalids/fraud-detection)  
**Size:** ~150,000+ transaction records  

| Column | Description |
|:--------|:------------|
| `step` | Time step of the transaction |
| `type` | Transaction type (`CASH_IN`, `TRANSFER`, `CASH_OUT`, `PAYMENT`, etc.) |
| `amount` | Transaction amount |
| `nameOrig` | Sender account ID |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account ID |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFlaggedFraud` | Flagged by rule-based system (0/1) |
| `isFraud` | Ground truth label (1 = Fraud, 0 = Legitimate) |

---

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
