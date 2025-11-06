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

## ⚙️ Data Preprocessing

- Encoded categorical columns using **LabelEncoder**  
- Scaled numeric features with **StandardScaler**  
- Balanced data using **SMOTE** (Synthetic Minority Oversampling Technique)  
- Split into training and testing sets  
- Saved reusable artifacts (`model.pkl`, `scaler.pkl`, `encoders.pkl`)  

---

## 📈 Model Performance

| Model | Accuracy | F1-score | ROC-AUC |
|:------|:----------|:---------|:--------|
| **Random Forest** | 99.95% | 99.96% | 99.96% |
| **XGBoost** | 99.92% | 99.93% | 99.93% |

✅ Random Forest was chosen for deployment due to superior performance and interpretability.

## 🧰 Tech Stack

Language:	Python
ML Libraries:	Scikit-learn, XGBoost, imbalanced-learn
Data Processing:	pandas, numpy
Visualization:	Plotly, Matplotlib, Seaborn
Frontend:	Streamlit
Deployment:	AWS EC2 / Streamlit Cloud

## ⚡ How It Works

- 1️⃣ Upload CSV / Enter Transaction → via Streamlit UI
- 2️⃣ Data Processing → Encoding + Scaling + Validation
- 3️⃣ Model Prediction → Random Forest classifies as Fraud or Not Fraud
- 4️⃣ Visualization → Dashboard displays insights and fraud metrics

## 📊 Dashboard Features

- Fraud vs Legitimate Pie Chart

- Transaction Type vs Fraud Bar Chart

- Fraud Amount Distribution Histogram

- Summary Metrics: Total, Fraudulent, Legitimate, Avg. Fraud Amount

## 🧾 Setup Instructions

### 🧩 1. Clone the Repository
```bash
git clone https://github.com/Nelli-VivekReddy/Machine-Learning-Based-Anomaly-Detection-for-Fraud-Prevention.git
cd fraud_detection

- Create Virtual Environment:python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows

- Install Dependencies:
pip install -r requirements.txt

- Run Streamlit App:
streamlit run app/app.py


- Then open your browser at http://localhost:8501
```

## 💡 Future Enhancements

- Add model comparison toggle (Random Forest vs XGBoost)

- Feature importance visualization (SHAP)

- Alert system for suspicious transactions

- Dockerized deployment on AWS

- Database feedback loop for retraining

## 🧮 Project Structure

```text
fraud_detection/
│
├── app/
│   ├── app.py                # Streamlit dashboard (real-time + batch prediction)
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

