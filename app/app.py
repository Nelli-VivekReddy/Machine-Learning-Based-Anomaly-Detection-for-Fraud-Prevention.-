import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ===============================
# Load Model and Preprocessing Objects
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/random_forest_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# ===============================
# Page Config and Styling
# ===============================
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0e1117; color: #fafafa; }
        .block-container { padding: 2rem; }
        h1, h2, h3 { color: #FFD700; }
        .metric-card {
            background: linear-gradient(145deg, #202020, #181818);
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
            text-align: center;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.2);
        }
        .section {
            background-color: #1a1a1a;
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #777;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Header Section
# ===============================
st.markdown("<h1 style='text-align:center;'>💳 AI-Powered Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Detect and visualize fraudulent transactions in real-time or batch mode.</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Select Mode")
mode = st.sidebar.radio("", ["📂 Batch Prediction (Upload CSV)", "⚡ Real-Time Transaction Check"])

# ===============================
# 1️⃣ Batch Prediction Mode
# ===============================
if mode == "📂 Batch Prediction (Upload CSV)":
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Transaction Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        st.cache_data.clear()
        df = pd.read_csv(uploaded_file)

        # Drop target columns if any
        drop_cols = ["isFraud", "is_flagged_fraud", "isFlaggedFraud"]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        st.success(f"✅ File uploaded successfully — {df.shape[0]} transactions loaded.")
        st.dataframe(df.head(), use_container_width=True)

        # --- Preprocess for Model ---
        X = df.copy()
        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else np.nan)
                encoder_classes = np.append(encoder.classes_, np.nan)
                encoder.classes_ = encoder_classes
                X[col] = encoder.transform(X[col].fillna(np.nan))

        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[:, 1] * 100

        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({0: "Not Fraud", 1: "Fraud"})
        df["Fraud Probability (%)"] = proba.round(2)

        # --- Metrics ---
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("📊 Fraud Insights Summary")

        total_tx = df.shape[0]
        fraud_tx = (df["Prediction"] == "Fraud").sum()
        legit_tx = total_tx - fraud_tx
        fraud_percent = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
        avg_fraud_amt = df.loc[df["Prediction"] == "Fraud", "amount"].mean() if "amount" in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Total Transactions</h3><h2>{total_tx:,}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Fraudulent</h3><h2 style='color:#FF4B4B'>{fraud_tx:,}</h2><p>{fraud_percent:.2f}%</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Legitimate</h3><h2 style='color:#00CC96'>{legit_tx:,}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>Avg Fraud Amount</h3><h2>₹{0 if np.isnan(avg_fraud_amt) else avg_fraud_amt:,.2f}</h2></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Visualizations ---
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("📈 Transaction Insights")

        if "Prediction" in df.columns:
            pie_data = df["Prediction"].value_counts().reset_index()
            pie_data.columns = ["Category", "Count"]
            fig_pie = px.pie(pie_data, values="Count", names="Category",
                             title="Fraud vs Legitimate Transactions",
                             color_discrete_sequence=["#FF4B4B", "#00CC96"])
            st.plotly_chart(fig_pie, use_container_width=True)

        if "type" in df.columns:
            fraud_type = df.groupby(["type", "Prediction"]).size().reset_index(name="Count")
            fig_bar = px.bar(fraud_type, x="type", y="Count", color="Prediction",
                             title="Transaction Type vs Fraud Count",
                             color_discrete_sequence=["#FF4B4B", "#00CC96"])
            st.plotly_chart(fig_bar, use_container_width=True)

        if "amount" in df.columns:
            fig_hist = px.histogram(df, x="amount", color="Prediction", nbins=30,
                                    title="Transaction Amount Distribution",
                                    color_discrete_sequence=["#FF4B4B", "#00CC96"])
            st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Results Table ---
        st.subheader("🔍 Detailed Predictions")
        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions as CSV", csv, "fraud_detection_results.csv", "text/csv")

    else:
        st.info("👆 Upload a CSV file to start analysis.")

# ===============================
# 2️⃣ Real-Time Transaction Mode
# ===============================
else:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("⚡ Real-Time Transaction Prediction")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            type_ = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT"])
            amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
            oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
            newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
        with c2:
            oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
            newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=1000.0)
            isFlaggedFraud = st.selectbox("Is Flagged by Rules?", [0, 1])

        submitted = st.form_submit_button("🔍 Predict Fraud")

    if submitted:
        input_data = pd.DataFrame([{
            "step": 1,
            "type": type_,
            "amount": amount,
            "nameOrig": "C0000",
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "nameDest": "M0000",
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud
        }])

        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: x if x in encoder.classes_ else np.nan)
                encoder_classes = np.append(encoder.classes_, np.nan)
                encoder.classes_ = encoder_classes
                input_data[col] = encoder.transform(input_data[col].fillna(np.nan))

        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_cols]

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        st.markdown("---")
        if pred == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob:.2f}%)")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {100 - prob:.2f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("<div class='footer'>© 2025 Fraud Detection Dashboard | Developed by Vivek Reddy</div>", unsafe_allow_html=True)
