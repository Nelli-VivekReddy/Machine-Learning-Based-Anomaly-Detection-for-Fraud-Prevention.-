import pandas as pd
import os

# ===============================
# CONFIG
# ===============================
# Change this to your dataset path
file_path = "data/fraud_data.csv"  # or "data/AIML Dataset (1).csv"
output_path = "data/fraud_test_sample_1000.csv"

# ===============================
# LOAD DATA
# ===============================
print(f"📂 Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

if "isFraud" not in df.columns:
    raise ValueError("❌ The dataset does not contain 'isFraud' column!")

# ===============================
# SAMPLE FRAUD AND NON-FRAUD TRANSACTIONS
# ===============================
fraud_df = df[df["isFraud"] == 1]
non_fraud_df = df[df["isFraud"] == 0]

if len(fraud_df) < 500 or len(non_fraud_df) < 500:
    print(f"⚠️ Dataset has limited fraud cases. Sampling available count only.")
    fraud_n = min(500, len(fraud_df))
    non_fraud_n = min(500, len(non_fraud_df))
else:
    fraud_n, non_fraud_n = 500, 500

fraud_sample = fraud_df.sample(n=fraud_n, random_state=42, replace=True)
non_fraud_sample = non_fraud_df.sample(n=non_fraud_n, random_state=42)

# Combine and shuffle
test_sample = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# REMOVE LABEL COLUMN
# ===============================
if "isFraud" in test_sample.columns:
    test_sample = test_sample.drop(columns=["isFraud"])

# ===============================
# SAVE NEW TEST FILE
# ===============================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
test_sample.to_csv(output_path, index=False)

print(f"\n✅ Fraud test sample created successfully!")
print(f"📁 File saved at: {output_path}")
print(f"📊 Rows: {test_sample.shape[0]} | Columns: {test_sample.shape[1]}")
