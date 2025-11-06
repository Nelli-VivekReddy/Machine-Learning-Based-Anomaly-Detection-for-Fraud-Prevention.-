import os
import joblib

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def save_preprocessing(scaler, encoders, base_dir="model"):
    os.makedirs(base_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(base_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(base_dir, "encoder.pkl"))
    print(f"✅ Scaler and encoders saved to {base_dir}/")

