import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess_data, split_data
from evaluate import evaluate_model
from utils import save_model, save_preprocessing
import joblib
import os


def train_model(X_train, y_train, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Unsupported model type.")
    
    print(f"Training {model_type} ...")
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="model")
    args = parser.parse_args()

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Load & preprocess
    df = load_data(args.data_path)

    # 🔹 Optional fast debug (sample subset)
    df = df.sample(n=100000, random_state=42)
    print(f"Using subset for quick training: {df.shape}")

    X, y, scaler, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train both models
    results = {}
    for model_name in ["random_forest", "xgboost"]:
        model = train_model(X_train, y_train, model_name)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        save_model(model, f"{args.model_dir}/{model_name}_model.pkl")

    # Save preprocessors
    save_preprocessing(scaler, encoders, args.model_dir)

    # 🔹 Ensure scaler and encoders are definitely saved
    joblib.dump(scaler, f"{args.model_dir}/scaler.pkl")
    joblib.dump(encoders, f"{args.model_dir}/encoders.pkl")

    print("\n✅ Training completed. Results:")
    for m, r in results.items():
        print(f"{m}: {r}")
