import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df, target_col='isFraud'):
    df = df.copy()
    df = df.dropna().reset_index(drop=True)

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE: {X_res.shape}, {y_res.value_counts().to_dict()}")

    return X_res, y_res, scaler, encoders

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
