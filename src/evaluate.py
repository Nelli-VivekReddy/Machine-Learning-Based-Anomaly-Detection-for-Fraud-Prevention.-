from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, preds))

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}
