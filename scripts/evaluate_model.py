"""
Model Evaluation Script.

Loads the trained model and processed test data to generate performance 
benchmarks, including a classification report and a confusion matrix.
"""

import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config.loader import ConfigLoader

def run_evaluation():
    """
    Executes the full evaluation pipeline and saves artifacts to the reports directory.
    """
    # Initialize config loader
    cfg = ConfigLoader().load_all()

    # 1. Load Data and Model Artifacts
    # Using the new path taxonomy from paths.yaml
    test_data = pd.read_csv(cfg["data"]["final_dataset"])
    artifact = joblib.load(cfg["artifacts"]["current_model"])

    # Extract model and feature list from the stored dictionary
    model = artifact["model"]
    features = artifact["features"]
    target = cfg["feature_schema"]["target"]

    X_test = test_data[features]
    y_test = test_data[target]

    # 2. Model Inference
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    # 3. Generate Metrics Report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\n=== MODEL PERFORMANCE REPORT ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 4. Save Metrics to JSON (For CI/CD or Automated READMEs)
    metrics_path = cfg["outputs"]["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump({
            "roc_auc": roc_auc,
            "classification_report": report
        }, f, indent=4)
    print(f"\n[INFO] Metrics saved to: {metrics_path}")

    # 5. Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.title('Confusion Matrix - Churn Prediction Radar')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save visualization using path from config
    plot_path = cfg["outputs"]["confusion_matrix"]
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Confusion Matrix plot saved to: {plot_path}")

if __name__ == "__main__":
    run_evaluation()
