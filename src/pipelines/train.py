"""
End-to-end Training Pipeline Orchestrator.
Integrates MLflow tracking, hyperparameter tuning, and model persistence.
"""

import logging
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, log_loss
)
from mlflow.models.signature import infer_signature
from src.utils.evaluation import ModelEvaluator
from src.config.loader import ConfigLoader
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.validation import validate_dataframe
from src.models.xgboost import ChurnXGBoost
from src.models.baseline import BaselineModel
from src.models.training.tuner import ChurnTuner 

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        """Initializes the pipeline with configuration, data loader, and feature engineer."""
        self.cfg = ConfigLoader().load_all()
        self.dl = DataLoader(self.cfg)
        self.fe = FeatureEngineer(self.cfg)
        
        # Hardcoded IP for Docker internal network stability
        # Note: Ensure this IP matches your 'churn_mlflow' container IP
        uri = os.getenv("MLFLOW_TRACKING_URI")

        if not uri:
            raise RuntimeError("MLFLOW_TRACKING_URI não definido")

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.cfg["project"]["name"])

    def prepare_data(self):
        """
        Loads, validates, and transforms raw data while strictly following the YAML schema.
        Drops leaky features and IDs specified in 'excluded' section of the config.
        """
        # 1. Load and Validate Raw Data
        df_raw = self.dl.load_raw_data() 
        if not validate_dataframe(df_raw):
            raise ValueError("Data validation failed: Check raw data schema.")

        # 2. Feature Enrichment
        # Creates technical features (e.g., spend ratios, activity counts)
        df_enriched = self.fe.create_features(df_raw)
        
        # 3. Save Processed Data for Dashboard
        # Keeps all columns (including IDs) for visualization purposes
        self.dl.save_processed_data(df_enriched)

        # 4. Feature Selection based on YAML Schema
        feature_cfg = self.cfg.get("feature_schema", {})
        target = feature_cfg.get("target", "Churned")
        yaml_excluded = feature_cfg.get("excluded", []) 

        # Blacklist: Target + specific ID variations + YAML exclusions (Leakage/IDs)
        cols_to_exclude = [target, 'Customer_ID', 'CustomerID'] + yaml_excluded
        existing_exclude = [c for c in cols_to_exclude if c in df_enriched.columns]

        # Final X (features) and y (target) split
        X = df_enriched.drop(columns=existing_exclude)
        y = df_enriched[target]
        
        # 5. Categorical Encoding
        cat_features = [c for c in feature_cfg.get("categorical", []) if c in X.columns]
        if cat_features:
            X = pd.get_dummies(X, columns=cat_features)
        
        # Final type check to prevent XGBoost string errors
        bad_cols = X.select_dtypes(include=['object']).columns.tolist()
        if bad_cols:
            raise ValueError(f"XGBoost cannot process 'object' types in: {bad_cols}. Check your YAML exclusions.")
        
        logger.info(f"Final training set shape: {X.shape}. Target: {target}")
        
        return train_test_split(
            X, y, 
            test_size=self.cfg["model_metadata"]["test_size"], 
            random_state=self.cfg["environment"]["random_seed"],
            stratify=y
        )

    def run(self, tune=False):

        X_train, X_test, y_train, y_test = self.prepare_data()
        feature_names = X_train.columns.tolist()

        with mlflow.start_run(run_name="XGBoost_Production_Training") as run:

            # ------------------------------------------------
            # 1. Baseline
            # ------------------------------------------------
            baseline = BaselineModel()
            baseline.fit(X_train, y_train)

            baseline_acc = accuracy_score(
                y_test,
                baseline.predict(X_test)
            )

            mlflow.log_metric("baseline_accuracy", baseline_acc)

            # ------------------------------------------------
            # 2. Model + Tuning
            # ------------------------------------------------
            model_wrapper = ChurnXGBoost(self.cfg)

            if tune:
                logger.info("Running hyperparameter optimization")

                tuner = ChurnTuner(X_train, y_train)

                best_params = tuner.optimize(n_trials=30)

                model_wrapper.model.set_params(**best_params)

                mlflow.log_params(best_params)

            # ------------------------------------------------
            # 3. Training
            # ------------------------------------------------
            model_wrapper.train(X_train, y_train)

            y_pred = model_wrapper.predict(X_test)
            probs = model_wrapper.predict_proba(X_test)

            # ------------------------------------------------
            # 4. Evaluation (SINGLE SOURCE OF TRUTH)
            # ------------------------------------------------
            evaluator = ModelEvaluator(
                artifacts_path=self.cfg["artifacts"]["model_dir"],
                figures_path="reports/figures",
                reports_path="reports"
            )

            metrics = evaluator.evaluate(
                y_test=y_test,
                y_pred=y_pred,
                probs=probs,
                model_name="xgb_v1"
            )

            # Log everything to MLflow
            mlflow.log_metrics(metrics)

            mlflow.log_artifacts("reports")
            mlflow.log_artifacts("reports/figures")

            logger.info(f"Evaluation metrics: {metrics}")

            # ------------------------------------------------
            # 5. Feature Importance
            # ------------------------------------------------
            importance_dir = Path("data/interim")
            importance_dir.mkdir(parents=True, exist_ok=True)

            importance_path = importance_dir / "feature_importance.csv"

            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model_wrapper.model.feature_importances_
            }).sort_values("importance", ascending=False)

            importance_df.to_csv(importance_path, index=False)

            mlflow.log_artifact(importance_path)

            # ------------------------------------------------
            # 6. Model Registry
            # ------------------------------------------------
            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(
                sk_model=model_wrapper.model,
                artifact_path="model",
                signature=signature,
                registered_model_name="ChurnXGBoost-Prod"
            )

            # ------------------------------------------------
            # 7. Local Persistence
            # ------------------------------------------------
            model_path = Path(self.cfg["artifacts"]["current_model"])

            model_path.parent.mkdir(parents=True, exist_ok=True)

            model_wrapper.save(str(model_path), feature_names)

            logger.info(f"Run finished: {run.info.run_id}")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(tune=True)