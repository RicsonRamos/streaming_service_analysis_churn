"""
End-to-end Training Pipeline Orchestrator.
Integrates MLflow tracking, hyperparameter tuning, and model persistence.
"""

import logging
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pathlib import Path

from src.config.loader import ConfigLoader
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.validation import validate_dataframe
from src.models.xgboost import ChurnXGBoost
from src.models.baseline import BaselineModel
from src.models.training.tuner import ChurnTuner 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.cfg = ConfigLoader().load_all()
        self.dl = DataLoader(self.cfg)
        self.fe = FeatureEngineer(self.cfg)
        mlflow.set_experiment(self.cfg["project"]["name"])

    def prepare_data(self):
        """Loads, validates, and transforms raw data while preserving IDs for the dashboard."""
        # 1. Load (Agora com ID gerado se faltar)
        df_raw = self.dl.load_raw_data() 
        if not validate_dataframe(df_raw):
            raise ValueError("Data validation failed.")

        # 2. Enriquecimento (Cria as features técnicas)
        df_enriched = self.fe.create_features(df_raw)
        
        # --- PONTO CRÍTICO: SALVAMENTO PARA O DASHBOARD ---
        # Salvamos o dataset completo (com ID, Churn e Features Brutas)
        # Isso garante que o main_dash.py encontre a coluna 'Customer_ID'
        self.dl.save_processed_data(df_enriched)
        # --------------------------------------------------

        target = self.cfg["feature_schema"]["target"]


        cols_to_exclude = [target, 'Customer_ID']

        existing_exclude = [c for c in cols_to_exclude if c in df_enriched.columns]

        X = df_enriched.drop(columns=existing_exclude)
        
        # Encoding das categóricas (Gender, Region, etc.)
        X = pd.get_dummies(X, columns=self.cfg["feature_schema"]["categorical"])
        y = df_enriched[target]
        
        return train_test_split(
            X, y, 
            test_size=self.cfg["model_metadata"]["test_size"], 
            random_state=self.cfg["environment"]["random_seed"],
            stratify=y
        )

    def run(self, tune=False):
        """Executes the full training and logging cycle."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        feature_names = X_train.columns.tolist()

        with mlflow.start_run(run_name="XGBoost_Production_Training"):
            # 1. Baseline ROI Check
            baseline = BaselineModel()
            baseline.fit(X_train, y_train)
            mlflow.log_metric("baseline_acc", accuracy_score(y_test, baseline.predict(X_test)))

            # 2. Hyperparameter Tuning (Optional)
            model_wrapper = ChurnXGBoost(self.cfg)
            if tune:
                logger.info("Starting Hyperparameter Optimization...")
                tuner = ChurnTuner(X_train, y_train)
                best_params = tuner.optimize(n_trials=30)
                model_wrapper.model.set_params(**best_params)
                mlflow.log_params(best_params)

            # 3. Final Training
            model_wrapper.train(X_train, y_train)

            # 4. Evaluation
            probs = model_wrapper.predict_proba(X_test)
            auc = roc_auc_score(y_test, probs)
            acc = accuracy_score(y_test, model_wrapper.predict(X_test))
            
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", acc)
            logger.info(f"Training Complete. AUC: {auc:.4f} | ACC: {acc:.4f}")

            model_full_path = Path(self.cfg["artifacts"]["current_model"]).resolve()
            model_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to: {model_full_path}")
            
            
            model_wrapper.save(str(model_full_path), feature_names)
            mlflow.sklearn.log_model(model_wrapper.model, "churn_xgboost_model")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run(tune=True)
