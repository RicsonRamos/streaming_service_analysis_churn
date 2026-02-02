"""
Hyperparameter Optimization Module.

Uses Optuna to find the optimal set of hyperparameters for the XGBoost model,
focusing on maximizing the ROC-AUC score while preventing overfitting.
"""

import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class ChurnTuner:
    """
    Handles the optimization logic for the XGBoost classifier.
    """

    def __init__(self, X_train, y_train):
        """
        Initializes the tuner with training data.

        Args:
            X_train: Feature matrix for training.
            y_train: Target vector for training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial):
        """
        Objective function for Optuna to minimize/maximize.
        
        Defines the search space for XGBoost hyperparameters.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "random_state": 42,
            "eval_metric": "logloss"
        }

        # Use Cross-Validation to ensure the parameters generalize well
        model = xgb.XGBClassifier(**params)
        
        # We use 3-fold CV for speed, 5-fold for more precision
        score = cross_val_score(
            model, self.X_train, self.y_train, 
            cv=3, scoring="roc_auc", n_jobs=-1
        ).mean()

        return score

    def optimize(self, n_trials=30):
        """
        Runs the Optuna study.

        Args:
            n_trials (int): Number of optimization iterations.

        Returns:
            dict: The best hyperparameters found.
        """
        logger.info(f"Starting Optuna study with {n_trials} trials...")
        
        # Create a study to maximize the objective (ROC-AUC)
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        logger.info(f"Optimization finished. Best AUC: {study.best_value:.4f}")
        return study.best_params
