"""
Baseline Model Module.

Provides a Zero-Rule (ZeroR) classifier that always predicts the majority 
class. Used to establish a performance floor and justify the ROI of 
more complex models like XGBoost.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class BaselineModel(BaseEstimator, ClassifierMixin):
    """
    A Dummy Classifier that always predicts the most frequent class.
    
    Serves as the 'constant' baseline to compare against predictive 
    machine learning models.
    """

    def __init__(self):
        """Initializes the baseline model parameters."""
        self.majority_class_ = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Identifies the majority class from the target vector.

        Args:
            X (pd.DataFrame): Training features (unused).
            y (pd.Series): Target labels.

        Returns:
            self: The trained instance.
        """
        y_series = pd.Series(y)
        self.majority_class_ = y_series.mode()[0]
        self.classes_ = np.unique(y)

        print(f"[Baseline] Trained. Strategy: Always predict '{self.majority_class_}'")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates constant predictions based on the majority class.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Array of constant predictions.
        """
        if self.majority_class_ is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        return np.full(shape=(X.shape[0],), fill_value=self.majority_class_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability 1.0 for the majority class. Required for AUC metrics.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Probability estimates.
        """
        probs = np.zeros((X.shape[0], len(self.classes_)))
        major_idx = np.where(self.classes_ == self.majority_class_)[0][0]
        probs[:, major_idx] = 1.0
        return probs

    def evaluate_business_impact(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Quantifies the financial/operational risk of using a non-IA approach.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Actual labels.

        Returns:
            float: Nominal accuracy.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Impact Calculation: How many churners did the 'guess' ignore?
        total_churners = (y_test == 1).sum()
        # The BaselineModel misses 100% of churners if the majority is 0 (No Churn)
        missed_churners = total_churners 

        print("-" * 50)
        print("BASELINE MODEL RESULT (THE 'GUESS' APPROACH):")
        print(f"Nominal Accuracy: {acc:.2%}")
        print(f"Alert: This approach ignored {missed_churners} actual churners.")
        print("Conclusion: Zero investment in AI results in 100% loss of detectable churn.")
        print("-" * 50)

        return acc
