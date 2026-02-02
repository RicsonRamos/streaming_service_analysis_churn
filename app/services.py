"""
ChurnService: Core logic for data loading, inference, and risk classification.
"""
import pandas as pd
import joblib
import os

class ChurnService:
    def __init__(self, model_path: str, processed_path: str):
        self.model_path = model_path
        self.processed_path = processed_path
        # Strict alignment with the 17 features expected by the trained XGBoost model
        self.expected_features = [
            'Age', 'Subscription_Length', 'Support_Tickets_Raised', 'Satisfaction_Score', 
            'Last_Activity', 'Monthly_Spend', 'Estimated_LTV', 'Engagement_Score',
            'Gender_Female', 'Gender_Male', 
            'Region_Central', 'Region_East', 'Region_North', 'Region_South', 'Region_West',
            'Payment_Method_Credit Card', 'Payment_Method_PayPal'
        ]

    def load_assets(self):
        """Loads model artifacts and the processed historical dataset."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.processed_path):
            return None, None

        loaded = joblib.load(self.model_path)
        # Handle both direct model files and dictionary artifacts
        model = loaded.get('model') if isinstance(loaded, dict) else loaded
        df = pd.read_csv(self.processed_path)
        return model, df

    def predict_churn(self, model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Performs batch inference on the dataset and classifies risk levels.
        """
        X = df.copy()
        X = pd.get_dummies(X, columns=['Gender', 'Region', 'Payment_Method'])

        # Force column alignment to prevent shape mismatch (17 feature requirement)
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = 0

        X = X[self.expected_features]

        # Inference
        probabilities = model.predict_proba(X)[:, 1]
        df['Probability'] = probabilities
        
        # English-standard risk classification
        df['Risk_Level'] = df['Probability'].apply(
            lambda x: 'High' if x >= threshold else ('Medium' if x >= 0.4 else 'Low')
        )
        return df

    def predict_single_customer(self, model, customer_data: dict) -> float:
        """
        Predicts churn probability for a single customer scenario (Simulator).
        """
        X_single = pd.DataFrame([customer_data])
        X_single = pd.get_dummies(X_single)

        # Force alignment with the 17-feature model input
        for col in self.expected_features:
            if col not in X_single.columns:
                X_single[col] = 0

        X_single = X_single[self.expected_features]
        
        # Return probability as a float
        return float(model.predict_proba(X_single)[0, 1])
