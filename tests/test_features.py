import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer
from pytest import approx

@pytest.fixture
def mock_cfg():
    """Provides a controlled configuration without relying on YAML files."""
    return {
        "feature_schema": {
            "numeric": ["Age", "Subscription_Length", "Monthly_Spend", "Support_Tickets_Raised"],
            "categorical": ["Gender", "Region", "Payment_Method"]
        }
    }

@pytest.fixture
def sample_data():
    """Standard customer data for happy-path testing."""
    return pd.DataFrame({
        'Age': [25, 40],
        'Subscription_Length': [12, 1],
        'Monthly_Spend': [100.0, 50.0],
        'Support_Tickets_Raised': [2, 0],
        'Gender': ['Male', 'Female'],
        'Region': ['North', 'South'],
        'Payment_Method': ['Credit Card', 'PayPal']
    })

def test_create_features_math_logic(sample_data, mock_cfg):
    """Checks if the core ratios (LTV, Engagement) are calculated correctly."""
    fe = FeatureEngineer(mock_cfg)
    df_result = fe.create_features(sample_data)

    # 100 * 12 = 1200
    assert df_result['Estimated_LTV'].iloc[0] == 1200.0
    # 2 / (12 + 1) = 0.1538
    assert df_result['Engagement_Score'].iloc[0] == approx(0.1538, abs=1e-3)
    assert df_result['Is_Free_Trial'].iloc[0] == 0

def test_division_by_zero_safety(mock_cfg):
    """Ensures the code doesn't explode with zero tenure or zero spend."""
    fe = FeatureEngineer(mock_cfg)
    zero_data = pd.DataFrame({
        'Age': [18],
        'Subscription_Length': [0],
        'Monthly_Spend': [0.0],
        'Support_Tickets_Raised': [0],
        'Gender': ['Male'],
        'Region': ['North'],
        'Payment_Method': ['Credit Card']
    })
    df_result = fe.create_features(zero_data)
    assert np.isfinite(df_result['Engagement_Score'].iloc[0])
    assert df_result['Is_Free_Trial'].iloc[0] == 1

def test_feature_list_consistency(mock_cfg):
    """Verifies if the output column list matches expected count."""
    fe = FeatureEngineer(mock_cfg)
    expected_cols = fe.get_model_feature_names()
    # 4 numeric + 3 categorical + 6 engineered = 13
    assert len(expected_cols) == 13
    assert "Estimated_LTV" in expected_cols
