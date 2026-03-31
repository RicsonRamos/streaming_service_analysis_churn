import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer
from pytest import approx

@pytest.fixture
def mock_cfg():
    """
    Provides a controlled configuration without relying on YAML files.

    This fixture is useful when we want to test the feature engineering logic
    without relying on the YAML configuration files.

    Returns:
        dict: A dictionary containing the feature schema configuration.
    """
    return {
        "feature_schema": {
            # These columns are used for LTV estimation
            "numeric": ["Age", "Subscription_Length", "Monthly_Spend", "Support_Tickets_Raised"],
            # These columns are used for one-hot encoding
            "categorical": ["Gender", "Region", "Payment_Method"]
        }
    }

@pytest.fixture
def sample_data():
    """
    Standard customer data for happy-path testing.

    This fixture returns a Pandas DataFrame containing a minimal
    set of customer data. The data is used for testing the
    feature engineering logic.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the customer data.
    """
    return pd.DataFrame({
        'Age': [25, 40],  # Age of customers
        'Subscription_Length': [12, 1],  # Subscription length in months
        'Monthly_Spend': [100.0, 50.0],  # Monthly spend in dollars
        'Support_Tickets_Raised': [2, 0],  # Support tickets raised by customers
        'Gender': ['Male', 'Female'],  # Gender of customers
        'Region': ['North', 'South'],  # Region where customers are located
        'Payment_Method': ['Credit Card', 'PayPal']  # Payment method used by customers
    })

def test_create_features_math_logic(sample_data, mock_cfg):
    """
    Checks if the core ratios (LTV, Engagement) are calculated correctly.

    This test case verifies if the feature engineering logic is correct
    by comparing the output of the feature engineering pipeline with
    expected results.

    Args:
        sample_data (pd.DataFrame): A Pandas DataFrame containing
            customer data.
        mock_cfg (dict): A dictionary containing the feature schema
            configuration.

    Returns:
        None
    """
    fe = FeatureEngineer(mock_cfg)
    df_result = fe.create_features(sample_data)

    # 100 * 12 = 1200
    assert df_result['Estimated_LTV'].iloc[0] == 1200.0, "LTV calculation is incorrect"
    # 2 / (12 + 1) = 0.1538
    assert df_result['Engagement_Score'].loc[0] == approx(0.1538, abs=1e-3), "Engagement calculation is incorrect"
    assert df_result['Is_Free_Trial'].loc[0] == 0, "Is_Free_Trial calculation is incorrect"

def test_division_by_zero_safety(mock_cfg):
    """
    Ensures the code doesn't explode with zero tenure or zero spend.

    The FeatureEngineer should gracefully handle division by zero and return
    valid results.

    Args:
        mock_cfg (dict): A dictionary containing the feature schema
            configuration.

    Returns:
        None
    """
    fe = FeatureEngineer(mock_cfg)
    zero_data = pd.DataFrame({
        # Age of customers
        'Age': [18],
        # Subscription length in months
        'Subscription_Length': [0],
        # Monthly spend in dollars
        'Monthly_Spend': [0.0],
        # Support tickets raised by customers
        'Support_Tickets_Raised': [0],
        # Gender of customers
        'Gender': ['Male'],
        # Region where customers are located
        'Region': ['North'],
        # Payment method used by customers
        'Payment_Method': ['Credit Card']
    })
    df_result = fe.create_features(zero_data)
    # Verify that division by zero doesn't result in NaN
    assert np.isfinite(df_result['Engagement_Score'].iloc[0]), "Division by zero must not result in NaN"
    # Verify that Is_Free_Trial is correctly set to 1
    assert df_result['Is_Free_Trial'].iloc[0] == 1, "Is_Free_Trial must be set to 1 if Subscription_Length is 0"

def test_feature_list_consistency(mock_cfg):
    """
    Verifies if the output column list matches expected count.

    This test case checks if the FeatureEngineer correctly generates
    the expected number of columns. The output column list should
    contain the engineered features (LTV, Engagement, Is_Free_Trial),
    as well as the numeric and categorical features.

    Args:
        mock_cfg (dict): A dictionary containing the feature schema
            configuration.

    Returns:
        None
    """
    fe = FeatureEngineer(mock_cfg)
    expected_cols = fe.get_model_feature_names()
    # 4 numeric + 3 categorical + 6 engineered = 13
    assert len(expected_cols) == 13, "Expected column count mismatch"
    assert "Estimated_LTV" in expected_cols, "Estimated_LTV column not found"
