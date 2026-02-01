import pytest
import pandas as pd
from src.features.feature_engineering import FeatureEngineer
from src.config.loader import ConfigLoader
from pytest import approx


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Age': [25],
        'Subscription_Length': [12],
        'Monthly_Spend': [100.0],
        'Support_Tickets_Raised': [2],
        'Estimated_LTV': [1200.0],
        'Engagement_Score': [5],
        'Gender': ['Male'],
        'Region': ['North'],
        'Payment_Method': ['Credit Card']
    })

def test_ltv_spend_ratio_calculation(sample_data):
    loader = ConfigLoader()
    cfg = loader.load_all()
    fe = FeatureEngineer(cfg)

    df_result = fe.create_features(sample_data)

    expected_ratio = 12.0
    assert df_result['LTV_Spend_Ratio'].iloc[0] == approx(12.0,rel=0.05)
    assert not df_result.isnull().values.any() # Verifique se nenhum valor é nulo