"""
Unit tests for Data Validation logic.
Ensures the Pydantic schema correctly filters out corrupt or illogical data.
"""

import pytest
import pandas as pd
from src.features.validation import validate_dataframe

@pytest.fixture
def base_valid_row():
    """Returns a dictionary with perfectly valid data."""
    return {
        'Age': 30,
        'Subscription_Length': 12,
        'Monthly_Spend': 50.0,
        'Support_Tickets_Raised': 1,
        'Estimated_LTV': 600.0,
        'Engagement_Score': 5.0,
        'Gender': 'Male',
        'Region': 'North',
        'Payment_Method': 'Credit Card'
    }

def test_validate_dataframe_accepts_good_data(base_valid_row):
    """Confirm that correct data returns True."""
    df = pd.DataFrame([base_valid_row])
    assert validate_dataframe(df) is True

def test_validate_dataframe_rejects_out_of_range_age(base_valid_row):
    """Checks the 'gt=17, lt=100' constraint."""
    # Test upper bound
    bad_data_old = pd.DataFrame([{**base_valid_row, 'Age': 150}])
    # Test lower bound
    bad_data_young = pd.DataFrame([{**base_valid_row, 'Age': 5}])
    
    assert validate_dataframe(bad_data_old) is False
    assert validate_dataframe(bad_data_young) is False

def test_validate_dataframe_rejects_invalid_categorical_values(base_valid_row):
    """Ensures our field_validators are working for Gender and Region."""
    # Invalid Gender
    bad_gender = pd.DataFrame([{**base_valid_row, 'Gender': 'Robot'}])
    # Invalid Region (Not in our whitelist)
    bad_region = pd.DataFrame([{**base_valid_row, 'Region': 'Mars'}])
    
    assert validate_dataframe(bad_gender) is False
    assert validate_dataframe(bad_region) is False

def test_validate_dataframe_rejects_negative_values(base_valid_row):
    """Business logic: Spend and Length cannot be negative."""
    bad_spend = pd.DataFrame([{**base_valid_row, 'Monthly_Spend': -10.0}])
    bad_length = pd.DataFrame([{**base_valid_row, 'Subscription_Length': 0}]) # ge=1
    
    assert validate_dataframe(bad_spend) is False
    assert validate_dataframe(bad_length) is False

def test_validate_dataframe_handles_mixed_batch(base_valid_row):
    """A batch with one single error should fail the entire validation."""
    mixed_data = pd.DataFrame([
        base_valid_row,                # Good
        {**base_valid_row, 'Age': 200} # Bad
    ])
    assert validate_dataframe(mixed_data) is False
