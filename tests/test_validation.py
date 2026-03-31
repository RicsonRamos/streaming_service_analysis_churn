"""
Unit tests for Data Validation logic.
Ensures the Pydantic schema correctly filters out corrupt or illogical data.
"""

import pytest
import pandas as pd
from src.features.validation import validate_dataframe

@pytest.fixture
def base_valid_row():
    """
    Returns a dictionary with perfectly valid data.

    This fixture provides a row of data that is known to pass the validation
    checks. It is used as the basis for testing that the validation logic
    correctly filters out corrupt or illogical data.
    """
    return {
        # Age must be between 17 and 100
        'Age': 30,
        # Subscription length must be at least 1
        'Subscription_Length': 12,
        # Monthly spend must be greater than 0
        'Monthly_Spend': 50.0,
        # Support tickets raised must be at least 0
        'Support_Tickets_Raised': 1,
        # Estimated LTV is a calculated value
        'Estimated_LTV': 600.0,
        # Engagement score is a calculated value
        'Engagement_Score': 5.0,
        # Gender must be one of 'Male', 'Female'
        'Gender': 'Male',
        # Region must be one of 'North', 'South'
        'Region': 'North',
        # Payment method must be one of 'Credit Card', 'PayPal'
        'Payment_Method': 'Credit Card'
    }

def test_validate_dataframe_accepts_good_data(base_valid_row):
    """
    Confirm that correct data returns True.

    This test ensures that when we pass a DataFrame with valid data, the
    validation logic returns True.
    """
    df = pd.DataFrame([base_valid_row])
    assert validate_dataframe(df) is True

def test_validate_dataframe_rejects_out_of_range_age(base_valid_row):
    """
    Checks the 'gt=17, lt=100' constraint on the 'Age' column.
    
    Ensures that the validation logic correctly filters out data that
    is outside of the expected range.
    """
    # Test upper bound
    bad_data_old = pd.DataFrame([{**base_valid_row, 'Age': 150}])
    # Test lower bound
    bad_data_young = pd.DataFrame([{**base_valid_row, 'Age': 5}])
    
    # The validation logic should reject data that is outside of the expected range
    assert validate_dataframe(bad_data_old) is False
    assert validate_dataframe(bad_data_young) is False

def test_validate_dataframe_rejects_invalid_categorical_values(base_valid_row):
    """
    Ensures our field_validators are working for Gender and Region.

    Field validators are Pydantic's way of validating individual fields.
    They are used to ensure that the data being passed into the model is correct
    and consistent.

    In this case, we are checking that our field_validators correctly reject
    invalid categorical values.
    """
    # Invalid Gender
    bad_gender = pd.DataFrame([{**base_valid_row, 'Gender': 'Robot'}])
    # Invalid Region (Not in our whitelist)
    bad_region = pd.DataFrame([{**base_valid_row, 'Region': 'Mars'}])
    
    # The validation logic should reject data with invalid categorical values
    assert validate_dataframe(bad_gender) is False, "Invalid Gender should be rejected"
    assert validate_dataframe(bad_region) is False, "Invalid Region should be rejected"

def test_validate_dataframe_rejects_negative_values(base_valid_row):
    """
    Business logic: Spend and Length cannot be negative.

    The validation logic should reject data with negative values.
    """
    # Test negative spend
    bad_spend = pd.DataFrame([{**base_valid_row, 'Monthly_Spend': -10.0}])
    # Test negative length
    bad_length = pd.DataFrame([{**base_valid_row, 'Subscription_Length': 0}])
    
    # The validation logic should reject data with negative values
    assert validate_dataframe(bad_spend) is False, "Negative spend should be rejected"
    assert validate_dataframe(bad_length) is False, "Negative length should be rejected"

def test_validate_dataframe_handles_mixed_batch(base_valid_row):
    """
    Test that a batch with one single error should fail the entire validation.

    This test ensures that the validation logic correctly handles batches with
    mixed good and bad data. It should reject the entire batch if even a single
    record fails the validation rules.
    """
    # Create a batch with one good record and one bad record
    mixed_data = pd.DataFrame([
        base_valid_row,                # Good
        {**base_valid_row, 'Age': 200} # Bad
    ])
    
    # The validation logic should reject the entire batch if any record fails
    assert validate_dataframe(mixed_data) is False
