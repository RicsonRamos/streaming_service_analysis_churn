# No arquivo tests/test_validation.py
from src.features.validation import validate_dataframe
import pandas as pd 

def test_invalid_age_rejection():
    bad_data = pd.DataFrame({
        'Age': [500],
        'Subscription_Length': [12],
        'Monthly_Spend': [100.0],
        'Support_Tickets_Raised': [2],
        'Estimated_LTV': [1200.0],
        'Engagement_Score': [5],
        'Gender': ['Male'],
        'Region': ['North'],
        'Payment_Method': ['Credit Card']
    })
    
    assert validate_dataframe(bad_data) is False

    good_data = pd.DataFrame({
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
    
    assert validate_dataframe(good_data) is True