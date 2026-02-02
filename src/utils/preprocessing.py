"""
General preprocessing utilities for data cleaning and formatting.
"""
import re

def clean_column_names(df):
    """
    Removes special characters and spaces from DataFrame columns.
    """
    df.columns = [re.sub(r'\W+', '_', col).strip('_') for col in df.columns]
    return df

def format_currency(value: float) -> str:
    """Formats a float as USD currency string."""
    return f"USD {value:,.2f}"
