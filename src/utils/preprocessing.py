"""
General preprocessing utilities for data cleaning and formatting.
"""
import re
import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes special characters and spaces from DataFrame columns."""
    df.columns = [re.sub(r'\W+', '_', col).strip('_') for col in df.columns]
    return df

def format_currency(value: float) -> str:
    """Formats a float as USD currency string."""
    return f"USD {value:,.2f}"

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa NaN e valores inválidos:
    - 'Age': preenche com mediana
    - 'Monthly_Spend': preenche valores <0 ou NaN com 0
    """
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)

    if 'Monthly_Spend' in df.columns:
        df['Monthly_Spend'] = df['Monthly_Spend'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)

    return df