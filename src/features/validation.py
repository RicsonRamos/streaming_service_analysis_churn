"""
Data Validation Module.

Defines the contract for customer data using Pydantic schemas to ensure 
type safety and business logic compliance before inference.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from typing import List, Optional
import pandas as pd
from collections import Counter

class CustomerSchema(BaseModel):
    """
    Data contract for a single customer record.
    Ensures types and ranges are valid before feature engineering.
    """
    model_config = ConfigDict(
        coerce_numbers_to_str=False, 
        extra='ignore',  # Ignore extra columns to keep the model focused
        strict=False 
    )

    Age: int = Field(gt=17, lt=100)
    Subscription_Length: int = Field(ge=1)
    Monthly_Spend: float = Field(ge=0)
    Support_Tickets_Raised: int = Field(ge=0)
    Gender: str
    Region: str
    Payment_Method: str
    
    # Note: Engineered features are usually validated after creation
    # or made optional if validating raw input.
    Estimated_LTV: Optional[float] = Field(ge=0, default=0.0)
    Engagement_Score: Optional[float] = Field(ge=0, default=0.0)

    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validates that gender matches expected categories."""
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v

    @field_validator('Region')
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validates that region matches the training set categories."""
        allowed = ['North', 'South', 'East', 'West', 'Central', 'Germany', 'France', 'Spain']
        if v not in allowed:
            raise ValueError(f"Region '{v}' is not supported.")
        return v

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validates a pandas DataFrame against the CustomerSchema.
    
    Args:
        df (pd.DataFrame): Data to validate.

    Returns:
        bool: True if all records are valid, False otherwise.
    """
    errors = []
    sample_details = []

    # Record conversion for bulk validation
    records = df.to_dict(orient="records")

    for i, record in enumerate(records):
        try:
            CustomerSchema(**record)
        except ValidationError as e:
            # Capture the specific field that failed
            for error in e.errors():
                loc = error['loc'][0]
                msg = error['msg']
                errors.append(f"Field '{loc}': {msg}")
                
            if len(sample_details) < 3:
                sample_details.append(f"Row {i} failure: {e.json()}")

    if errors:
        print("\nDATA VALIDATION REPORT")
        print("-" * 35)
        error_summary = Counter(errors)
        for err, count in error_summary.items():
            print(f"- {count} occurrences of: {err}")

        if sample_details:
            print("\nDEBUG SAMPLE (First 3 errors):")
            for detail in sample_details:
                print(detail)
        return False

    print("Data validation successful!")
    return True
