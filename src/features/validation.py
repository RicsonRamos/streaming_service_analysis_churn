"""
Data Validation Module.

Defines the contract for customer data using Pydantic schemas to ensure
type safety and business logic compliance before inference.
"""

from typing import List, Optional
from collections import Counter

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict, validator


class CustomerSchema(BaseModel):
    """
    Data contract for a single customer record.
    Ensures types and ranges are valid before feature engineering.
    """

    model_config = ConfigDict(
        extra="forbid",  # Corrigido: extra em minúsculo
    )

    Age: int = Field(gt=17, lt=100)
    Subscription_Length: int = Field(ge=1)
    Monthly_Spend: float = Field(ge=0)
    Support_Tickets_Raised: int = Field(ge=0)
    Gender: str
    Region: str
    Payment_Method: str
    estimated_LTV: Optional[float] = Field(ge=0, default=0.0)
    Engagement_Score: Optional[float] = Field(ge=0, default=0.0)

    @validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = ["Male", "Female"]
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v

    @validator("Region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        allowed = ["North", "South", "East", "West"]
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
    errors: List[str] = []
    sample_details: List[str] = []

    records = df.to_dict(orient="records")
    for i, record in enumerate(records):
        try:
            CustomerSchema(**record)  # Corrigido: sem aspas
        except ValidationError as e:
            for error in e.errors():
                loc = error["loc"][0]
                msg = error["msg"]
                errors.append(f"Field '{loc}': {msg}")
            if len(sample_details) < 3:
                sample_details.append(f"Row {i} failure: {e.json()}")

    if errors:
        print("\nDATA VALIDATION REPORT")
        print("-" * 35)
        for err, count in Counter(errors).items():
            print(f"- {count} occurrences of: {err}")

        if sample_details:
            print("\nDEBUG SAMPLE (First 3 errors):")
            for detail in sample_details:
                print(detail)
        return False

    print("Data validation successful!")
    return True