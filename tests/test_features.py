import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def mock_cfg():
    return {
        "feature_schema": {
            "target": "Churned",
            "drop_columns": ["Customer_ID", "Satisfaction_Score"],
            "numeric": ["Age", "Subscription_Length", "Monthly_Spend", "Support_Tickets_Raised"],
            "categorical": ["Gender", "Region", "Payment_Method"],
        }
    }


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2"],
            "Satisfaction_Score": [5, 2],
            "Age": [25, 40],
            "Subscription_Length": [12, 1],
            "Monthly_Spend": [100.0, 50.0],
            "Support_Tickets_Raised": [2, 0],
            "Gender": ["Male", "Female"],
            "Region": ["North", "South"],
            "Payment_Method": ["Credit Card", "PayPal"],
        }
    )


def test_feature_engineer_fit_transform(sample_data, mock_cfg):
    fe = FeatureEngineer(mock_cfg)
    fe.fit(sample_data)
    transformed_df = fe.transform(sample_data)

    # Verifica se colunas de drop foram removidas
    assert "Customer_ID" not in transformed_df.columns
    assert "Satisfaction_Score" not in transformed_df.columns

    # Verifica se categóricas viraram categoria
    assert str(transformed_df["Gender"].dtype) == "category"
    assert str(transformed_df["Region"].dtype) == "category"
    assert str(transformed_df["Payment_Method"].dtype) == "category"


def test_feature_engineer_handles_missing_categories(sample_data, mock_cfg):
    fe = FeatureEngineer(mock_cfg)
    fe.fit(sample_data)

    # Novo dado com categoria não vista no fit
    unseen_data = sample_data.copy()
    unseen_data.loc[0, "Region"] = "UnknownRegion"

    transformed_df = fe.transform(unseen_data)
    # Categorias não vistas viram NaN de forma segura
    assert pd.isna(transformed_df.loc[0, "Region"])
