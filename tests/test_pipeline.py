from src.config.loader import ConfigLoader
from src.features.feature_engineering import FeatureEngineer


def test_config_loader():
    cfg = ConfigLoader().load()
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "feature_schema" in cfg


def test_feature_engineering_initialization():
    cfg = ConfigLoader().load()
    fe = FeatureEngineer(cfg)
    assert fe is not None
    assert fe.target == "Churned"
