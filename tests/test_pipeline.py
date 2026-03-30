from src.pipelines.train import TrainingPipeline

def test_pipeline_runs():
    pipeline = TrainingPipeline()
    result = pipeline.run(tune=False, sample=True)

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "model_path" in result

    assert result["metrics"]["roc_auc"] > 0.5