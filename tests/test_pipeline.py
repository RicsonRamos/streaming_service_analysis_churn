from src.pipelines.train import TrainingPipeline


def test_pipeline_runs():
    """
    Testa se o pipeline de treinamento executa corretamente e retorna um dicionário com as métricas e o caminho do modelo treinado.
    """
    pipeline = TrainingPipeline()
    result = pipeline.run(tune=False, sample=True)

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "model_path" in result

    # Verifica se a métrica de ROC-AUC do modelo treinado é maior que 0.5
    assert result["metrics"]["roc_auc"] > 0.5
