import mlflow
import logging
from src.config.loader import ConfigLoader

logger = logging.getLogger(__name__)

class ModelPromoter:
    def __init__(self):
        """
        Inicializa o Promoter com as configurações globais e o cliente MLflow.
        """
        self.cfg = ConfigLoader().load()
        # Garante que o Promoter olha para o mesmo banco de dados do treino
        self.tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.model_name = "Churn-XGB-Prod"

    def evaluate_and_promote(self, run_id: str) -> bool:
        """
        Avalia as métricas de um Run e promove para 'Production' no Model Registry
        se os requisitos mínimos forem atingidos.
        """
        try:
            # 1. Recuperar dados da Run
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            
            logger.info(f"Métricas encontradas na Run {run_id}: {metrics}")

            # 2. Busca Resiliente da Métrica de Performance
            # Tentamos 'roc_auc' (definido no novo train.py) ou 'auc' (legado)
            auc_score = metrics.get("roc_auc", metrics.get("auc", 0.0))
            
            # 3. Critério de Aceitação (Rigor Técnico)
            # Valor sugerido: 0.75 (Pode ser parametrizado no config.yaml)
            threshold = 0.75 
            
            if auc_score < threshold:
                logger.error(
                    f"REPROVADO: AUC {auc_score:.4f} abaixo do threshold {threshold}. "
                    "Verifique se as métricas foram corretamente logadas no treino."
                )
                return False

            # 4. Promoção no Model Registry
            logger.info(f"APROVADO: Promovendo modelo {run_id} para o estágio 'Production'...")

            # Regista o modelo se ainda não estiver registado
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, self.model_name)

            # Transição de Estágio
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True # Remove a versão antiga da produção
            )

            logger.info(f"Sucesso: {self.model_name} versão {model_version.version} está em Produção.")
            return True

        except Exception as e:
            logger.error(f"Erro durante a promoção do modelo: {str(e)}")
            return False