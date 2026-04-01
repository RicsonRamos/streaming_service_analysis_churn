import logging

import mlflow

from src.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Classe que avalia as métricas de um Run e promove para 'Production' no Model Registry
    se os requisitos mínimos forem atingidos.
    """

    def __init__(self, tracking_uri=None):
        """
        Inicializa o Promoter com as configurações globais e o cliente MLflow.

        Args:
            tracking_uri: URI do servidor MLflow. Se None, tenta usar variável de ambiente
                           ou default para sqlite local (legado).
        """
        self.cfg = ConfigLoader().load()

        # Usa o tracking_uri passado, ou variável de ambiente, ou default legado
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

        # Configura o MLflow para usar o servidor
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

        self.model_name = "Churn-XGB-Prod"

        logger.info(f"ModelPromoter conectado ao MLflow em: {self.tracking_uri}")

    def evaluate_and_promote(self, run_id: str) -> bool:
        """
        Avalia as métricas de um Run e promove para 'Production' no Model Registry
        se os requisitos mínimos forem atingidos.

        Args:
            run_id: ID da run a ser promovida.

        Returns:
            bool: True se a promoção for bem-sucedida, False caso contrário.
        """
        try:
            # 1. Recuperar dados da Run
            run = self.client.get_run(run_id)
            metrics = run.data.metrics

            logger.info(f"Métricas encontradas na Run {run_id}: {metrics}")

            # 2. Busca Resiliente da Métrica de Performance
            auc_score = metrics.get("roc_auc", metrics.get("auc", 0.0))

            # 3. Critério de Aceitação
            threshold = 0.75

            if auc_score < threshold:
                logger.error(f"REPROVADO: AUC {auc_score:.4f} abaixo do threshold {threshold}.")
                return False

            # 4. Promoção no Model Registry
            logger.info(f"APROVADO: Promovendo modelo {run_id} para 'Production'...")

            # Regista o modelo
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, self.model_name)

            # Transição de Estágio (deprecated em MLflow 2.9+, mas ainda funciona)
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(
                f"Sucesso: {self.model_name} versão {model_version.version} está em Produção."
            )
            return True

        except Exception as e:
            logger.error(f"Erro durante a promoção do modelo: {str(e)}")
            return False

    def run(self, run_id=None):
        """
        Método wrapper para compatibilidade com main.py

        Args:
            run_id: ID da run a ser promovida.
        """
        if not run_id:
            logger.error("Run ID não fornecido para promoção")
            return False

        return self.evaluate_and_promote(run_id)
