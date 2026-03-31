import argparse
import logging
import sys
import os
import mlflow
from pathlib import Path

from src.config.loader import ConfigLoader
from src.pipelines.train import TrainingPipeline
from src.pipelines.promotion import ModelPromoter 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_mlflow_env(experiment_name: str):
    """Configura a conexão com o servidor de rastreamento."""
    # CRÍTICO: Usar o servidor MLflow, não SQLite local
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"Conectado ao MLflow em: {tracking_uri}")
    return tracking_uri

def main():
    parser = argparse.ArgumentParser(description="Churn Radar: Pipeline de MLOps")
    parser.add_argument("--mode", choices=["train", "promote", "full"], default="full")
    parser.add_argument("--tune", action="store_true", help="Habilitar otimização (Optuna)")
    args = parser.parse_args()

    # Carga de configuração
    try:
        cfg_loader = ConfigLoader()
        cfg = cfg_loader.load()
        experiment_name = cfg.get("project", {}).get("name", "ChurnRadar")
    except Exception as e:
        logger.error(f"Falha ao carregar configurações: {e}")
        sys.exit(1)

    # Inicialização do ambiente MLflow
    tracking_uri = setup_mlflow_env(experiment_name)
    run_id = None

    try:
        # Fase de Treinamento
        if args.mode in ["train", "full"]:
            logger.info("--- FASE 1: TREINAMENTO ---")
            pipeline = TrainingPipeline()
            run_id = pipeline.run(tune=args.tune)
            
            if not run_id:
                raise ValueError("O pipeline de treino falhou ao retornar um Run ID.")
            
            logger.info(f"Sucesso: Run {run_id} concluída.")

        # Fase de Promoção
        if args.mode in ["promote", "full"]:
            logger.info("--- FASE 2: AVALIAÇÃO E PROMOÇÃO ---")
            
            if not run_id:
                logger.info("Buscando última run disponível...")
                last_runs = mlflow.search_runs(
                    experiment_names=[experiment_name],
                    order_by=["start_time DESC"], 
                    max_results=1
                )
                if not last_runs.empty:
                    run_id = last_runs.iloc[0].run_id
                else:
                    logger.warning("Nenhuma run encontrada.")
                    return

            # Passar tracking_uri para o promoter
            promoter = ModelPromoter(tracking_uri=tracking_uri)
            success = promoter.run(run_id=run_id)

            if success:
                logger.info(f"RESULTADO: Modelo da Run {run_id} promovido para PRODUCTION.")

    except Exception as e:
        logger.error(f"Erro crítico no pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()