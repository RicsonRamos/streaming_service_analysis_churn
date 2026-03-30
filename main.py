import argparse
import logging
import sys
import os
import mlflow
from pathlib import Path

from src.config.loader import ConfigLoader
from src.pipelines.train import TrainingPipeline
from src.pipelines.promotion import ModelPromoter 

# 1. CONFIGURAÇÃO DE LOGGING (Imediata)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_mlflow_env(experiment_name: str):
    """
    Configura o Backend Store e o Experimento Ativo.
    Rigor: Sem isso, as runs caem no ID 0 (Default).
    """
    db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(db_path)
    
    # Define o experimento globalmente para todas as funções subsequentes
    mlflow.set_experiment(experiment_name)
    
    # Garante diretório de artefatos local
    Path("mlruns").mkdir(exist_ok=True)
    
    logger.info(f"MLflow: URI '{db_path}' | Experimento '{experiment_name}'")

def main():
    parser = argparse.ArgumentParser(description="Churn Radar: Pipeline de MLOps")
    
    parser.add_argument(
        "--mode", 
        choices=["train", "promote", "full"], 
        default="full",
        help="Modo de execução: train, promote ou full"
    )
    parser.add_argument(
        "--tune", 
        action="store_true", 
        help="Habilitar otimização de hiperparâmetros (Optuna)"
    )
    
    args = parser.parse_args()
    
    # 2. CARGA DE CONFIGURAÇÃO
    try:
        cfg_loader = ConfigLoader()
        cfg = cfg_loader.load()
        experiment_name = cfg.get("project", {}).get("name", "ChurnRadar")
    except Exception as e:
        logger.error(f"Falha ao carregar configurações: {e}")
        sys.exit(1)

    # 3. INICIALIZAÇÃO DO AMBIENTE MLFLOW
    setup_mlflow_env(experiment_name)
    
    run_id = None

    try:
        # 4. FASE DE TREINAMENTO
        if args.mode in ["train", "full"]:
            logger.info("--- FASE 1: TREINAMENTO ---")
            pipeline = TrainingPipeline()
            run_id = pipeline.run(tune=args.tune)
            
            if not run_id:
                raise ValueError("O pipeline de treino falhou ao retornar um Run ID.")
            
            logger.info(f"Sucesso: Run {run_id} concluída.")

        # 5. FASE DE PROMOÇÃO (Governança de Modelo)
        if args.mode in ["promote", "full"]:
            logger.info("--- FASE 2: AVALIAÇÃO E PROMOÇÃO ---")
            
            # Se rodou o treino agora, usa o run_id. Caso contrário, busca a última do banco.
            target_run = run_id
            if not target_run:
                logger.info("Buscando última run disponível no banco SQLite...")
                try:
                    last_run = mlflow.search_runs(
                        experiment_names=[experiment_name],
                        order_by=["start_time DESC"], 
                        max_results=1
                    )
                    if not last_run.empty:
                        target_run = last_run.iloc[0].run_id
                    else:
                        logger.warning(f"Nenhuma run encontrada para o experimento '{experiment_name}'.")
                        return
                except Exception as e:
                    logger.error(f"Erro ao acessar histórico do MLflow: {e}")
                    return

            # Executa a lógica de promoção (Check de métricas vs Baseline)
            promoter = ModelPromoter()
            success = promoter.evaluate_and_promote(target_run)
            
            if success:
                logger.info(f"RESULTADO: Modelo da Run {target_run} promovido para PRODUCTION.")
            else:
                logger.warning(f"RESULTADO: Promoção negada para a Run {target_run} (Métricas insuficientes).")

    except Exception as e:
        logger.error(f"Erro crítico no fluxo principal: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()