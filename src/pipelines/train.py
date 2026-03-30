import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.config.loader import ConfigLoader
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost import ChurnXGBoost

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        """
        Inicializa o pipeline carregando configurações e instanciando ferramentas.
        """
        self.cfg = ConfigLoader().load()
        self.dl = DataLoader(self.cfg)
        self.fe = FeatureEngineer(self.cfg)
        
        # Nome do experimento no MLflow
        mlflow.set_experiment(self.cfg.get("project", {}).get("name", "ChurnRadar"))

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas ruidosas ou que causam leakage, 
        mas PROTEGE o Target para permitir o split e treino.
        """
        target_col = self.cfg['feature_schema']['target']
        drop_cols = self.cfg.get('feature_schema', {}).get('drop_columns', [])
        
        # Filtro Rigoroso: Remove apenas o que não for o Target.
        # Isso evita o KeyError: 'Churn' no momento do split.
        to_drop = [c for c in drop_cols if c in df.columns and c != target_col]
        
        df_clean = df.drop(columns=to_drop)
        logger.info(f"Colunas removidas (Leakage/Ruído): {to_drop}")
        
        return df_clean

    def run(self, tune: bool = False):
        """
        Executa o ciclo de MLOps: Carga -> Sanitização -> Split -> FE -> Treino -> Log.
        """
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Iniciando Run do MLflow: {run_id}")

            # 1. CARGA DE DADOS
            df_raw = self.dl.load_raw_data()
            
            # 2. SANITIZAÇÃO (Anti-Leakage)
            # Aqui o Churn deve permanecer no DataFrame.
            df_clean = self._sanitize_data(df_raw)
            
            target_col = self.cfg['feature_schema']['target']
            
            # Validação de segurança antes do Split
            if target_col not in df_clean.columns:
                raise KeyError(f"Erro Crítico: A coluna target '{target_col}' não foi encontrada após a sanitização.")

            # 3. SPLIT (Estratificado para manter proporção de Churn)
            train_df, test_df = train_test_split(
                df_clean, 
                test_size=self.cfg['model_metadata']['test_size'],
                random_state=self.cfg['hyperparameters']['random_state'],
                stratify=df_clean[target_col] 
            )

            # 4. ENGENHARIA DE FEATURES (Fit no treino, Transform em ambos)
            logger.info("Executando Feature Engineering (Zero Leakage)...")
            self.fe.fit(train_df)
            
            X_train = self.fe.transform(train_df)
            X_test = self.fe.transform(test_df)
            
            # 5. SEPARAÇÃO DO TARGET (Somente após a engenharia)
            y_train = X_train.pop(target_col)
            y_test = X_test.pop(target_col)

            # 6. TREINAMENTO
            model_wrapper = ChurnXGBoost(self.cfg)
            model_wrapper.train(X_train, y_train)

            # 7. AVALIAÇÃO
            y_proba = model_wrapper.predict_proba(X_test)
            y_pred = model_wrapper.predict(X_test)

            metrics = {
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred))
            }

            # 8. LOG DE PARÂMETROS E MÉTRICAS NO MLFLOW
            mlflow.log_params(self.cfg['hyperparameters'])
            mlflow.log_metrics(metrics)
            
            # Registro das colunas para consistência na inferência
            feature_names = X_train.columns.tolist()
            
            mlflow.sklearn.log_model(
                sk_model=model_wrapper.model,
                artifact_path="model",
                registered_model_name="Churn-XGB-Prod"
            )

            # 9. SALVAMENTO DO ARTEFATO (Dicionário completo para o Dashboard)
            artifact = {
                "model": model_wrapper.model,
                "features": feature_names,
                "metrics": metrics,
                "params": self.cfg['hyperparameters']
            }
            
            model_wrapper.save(
                path=self.cfg['artifacts']['current_model'], 
                artifact=artifact
            )

            logger.info(f"Pipeline finalizada. AUC: {metrics['roc_auc']:.4f}")
            return run_id