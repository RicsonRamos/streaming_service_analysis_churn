import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from src.config.loader import ConfigLoader
from xgboost import XGBClassifier
from src.features.feature_engineering import FeatureEngineer
from src.features.validation import validate_dataframe
from src.models.training.tuner import ChurnTuner


# Configuração de logging profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_pipeline():
    # 0. Iniciar Configuração
    loader = ConfigLoader()
    cfg = loader.load_all()

    # Inicializar engenheiro de features 
    fe = FeatureEngineer(cfg)

    # 1. Carregar Dados
    data_path = Path(cfg["paths"]["data"]["processed"])
    if not data_path.exists():
        logger.error(f"Arquivo não encontrado: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    logger.info(f"Dados carregados com sucesso de: {data_path}")

    if not validate_dataframe(df):
        logger.error("Pipeline abortado: Dados inválidos.")
        return

    # 2. FEATURE ENGINEERING & REGISTRY (A regra de ouro)
    # Aqui o FeatureEngineer filtra as colunas e cria as novas métricas
    df_enriched = fe.create_features(df)
    features_list = fe.get_feature_names()
    target = cfg["model"]["features"]["target"]

    logger.info(f"Treinando com {len(features_list)} colunas (Oficiais + Engineered).")
    
    # 3. Preparar X e y (Usando apenas o que o Engenheiro autorizou)
    X = df_enriched[features_list].copy()
    y = df_enriched[target]

    # 4. Train/Test Split
    runtime_cfg = cfg["base"]["runtime"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=runtime_cfg["test_size"], 
        random_state=runtime_cfg["random_state"],
        stratify=y
    )

    # 5. Preprocessamento (One-Hot Encoding para colunas categóricas oficiais)
    cat_features = cfg["model"]["features"]["categorical"]
    X_train_final = pd.get_dummies(X_train, columns=cat_features)
    X_test_final = pd.get_dummies(X_test, columns=cat_features)

    # Garantir alinhamento de colunas
    X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

    model_params = cfg["model"]["model_params"]

    # Optimização de hiperparâmetros (opcional)
    if cfg['model'].get('tune_hyperprameters', True):
        tuner = ChurnTuner(X_train_final, y_train)
        best_params = tuner.optimize(n_trials=30)
        logger.info(f'New best hyperparameters: {best_params}')
        # Atualizar parâmetros do modelo com os melhores encontrados
        model_params.update(best_params)

        # save best params
        best_params_path = Path(cfg["paths"]["models"]["artifacts"]) / "best_model_params.yaml"
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, 'w') as f:
            import yaml
            yaml.dump(best_params, f)
        logger.info(f"Melhores parâmetros salvos em: {best_params_path}")


    # 6. Treinar XGBoost (Parâmetros do model.yaml)
    
    model = XGBClassifier(**model_params)
       
    logger.info("Iniciando treinamento do XGBoost...")
    model.fit(X_train_final, y_train)

    # 7. Avaliação
    probs = model.predict_proba(X_test_final)[:, 1]
    auc = roc_auc_score(y_test, probs)
    preds = model.predict(X_test_final)
    acc = accuracy_score(y_test, preds)

    logger.info(f"RESULTADOS REAIS: AUC={auc:.4f} | ACC={acc:.4f}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, preds))

    
    
    # Validação visual final no terminal
    logger.info(f"Colunas finais enviadas ao modelo: {list(X_train_final.columns)}")
    

    mlflow.sklearn.log_model(model, "Churn_Prediction_Streaming")

    y_pred = model.predict(X_test_final)
    
    report = classification_report(y_test, y_pred, output_dict=True)


    precision_1 = report['1']['precision']
    recall_1 = report['1']['recall']
    f1_1 = report['1']['f1-score']

    with mlflow.start_run(run_name='XGBoost_Optimized'):
        # Logar parâmetros e métricas
        mlflow.log_params(model_params)
   
        # Logar métricas
        mlflow.log_metric('precision', precision_1)
        mlflow.log_metric('recall_churn', recall_1)
        mlflow.log_metric('f1_churn', f1_1)

        # Salvar o modelo dentro do MLflow
        mlflow.sklearn.log_model(model, "Churn_Prediction_Streaming")

        logger.info("Modelo e métricas logados no MLflow com sucesso.")

        import matplotlib.pyplot as plt
        from xgboost import plot_importance


        # Plotar e salvar a importância das features
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(model, ax=ax, max_num_features=10)
        plt.tight_layout()

        # Salvar a figura
        chart_path = Path(cfg["paths"]["reports"]["feature_importance_plot"])
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        logger.info(f"Importância das features salva em: {chart_path}")
        
    # Validação visual final no terminal
    logger.info(f"Colunas finais enviadas ao modelo: {list(X_train_final.columns)}")

    # Salvamento de Artefatos
    output_path = Path(cfg["paths"]["models"]["churn_model"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "features": list(X_train_final.columns),
        "params": model_params,
        "metrics": {"auc": auc, "accuracy": acc}
    }
    
    joblib.dump(model_data, output_path)
    logger.info(f"Modelo salvo em: {output_path}")

if __name__ == "__main__":
    train_pipeline()