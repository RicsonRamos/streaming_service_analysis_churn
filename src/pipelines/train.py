import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from src.config.loader import ConfigLoader
from xgboost import XGBClassifier
from src.features.feature_engineering import FeatureEngineer
from src.features.validation import validate_dataframe
from src.models.training.tuner import ChurnTuner

mlflow.set_experiment("Churn_Streaming_Production")

# Configuração de logging profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():

    # 1. SETUP & CONFIG

    loader = ConfigLoader()

    cfg = loader.load_all()

    fe = FeatureEngineer(cfg)    

    # 2. CARREGAMENTO (Usando caminhos do Config)

    raw_data_path = Path(cfg["paths"]["data"]["raw"]) / "streaming.csv"

    df = pd.read_csv(raw_data_path)

    

    # 3. ENGENHARIA DE FEATURES (Centralizada)

    # Aqui o FeatureEngineer cria LTV, Engagement, etc., e filtra as colunas

    df_enriched = fe.create_features(df)

    features = fe.get_feature_names()

    target = cfg["model"]["features"]["target"]

    

    X = df_enriched[features]

    y = df_enriched[target]

    

    # 4. SPLIT

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, 

        test_size=cfg["base"]["runtime"]["test_size"], 

        random_state=cfg["base"]["runtime"]["random_state"]

    )

    

    # 5. PREPROCESSAMENTO (Categorical)

    cat_features = cfg["model"]["features"]["categorical"]

    X_train_final = pd.get_dummies(X_train, columns=cat_features)

    X_test_final = pd.get_dummies(X_test, columns=cat_features).reindex(columns=X_train_final.columns, fill_value=0)

    

    # 6. TREINO

    model = XGBClassifier(**cfg["model"]["model_params"])

    model.fit(X_train_final, y_train)

    

    # 7. EXPORTAÇÃO (O formato que o seu Dashboard espera)

    model_path = Path(cfg["paths"]["models"]["churn_model"])

    model_path.parent.mkdir(parents=True, exist_ok=True)

    

    artifact = {

        "model": model,

        "features": X_train_final.columns.tolist()

    }

    

    joblib.dump(artifact, model_path)

    print(f"✅ Pipeline Finalizado! Modelo salvo em: {model_path}")

def train_pipeline():
    # 0. Iniciar Configuração
    loader = ConfigLoader()
    cfg = loader.load_all()
    fe = FeatureEngineer(cfg)

    # 1. Carregar e Validar Dados
    data_path = Path(cfg["paths"]["data"]["processed"])
    df = pd.read_csv(data_path)
    if not validate_dataframe(df): return

    # 2. Feature Engineering
    df_enriched = fe.create_features(df)
    features_list = fe.get_feature_names()
    target = cfg["model"]["features"]["target"]

    X = df_enriched[features_list].copy()
    y = df_enriched[target]

    # 3. Split e Preprocessamento
    runtime_cfg = cfg["base"]["runtime"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=runtime_cfg["test_size"], 
        random_state=runtime_cfg["random_state"], stratify=y
    )

    cat_features = cfg["model"]["features"]["categorical"]
    X_train_final = pd.get_dummies(X_train, columns=cat_features)
    X_test_final = pd.get_dummies(X_test, columns=cat_features).reindex(columns=X_train_final.columns, fill_value=0)

    # 4. Iniciar Experimento MLflow
    with mlflow.start_run(run_name='XGBoost_Final_Refactor'):
        model_params = cfg["model"]["model_params"]

        # Otimização (Opcional)
        if cfg['model'].get('tune_hyperparameters', True):
            tuner = ChurnTuner(X_train_final, y_train)
            best_params = tuner.optimize(n_trials=30)
            model_params.update(best_params)
            mlflow.log_params(best_params)

        # 5. TREINAR (Agora sim o objeto 'model' passa a existir)
        model = XGBClassifier(**model_params)
        model.fit(X_train_final, y_train)

        # 6. Avaliação
        probs = model.predict_proba(X_test_final)[:, 1]
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, model.predict(X_test_final))
        
        mlflow.log_metric('auc', auc)
        mlflow.log_metric('accuracy', acc)

        # Otimização de hiperparâmetros
        if cfg['model'].get('tune_hyperparameters', True):
            logger.info("Iniciando a busca pelos melhores hiperparâmetros...")
            tuner = ChurnTuner(X_train_final, y_train)
            best_params = tuner.optimize(n_trials=30)
            
            logger.info(f'Novos melhores parâmetros: {best_params}')
            model_params.update(best_params)

            # SALVAMENTO DO YAML (O que estava a faltar)
            # Definir o caminho (ex: models/artifacts/best_model_params.yaml)
            best_params_path = Path(cfg["paths"]["models"]["artifacts"]) / "best_model_params.yaml"
            
            with open(best_params_path, 'w') as f:
                import yaml
                yaml.dump(best_params, f)
            
            logger.info(f"✅ Hiperparâmetros salvos em: {best_params_path}")
            mlflow.log_params(best_params)

        # 7. SALVAMENTO PARA O DASHBOARD (Usando a função auxiliar)
        # Salvamos as colunas do X_train_final porque é o que o modelo espera após o dummies
        save_model(model, X_train_final.columns.tolist(), cfg)

        # 8. Registro no MLflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="Churn_XGB_Prod")
        
        logger.info(f"Treino concluído: AUC {auc:.4f}")

def save_model(model, features_list, cfg):
    # Use o nome correto da chave do seu YAML (provavelmente churn_model no singular)
    model_path = cfg['paths']['models']['churn_model']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    artifact = {
        "model": model,
        "features": features_list
    }

    joblib.dump(artifact, model_path)
    logger.info(f'✅ Artefato exportado para: {model_path}')

if __name__ == "__main__":
    train_pipeline()