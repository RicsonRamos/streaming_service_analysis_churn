import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from src.config.loader import ConfigLoader
from xgboost import XGBClassifier
from src.features.feature_engineering import FeatureEngineer
from src.features.validation import validate_dataframe

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

    # 6. Treinar XGBoost (Parâmetros do model.yaml)
    model_params = cfg["model"]["model_params"]
    model = XGBClassifier(**model_params, eval_metric='logloss')
    
    logger.info("Iniciando treinamento do XGBoost...")
    model.fit(X_train_final, y_train)

    # 7. Avaliação
    probs = model.predict_proba(X_test_final)[:, 1]
    auc = roc_auc_score(y_test, probs)
    preds = model.predict(X_test_final)
    acc = accuracy_score(y_test, preds)

    logger.info(f"RESULTADOS REAIS: AUC={auc:.4f} | ACC={acc:.4f}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, preds))

    # 8. Salvamento de Artefatos
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
    
    # Validação visual final no terminal
    logger.info(f"Colunas finais enviadas ao modelo: {list(X_train_final.columns)}")

if __name__ == "__main__":
    train_pipeline()