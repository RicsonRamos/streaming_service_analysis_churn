# src/pipelines/train.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from src.preprocessing import DataCleaner, get_preprocessor
from src.config.loader import ConfigLoader
from xgboost import XGBClassifier


def train_pipeline(cfg):
    # 1. Carregar CSV limpo
    data_path = cfg["paths"]["data"]["processed"]
    df = pd.read_csv(data_path)

    target_col = "Churned"

    # 2. Limpeza final
    cleaner = DataCleaner()
    df_clean, num_cols, cat_cols = cleaner.clean_and_prepare_data(df, target_col=target_col)

    # 3. Separar features e target
    X = df_clean[num_cols + cat_cols]
    y = df_clean[target_col]

    # 4. Separar treino/teste
    random_state = cfg["base"]["runtime"]["random_state"]
    test_size = cfg["base"]["runtime"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 5. Criar preprocessor
    preprocessor = get_preprocessor(num_cols, cat_cols)

    # 6. Transformar os dados
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 7. Treinar modelo XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_transformed, y_train)

    # 8. Avaliar
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    y_pred = model.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)

    print(f"[Metrics] AUC: {auc:.4f} | Accuracy: {acc:.4f}")

    # 9. Salvar modelo + pipeline
    model_path = Path(cfg["paths"]["models"]["churn_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Salvando o pipeline completo para deploy
    joblib.dump({"preprocessor": preprocessor, "model": model}, model_path)
    print(f"[Success] Modelo e pipeline salvos em: {model_path}")

if __name__ == "__main__":
    train_pipeline()
