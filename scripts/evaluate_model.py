import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config.loader import ConfigLoader

def run_evaluation():
    cfg = ConfigLoader().load_all()
    
    # 1. Carregar Dados e Modelo
    test_data = pd.read_csv(cfg["paths"]["data"]["processed"]) # Ou seu test.csv
    artifact = joblib.load(cfg["paths"]["models"]["churn_model"])
    
    model = artifact["model"]
    features = artifact["features"]
    
    X_test = test_data[features]
    y_test = test_data["Churned"]
    
    # 2. Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Métricas para o README
    print("=== RELATÓRIO DE PERFORMANCE ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 4. Gerar e Salvar Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Radar de Churn')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    
    # Salva o arquivo para você linkar no README
    plt.savefig("reports/confusion_matrix.png")
    print("\nMatriz de Confusão salva em reports/confusion_matrix.png")

if __name__ == "__main__":
    run_evaluation()
