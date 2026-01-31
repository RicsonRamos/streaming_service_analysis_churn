import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing import DataCleaner
from models.xgboost import ChurnXGBoost 

def main():
    print("🚀 Iniciando Pipeline de Churn Corrigido...")
    
    # 1. CARREGAMENTO
    df = pd.read_csv(r'data\raw\streaming.csv') # Use o bruto para ter Gender, Region, etc.
    
    # 2. ENGENHARIA DE FEATURES (Antes da limpeza pesada)
    # Criamos as colunas que o modelo e o dashboard precisam
    df['Estimated_LTV'] = df['Monthly_Spend'] * df['Subscription_Length']
    df['Engagement_Score'] = df['Support_Tickets_Raised'] / (df['Subscription_Length'] + 1)
    
    # 3. PRÉ-PROCESSAMENTO
    cleaner = DataCleaner()
    
    # Definimos exatamente o que é numérico e o que é categórico
    # IMPORTANTE: Customer_ID fica fora. Gender e Region entram como categóricas originais.
    num_cols = ['Age', 'Subscription_Length', 'Support_Tickets_Raised', 
                'Satisfaction_Score', 'Monthly_Spend', 'Estimated_LTV', 'Engagement_Score']
    cat_cols = ['Gender', 'Region', 'Payment_Method']
    target = 'Churned'

    # Limpeza básica (remover duplicatas, tratar nulos residuais)
    df_clean = cleaner.input_missing_values(df, num_cols, cat_cols)
    
    # 4. SEPARAÇÃO DE DADOS
    # Mantemos apenas as colunas que o Pipeline vai processar
    features = num_cols + cat_cols
    X = df_clean[features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. TREINAMENTO COM O PIPELINE
    # O seu modelo ChurnXGBoost deve conter o Pipeline com OneHotEncoder dentro
    xgb_model = ChurnXGBoost() 
    print(f"Training XGBoost model on {len(X_train)} samples...")
    xgb_model.train(X_train, y_train)

    # 6. AVALIAÇÃO (Verifique se o AUC não é 0.9999!)
    metrics = xgb_model.evaluate(X_test, y_test)
    print("\n✅ Performance do Modelo:")
    print(metrics)

    # 7. EXPORTAÇÃO DO PACOTE COMPLETO
    # Salvamos o objeto xgb_model que contém o Pipeline + Modelo
    model_path = Path("models/churn_model_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvamos o wrapper completo
    joblib.dump(xgb_model, model_path)
    
    # Opcional: Salvar uma versão do CSV limpo para o Dashboard usar de base
    df_clean.to_csv('data/processed/Streaming_Clean.csv', index=False)
    
    print(f"\n💾 Sucesso! Modelo exportado para: {model_path}")

if __name__ == "__main__":
    main()