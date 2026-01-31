import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

# Importando seu estilo padronizado
# Certifique-se de que este módulo existe ou comente se não for usar no app.py
try:
    from src.eda import apply_storytelling_style
except ImportError:
    def apply_storytelling_style(): pass

class ChurnXGBoost:
    def __init__(self):
        # Define exatamente as colunas que o Pipeline deve gerenciar
        self.num_cols = ['Age', 'Subscription_Length', 'Support_Tickets_Raised', 
                         'Satisfaction_Score', 'Monthly_Spend', 'Estimated_LTV', 'Engagement_Score']
        self.cat_cols = ['Gender', 'Region', 'Payment_Method']
        self.is_trained = False
        self.feature_names = None

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols)
            ])

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])

    def train(self, X_train, y_train):
        """Treina o modelo garantindo que X_train seja um DataFrame."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Balanceamento de carga (Churn vs Stay)
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.model.set_params(classifier__scale_pos_weight=pos_weight)
        
        # TREINO: Passamos o DataFrame PURO para o Pipeline
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"🚀 [XGBoost] Modelo treinado com sucesso.")

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Relatório de métricas padrão."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        return f"{report}\nROC-AUC Score: {auc:.4f}"

    def evaluate_and_plot(self, X_test, y_test):
        """Relatório visual corrigido: REMOVIDO .values que causava erro."""
        if not self.is_trained: 
            raise Exception("Modelo não treinado.")

        # --- CORREÇÃO AQUI: Não usamos mais .values ---
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # Plotagem (Se necessário no seu ambiente)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax[0])
        ax[0].set_title('Matriz de Confusão')

        # 2. Importância das Variáveis
        # Nota: Acessar importâncias dentro de um Pipeline requer acessar o step 'classifier'
        importances = self.model.named_steps['classifier'].feature_importances_
        
        # O preprocessor cria novas colunas (OneHot), então o tamanho muda. 
        # Para simplificar o gráfico, plotamos as importâncias brutas.
        indices = np.argsort(importances)[-10:]
        ax[1].barh(range(len(indices)), importances[indices], color='#448aff')
        ax[1].set_title('Importância das Variáveis (Processadas)')
        
        plt.tight_layout()
        plt.show()

        print(f"\n🏆 AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))