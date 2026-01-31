import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

class DumbModel(BaseEstimator, ClassifierMixin):
    """
    Modelo de Baseline que sempre prevê a classe majoritária.
    Essencial para provar o ROI (Retorno sobre Investimento) do projeto de IA.
    """
    def __init__(self):
        self.majority_class_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Descobre a classe majoritária (Quem não cancela)."""
        # Converte para Series para facilitar o cálculo da moda
        y_series = pd.Series(y)
        self.majority_class_ = y_series.mode()[0]
        self.classes_ = np.unique(y)
        
        print(f"[Baseline] Treinado. Estratégia: Prever sempre '{self.majority_class_}'")
        return self

    def predict(self, X):
        """Gera previsões constantes baseadas na maioria."""
        if self.majority_class_ is None:
            raise ValueError("O modelo precisa ser treinado com 'fit' antes de prever.")
        return np.full(shape=(X.shape[0],), fill_value=self.majority_class_)

    def predict_proba(self, X):
        """Retorna probabilidade 1.0 para a classe majoritária (necessário para AUC)."""
        probs = np.zeros((X.shape[0], len(self.classes_)))
        major_idx = np.where(self.classes_ == self.majority_class_)[0][0]
        probs[:, major_idx] = 1.0
        return probs

    def evaluate_business_impact(self, X_test, y_test):
        """
        Explica para o acionista o perigo de não usar IA.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Cálculo de impacto: quantos cancelamentos o 'chute' ignorou?
        total_churners = (y_test == 1).sum()
        missed_churners = total_churners # O DumbModel erra todos os churners por definição
        
        print("-" * 50)
        print(f"RESULTADO DO MODELO BASELINE (CHUTE):")
        print(f"Acurácia Nominal: {acc:.2%}")
        print(f"Alerta: O modelo ignorou {missed_churners} clientes que realmente cancelaram.")
        print(f"Conclusão: O chute ignora 100% do prejuízo financeiro.")
        print("-" * 50)
        
        return acc