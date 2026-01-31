import pandas as pd
import numpy as np

def analyze_clv_risk(X_test, y_proba, feature_names, monthly_spend_col='Monthly_Spend'):
    """
    Transforma predições em métricas financeiras.
    """
    # Criar DataFrame de resultados garantindo os nomes das colunas
    results = pd.DataFrame(X_test, columns=feature_names).copy()
    results['Churn_Probability'] = y_proba
    
    # Premissa de Negócio: Projeção para 12 meses
    avg_lifespan = 12 
    
    # CLV Projetado (Valor que o cliente traria em 1 ano)
    results['Projected_CLV'] = results[monthly_spend_col] * avg_lifespan
    
    # Receita em Risco (Valor esperado de perda ajustado pela probabilidade)
    results['Revenue_at_Risk'] = results['Projected_CLV'] * results['Churn_Probability']
    
    return results

def calculate_retention_roi(n_high_risk, avg_clv, cost_per_client, conversion_rate=0.3):
    """
    Calcula o retorno sobre investimento (ROI) de uma campanha de retenção.
    
    :param n_high_risk: Número de clientes identificados com alto risco.
    :param avg_clv: Valor médio que o cliente traz de volta (LTV).
    :param cost_per_client: Quanto custa a ação de retenção por cliente.
    :param conversion_rate: Porcentagem de clientes que aceitam a oferta (default 30%).
    :return: (receita_recuperada, investimento_total, roi_percentual)
    """
    if n_high_risk == 0:
        return 0.0, 0.0, 0.0
    
    investimento_total = n_high_risk * cost_per_client
    # Estimativa de quantos clientes salvaremos de fato
    clientes_salvos = n_high_risk * conversion_rate
    receita_recuperada = clientes_salvos * avg_clv
    
    lucro_liquido = receita_recuperada - investimento_total
    roi_percentual = (lucro_liquido / investimento_total) * 100 if investimento_total > 0 else 0
    
    return receita_recuperada, investimento_total, roi_percentual