
# 🛡️ Streaming Service Churn Radar (v3.0 - Production Ready)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-orange.svg)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Native-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Interface-Operational%20Dash-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)

Plataforma completa de **MLOps** para predição e gestão de Churn em serviços de streaming. Arquitetura desenhada para **ação tática em tempo real**, integrando treinamento rastreável, governança de modelos e interface de decisão com explicabilidade individual.

## 🎯 Objetivo Estratégico

Transformar dados históricos em **inteligência preventiva acionável**. O sistema identifica padrões comportamentais que precedem o cancelamento, permitindo que a equipe de retenção atue proativamente nos clientes de maior valor antes da perda definitiva.

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Layer    │────▶│   ML Pipeline    │────▶│  MLflow Server  │
│  (SQLite/CSV)   │     │ (XGBoost + SHAP) │     │  (Registry)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                         │
                              ┌──────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Streamlit Dash  │
                    │  (Operational)   │
                    └──────────────────┘
```

### Componentes Principais

| Camada | Tecnologia | Função |
|:---|:---|:---|
| **Data Pipeline** | Pandas + YAML | Carregamento sanitizado com remoção automática de *Target Leakage* |
| **ML Engine** | XGBoost (Native Categorical) | Modelo com suporte nativo a variáveis categóricas |
| **Explainability** | SHAP | Análise de contribuição individual por feature |
| **Governança** | MLflow | Tracking, Model Registry e versionamento |
| **Interface** | Streamlit | Dashboard operacional com lista de alvos e SHAP |

---

## 📊 Performance e Rigor Estatístico

Após limpeza rigorosa de variáveis viciadas (`Customer_ID`, `Satisfaction_Score`, `Last_Activity`):

| Métrica | Valor | Metodologia | Status |
|:---|:---|:---|:---|
| **ROC AUC** | **0.8493** | Split Temporal (Out-of-Time) | ✅ Robusto |
| **Accuracy** | **0.822** | Validação no "futuro" | ✅ Confiável |
| **F1-Score** | **0.779** | Balanceado Precision/Recall | ✅ Equilibrado |

> **⚠️ Nota de Rigor:** Modelos com AUC > 0.95 foram descartados por apresentarem *data leakage*. A versão atual (v3.0) reflete comportamento real e generalizável, validado temporalmente.

### Por que Split Temporal?

Diferente de split aleatório tradicional, nossa abordagem simula a **passagem do tempo**:

```
Dados Históricos: [████████████████████░░░░░░░░░░]
                  └─ Treino (80%) ─┘└── Teste (20%) ─┘
                                     (Futuro simulado)
```

Isso garante que o modelo seja avaliado em dados que **nunca viu**, refletindo cenário real de produção.

---

## 🔍 Explicabilidade SHAP

Cada predição é acompanhada de análise de contribuição das features:

| Visualização | Propósito |
|:---|:---|
| **Waterfall Plot** | Explicação individual do cliente selecionado |
| **Summary Plot** | Importância global das features na base |
| **Force Plot** | Direção do impacto (positivo/negativo para retenção) |

> **Exemplo de Insight:** *"Cliente ID-4523 tem 85% de risco de churn porque seus `Support_Tickets_Raised` (3 tickets) e `Discount_Offered` (0%) contribuem +12% e +8% respectivamente para o risco."*

---

## 🚀 Execução do Sistema

### 1. Preparação do Ambiente (Docker - Recomendado)

```bash
# Clonar e entrar no diretório
git clone <repo-url>
cd streaming_service_analysis_churn

# Subir infraestrutura completa
docker-compose up -d

# Executar pipeline de treinamento
docker exec -it churn_trainer python main.py --mode full
```

### 2. Acesso às Interfaces

| Serviço | URL | Descrição |
|:---|:---|:---|
| **MLflow UI** | http://localhost:5000 | Experiments, runs e model registry |
| **Dashboard** | http://localhost:8501 | Interface operacional com SHAP |
| **Modelo API** | `http://mlflow:5000` | Endpoint interno para predições |

### 3. Ciclo de Vida do Modelo

```bash
# Treino completo com promoção automática
docker exec -it churn_trainer python main.py --mode full

# Apenas treino (sem promoção)
docker exec -it churn_trainer python main.py --mode train

# Apenas promoção do modelo existente
docker exec -it churn_trainer python main.py --mode promote

# Com otimização de hiperparâmetros (Optuna)
docker exec -it churn_trainer python main.py --mode full --tune
```

---

## 🖥️ Interface Operacional

O dashboard Streamlit oferece:

### Painel Principal
1. **KPIs em Tempo Real:** Total de clientes, alto risco, probabilidade média
2. **Distribuição de Risco:** Histograma interativo com threshold ajustável
3. **Análise Geográfica:** Risco médio por região

### Lista Tática de Alvos
- **Filtro de Prioridade:** Slider de threshold (0.0 - 1.0)
- **Colunas Estratégicas:** Customer_ID, probabilidade, spend, região
- **Exportação CSV:** Download direto para CRM

### Análise SHAP Individual
```
Cliente selecionado: ID-4523
├── Probabilidade de Churn: 85%
├── Top 3 Motivos de Risco:
│   ├── Support_Tickets_Raised: +12% (3 tickets)
│   ├── Discount_Offered: +8% (0% desconto)
│   └── Monthly_Spend: +5% (R$ 89,90)
└── Recomendação: Oferecer desconto de 20% + atendimento prioritário
```

---

## 📂 Estrutura do Projeto

```text
streaming_service_analysis_churn/
├── app/
│   ├── main_dash.py              # Interface Streamlit (com SHAP)
│   ├── components.py             # Componentes UI reutilizáveis
│   └── services.py               # Lógica de negócio
├── docker/
│   ├── Dockerfile                # Imagem da aplicação
│   └── Dockerfile.mlflow         # Imagem do MLflow server
├── src/
│   ├── config/
│   │   └── loader.py             # Gestão de YAML
│   ├── data/
│   │   └── data_loader.py        # Sanitização e leakage removal
│   ├── features/
│   │   └── feature_engineering.py # Pipeline de FE
│   ├── models/
│   │   └── xgboost.py            # Wrapper XGBoost + SHAP
│   └── pipelines/
│       ├── train.py              # Pipeline com split temporal
│       └── promotion.py          # Governança de modelos
├── data/
│   └── raw/
│       └── streaming.csv         # Dataset base
├── docker-compose.yml            # Orquestração completa
├── main.py                       # Entry point CLI
└── requirements.txt              # Dependências
```

---

## 🛠️ Tecnologias e Stack

| Categoria | Ferramentas |
|:---|:---|
| **Core** | Python 3.12, Pandas, NumPy, Scikit-Learn |
| **Modelagem** | XGBoost 2.0+ (categorical nativo) |
| **Explicabilidade** | SHAP, Matplotlib |
| **MLOps** | MLflow 2.9+ (Tracking, Registry, Artifacts) |
| **Interface** | Streamlit, Plotly |
| **Infraestrutura** | Docker, Docker Compose |
| **Validação** | Split Temporal, ROC-AUC, F1-Score |

---

## 📈 Roadmap e Próximos Passos

- [x] **Split Temporal:** Validação out-of-time implementada
- [x] **SHAP Integration:** Explicabilidade individual por cliente
- [x] **Dockerização:** Ambiente completo containerizado
- [ ] **API FastAPI:** Endpoint REST para predições em tempo real
- [ ] **A/B Testing:** Framework para testes de modelos em produção
- [ ] **Drift Detection:** Monitoramento automático de degradação de performance
- [ ] **Auto-Retraining:** Pipeline de re-treino automático mensal

---

## 👤 Autor

**Ricson Ramos**  
*Analista de Dados & Engenheiro de Machine Learning*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ricsonramos/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/RicsonRamos)

---

## 📝 Notas de Versão

### v3.0 (Mar/2026)
- Implementação de **Split Temporal** para validação realista
- Integração de **SHAP** para explicabilidade individual
- **Dockerização** completa com MLflow server dedicado
- Correção de **data leakage** em variáveis de satisfação

### v2.0 (Fev/2026)
- Estrutura base com XGBoost e MLflow
- Dashboard Streamlit operacional
- Sistema de promoção automática para Production

---

**Licença:** MIT | **Status:** Production Ready ✅
