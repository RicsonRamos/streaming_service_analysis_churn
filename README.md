# 🛡️ Streaming Service Churn Radar (v2.0 - Production Ready)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-orange.svg)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Native-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Interface-Operational%20Dash-red.svg)](https://streamlit.io/)

Este projeto é uma plataforma completa de **MLOps** para predição e gestão de Churn em serviços de streaming. Diferente de modelos puramente acadêmicos, esta arquitetura foi desenhada para **ação tática**, integrando o treinamento rastreável à uma interface de decisão que identifica clientes em risco em tempo real.

## 🎯 Objetivo Estratégico
Transformar dados históricos em **inteligência preventiva**. O sistema identifica padrões comportamentais que precedem o cancelamento, permitindo que a equipe de retenção atue nos clientes de maior valor antes da perda definitiva.

---

## 🏗️ Arquitetura do Sistema
O projeto utiliza uma estrutura modular e desacoplada, garantindo que modificações na engenharia de atributos não quebrem a interface de usuário.

* **Data Pipeline:** Carregamento sanitizado com remoção automática de *Target Leakage* (IDs e métricas de satisfação pós-evento).
* **ML Engine:** XGBoost com suporte nativo a variáveis categóricas.
* **Governança (MLflow):** Experimentos registrados em banco SQLite local, com versionamento de modelos no *Model Registry*.
* **Operational Dash:** Interface Streamlit que consome o modelo em estágio de `Production` para gerar listas de alvos.

---

## 📊 Performance e Rigor Estatístico
Após a limpeza de variáveis viciadas (`Customer_ID`, `Satisfaction_Score`, `Last_Activity`), o modelo atingiu métricas realistas e robustas:

| Métrica | Valor Final | Status |
| :--- | :--- | :--- |
| **ROC AUC** | **0.8595** | ✅ Robusto |
| **Accuracy** | **0.838** | ✅ Confiável |
| **F1-Score** | **0.804** | ✅ Equilibrado |

> **Nota do Analista:** Modelos com AUC 1.00 foram descartados durante o desenvolvimento por apresentarem vazamento de dados (*data leakage*). A versão atual (v12) reflete o comportamento real e generalizável dos clientes.

---

## 🛠️ Tecnologias Utilizadas
* **Core:** Python 3.12, Pandas, Scikit-Learn.
* **Modelo:** XGBoost (Extreme Gradient Boosting).
* **MLOps:** MLflow (Tracking & Model Registry).
* **Interface:** Streamlit (Dashboard Preditivo).
* **Banco de Metadados:** SQLite.

---

## 🚀 Execução do Pipeline

### 1. Preparação do Ambiente
```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Ciclo de Vida do Modelo (Treino e Promoção)
O comando abaixo executa o pipeline completo: carrega dados, treina, valida métricas e, se aprovado, promove o modelo para o estágio de `Production` no MLflow.
```bash
python main.py --mode full
```

### 3. Monitoramento e Dashboard
Para visualizar as métricas de todos os experimentos e as versões do modelo:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Para abrir o Painel de Controle e gerar a **Lista de Alvos**:
```bash
streamlit run app/main_dash.py
```

---

## 🖥️ Interface Operacional
O dashboard foi desenvolvido para usuários de negócio e analistas de retenção, permitindo:
1.  **Ajuste de Threshold:** Definir a sensibilidade do risco (ex: focar apenas em quem tem > 80% de chance de sair).
2.  **Lista de Alvos:** Tabela identificável com `Customer_ID`, `Monthly_Spend` e `Region` para ação imediata.
3.  **Exportação:** Download da lista de clientes em risco (CSV) para integração com ferramentas de CRM.

---

## 📂 Estrutura do Projeto
```text
├── app/               # Interface Streamlit (Dashboard)
├── data/
│   └── raw/           # Dados brutos (streaming.csv)
├── models/
│   └── artifacts/     # Binários do modelo (.joblib)
├── src/
│   ├── config/        # Gestão de YAML (Schemas e Hyperparams)
│   ├── data/          # Data Loading e Sanitização
│   ├── features/      # Engenharia de Atributos
│   ├── models/        # Wrappers do XGBoost
│   └── pipelines/     # Orquestração de Treino e Promoção
├── main.py            # Entry point do sistema (CLI)
└── mlflow.db          # Banco de dados de governança (SQLite)
```

---

## 📈 Próximos Passos
- [ ] **Dockerização:** Empacotamento em Docker Compose para isolamento total de infraestrutura.
- [ ] **SHAP Integration:** Adicionar explicabilidade individual para entender os motivos específicos de cada churn.
- [ ] **API Endpoint:** Criar uma rota FastAPI para consultas externas em tempo real.

---
**Desenvolvido por Ricson Ramos** *Analista de Dados & Engenheiro de Machine Learning* [LinkedIn](https://www.linkedin.com/in/ricsonramos/) | [GitHub](https://github.com/RicsonRamos)
