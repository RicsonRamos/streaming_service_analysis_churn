# Streaming Service Churn Radar

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: XGBoost](https://img.shields.io/badge/model-XGBoost%20Native-green.svg)](https://xgboost.readthedocs.io/)
[![Tracking: MLflow](https://img.shields.io/badge/tracking-MLflow-orange.svg)](https://mlflow.org/)
[![Explainability: SHAP](https://img.shields.io/badge/explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Interface: Streamlit](https://img.shields.io/badge/interface-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset: Kaggle](https://img.shields.io/badge/dataset-Kaggle%20Streaming-informational.svg)](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)

Pipeline de ponta a ponta para predição probabilística de cancelamento (churn) em serviços de streaming, integrando governança de experimentos, calibração de modelos, explicabilidade individual via SHAP e simulação de retorno sobre investimento (ROI) de retenção.

---

## Autor

- **Nome:** Ricson Ramos
- **Perfil GitHub:** [github.com/RicsonRamos](https://github.com/RicsonRamos)
- **LinkedIn:** [linkedin.com/in/ricsonramos](https://www.linkedin.com/in/ricsonramos/)
- **Kaggle:** [kaggle.com/ricsonramos](https://www.kaggle.com/ricsonramos)
- **Portfólio:** [ricsonramos.github.io](https://ricsonramos.github.io)
- **Contato:** ricsonramos@outlook.com

---

## Contexto de Negócio e Objetivos

Em plataformas de assinatura, o custo de aquisição de novos clientes (CAC) supera significativamente os custos de retenção. No entanto, intervenções de retenção indiscriminadas desperdiçam orçamento operacional oferecendo descontos a assinantes que não tinham intenção de cancelar, ou tentando reter clientes cuja perda é irreversível.

Este projeto resolve essa assimetria estratégica ao conectar a predição probabilística à decisão financeira:
1. **Prevenção de Data Leakage:** Descarte fundamentado de variáveis pós-evento (como notas de satisfação pós-atendimento ou registros de última atividade vinculados ao encerramento) para evitar métricas infladas artificialmente.
2. **Modelagem Preditiva Calibrada:** Algoritmo XGBoost com calibração de probabilidades para que os scores representem riscos reais de desligamento, avaliados por ROC-AUC e Log Loss.
3. **Explicabilidade com SHAP:** Decomposição dos principais vetores de insatisfação (ex.: tempo de contrato, tickets de suporte não resolvidos, valor mensal).
4. **Simulação Operacional de ROI:** Tradução das probabilidades em tomada de decisão: identificação do threshold ótimo de intervenção com base no custo de campanha vs. Lifetime Value (LTV) preservado.

---

## Dataset

- **Fonte Oficial:** [Streaming Service Data no Kaggle](https://www.kaggle.com/datasets/akashanandt/streaming-service-data) (por Akash Anand)
- **Tamanho:** 5.000 registros de assinantes
- **Principais Atributos:** Idade, gênero, tempo de assinatura, região geográfica, método de pagamento, histórico de tickets de suporte, ticket médio mensal e status de cancelamento (`Churned`).

---

## Arquitetura da Solução

```text
streaming_service_analysis_churn/
├── app/
│   ├── main.py                        # Dashboard operacional em Streamlit
│   └── services.py                    # Regras de negócio e motor financeiro de ROI
├── configs/
│   └── config.yaml                    # Contrato de dados, esquemas e hiperparâmetros
├── data/
│   ├── raw/                           # Dataset bruto
│   └── processed/                     # Dados sanitizados e particionados
├── docker/
│   └── Dockerfile                     # Imagem de produção conteinerizada
├── docker-compose.yml                 # Orquestração local de serviços e MLflow
├── models/
│   └── artifacts/                     # Modelos serializados e pipelines de inferência
├── notebooks/
│   └── churn_prediction_roi_analysis.ipynb # Notebook executável completo para o Kaggle
├── src/
│   ├── config/loader.py               # Leitor centralizado de configurações
│   ├── data/cleaner.py                # Pipeline de higienização e validação de schema
│   ├── models/classifier.py           # Engine de modelagem preditiva em XGBoost
│   ├── explainability/shap_engine.py  # Geração e extração de valores SHAP
│   └── pipelines/train.py             # Pipeline completo de treino e tracking
├── tests/
│   └── test_smoke.py                  # Testes automatizados de validação
├── pyproject.toml                     # Dependências do projeto gerenciadas via uv
└── requirements.txt                   # Espelho para ambientes legados
```

---

## Como Executar

### Pré-requisitos

Instale o gerenciador de pacotes [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clonar e Configurar Dependências

```bash
git clone https://github.com/RicsonRamos/streaming_service_analysis_churn.git
cd streaming_service_analysis_churn
uv sync --all-groups
```

### 2. Download do Dataset via Kagglehub

Execute o script de download programático ou deixe que o pipeline obtenha os dados:

```bash
uv run python -c "import kagglehub, shutil, os; p = kagglehub.dataset_download('akashanandt/streaming-service-data'); os.makedirs('data/raw', exist_ok=True); shutil.copy(os.path.join(p, 'Streaming.csv'), 'data/raw/streaming.csv'); print('Dataset pronto em data/raw/streaming.csv')"
```

### 3. Executar o Treinamento e Rastreamento

```bash
uv run python main.py --mode train
```

### 4. Executar o Dashboard Streamlit

```bash
uv run streamlit run app/main.py
```

O painel interativo estará disponível em `http://localhost:8501`.

### 5. Execução via Docker Compose

```bash
docker compose up --build
```

---

## Notebook no Kaggle

Para executar a análise de ponta a ponta no Kaggle:
- Acesse o arquivo [`notebooks/churn_prediction_roi_analysis.ipynb`](./notebooks/churn_prediction_roi_analysis.ipynb).
- O notebook detecta automaticamente o ambiente do Kaggle (`/kaggle/input/streaming-service-data`) ou realiza o download via `kagglehub` quando executado localmente.
