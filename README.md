# Streaming Service Churn Radar

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: XGBoost](https://img.shields.io/badge/model-XGBoost%20Native-green.svg)](https://xgboost.readthedocs.io/)
[![Tracking: MLflow](https://img.shields.io/badge/tracking-MLflow-orange.svg)](https://mlflow.org/)
[![Explainability: SHAP](https://img.shields.io/badge/explainability-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Interface: Streamlit](https://img.shields.io/badge/interface-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset: Kaggle](https://img.shields.io/badge/dataset-Kaggle%20Streaming-informational.svg)](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)

Pipeline de ponta a ponta para predição probabilística de cancelamento (churn) em serviços de streaming, integrando governança de experimentos, calibração de modelos, explicabilidade individual via SHAP e simulação de retorno sobre investimento (ROI) de retenção.

---

## 1. Problema de Negócio e Contexto

Em plataformas de assinatura de conteúdo, o custo de aquisição de novos clientes (CAC) supera em até 5 a 7 vezes o custo de retenção da base existente. Entretanto, ações cegas e indiscriminadas de retenção geram duas ineficiências graves:
1. **Desperdício de Margem:** Concessão de descontos e incentivos financeiros a clientes que nunca tiveram a intenção real de cancelar.
2. **Campanhas Tardias e Inócuas:** Tentativas de contato com usuários cuja perda já é irreversível devido a experiências degradadas acumuladas.

O objetivo do projeto foi conceber um sistema preditivo que conecte diretamente probabilidades matemáticas de churn ao impacto econômico de campanhas de marketing:
- Identificar assinantes em risco com antecedência hábil para intervenção.
- Estimar probabilidades calibradas em vez de classes binárias rígidas.
- Fornecer os fatores causais individuais de cada previsão para apoiar a equipe de CRM.
- Simular o ponto de equilíbrio financeiro (threshold ótimo de corte) maximizando o Lifetime Value (LTV) líquido salvo.

---

## 2. Processo e Racional das Escolhas Técnicas

### 2.1. Prevenção Rígida de Target Leakage (Vazamento de Dados)
- **Eliminação de Variáveis Pós-Evento:** Variáveis como pontuações de satisfação obtidas no encerramento de contrato (`Satisfaction_Score`) e registros de data de término (`Last_Activity`) foram categoricamente excluídas da matriz de features. Modelos treinados com essas variáveis exibem acurácias ilusórias próximas a 100% em testes offline, mas falham completamente em produção.

### 2.2. Modelagem com XGBoost Native Categorical
- **XGBoost vs. Encodings Convencionais:** A utilização do suporte nativo do XGBoost a variáveis categóricas (`enable_categorical=True`) eliminou a necessidade de One-Hot Encoding com explosão de dimensionalidade, preservando a esparsidade dos dados e acelerando o tempo de convergência.
- **Calibração de Probabilidades:** Previsões puras de modelos baseados em árvores frequentemente concentram probabilidades nos extremos (0 ou 1). Foi realizada a avaliação da Curva de Calibração e do Brier Score para garantir que uma probabilidade atribuída de 0.70 corresponda de fato a uma taxa histórica empírica de 70% de cancelamentos.

### 2.3. Explicabilidade Causal via SHAP (SHapley Additive exPlanations)
- **Interpretabilidade Local e Global:** A integração do `TreeExplainer` do SHAP permitiu decompor exatamente o peso de cada vetor (ex.: histórico de chamados no suporte não resolvidos, plano mensal e tempo de permanência) na propensão ao churn de cada cliente específico, fornecendo insumos práticos para a régua de comunicação.

### 2.4. Engenharia de Produção e CI
- **Validação de Schemas com Pydantic v2:** Validação em tempo de execução dos dados de entrada antes da inferência.
- **Esteira de Integração Contínua:** Linting estrito com Ruff, cobertura de testes com Pytest e empacotamento em contêiner Docker multi-stage via `uv`.

---

## 3. Resultados Atingidos e Métricas de Validação

### 3.1. Performance do Modelo
- **Discriminação Preditiva:** O modelo XGBoost atingiu ROC-AUC superior a 0.83 no conjunto de teste independente, superando com folga o baseline linear e a média da classe majoritária.
- **Brier Score Controlado:** Erro quadrático médio de probabilidades minimizado, assegurando calibração adequada para a tomada de decisão financeira.

### 3.2. Simulação de ROI de Retenção
- **Otimização do Threshold de Decisão:** Sob a premissa de um custo de retenção de \$15 por cliente contatado e uma preservação de \$120 em 12 meses de LTV com 40% de taxa de conversão, o ponto ótimo de intervenção foi calibrado no limiar $p \approx 0.65 - 0.70$. Essa estratégia maximizou o retorno líquido comparada ao limiar padrão ingênuo de 0.50.

---

## 4. Arquitetura da Solução e Estrutura

```text
streaming_service_analysis_churn/
├── app/
│   ├── main.py                        # Dashboard operacional em Streamlit
│   └── components/                    # Componentes modulares da interface
├── src/
│   ├── data/                          # Módulos de ingestão automatizada
│   ├── features/                      # Engenharia de atributos e validação Pydantic
│   └── models/                        # Treinamento, calibração e inferência SHAP
├── tests/                             # Testes automatizados unitários e de integração
├── Dockerfile                         # Conteinerização multi-stage baseada em uv
├── pyproject.toml                     # Metadados e dependências padronizadas (PEP 621)
└── README.md                          # Documentação técnica do projeto
```

---

## 5. Como Executar

### 5.1. Pré-requisitos
- Python 3.10 ou superior
- Gerenciador de pacotes `uv` (recomendado) ou `pip`
- Docker (opcional, para execução isolada)

### 5.2. Execução Local com `uv`

1. Clone o repositório:
```bash
git clone https://github.com/RicsonRamos/streaming_service_analysis_churn.git
cd streaming_service_analysis_churn
```

2. Crie e ative o ambiente virtual:
```bash
uv venv
source .venv/bin/activate
```

3. Instale as dependências:
```bash
uv pip install -e ".[dev]"
```

4. Execute os testes automatizados e o linter:
```bash
pytest
ruff check .
```

5. Inicialize a aplicação Streamlit:
```bash
streamlit run app/main.py
```
Acesse `http://localhost:8501`.

### 5.3. Execução via Docker

```bash
docker build -t churn-radar .
docker run -p 8501:8501 churn-radar
```
Acesse `http://localhost:8501`.

---

## Autor

- **Nome:** Ricson Ramos
- **Perfil GitHub:** [github.com/RicsonRamos](https://github.com/RicsonRamos)
- **LinkedIn:** [linkedin.com/in/ricsonramos](https://www.linkedin.com/in/ricsonramos/)
- **Kaggle:** [kaggle.com/ricsonramos](https://www.kaggle.com/ricsonramos)
- **Portfólio:** [ricsonramos.github.io](https://ricsonramos.github.io)
