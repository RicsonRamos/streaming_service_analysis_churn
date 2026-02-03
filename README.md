Brutalmente honesto: Se o seu `README.md` for uma parede de texto genérica, ninguém vai ler. Se ele for um mapa técnico de decisões de engenharia, ele te consegue um emprego.

Abaixo está o `README.md` estruturado para um portfólio de **Sênior**. Ele reflete exatamente o que construímos: a briga contra o vazamento de dados, a precisão do XGBoost e a robustez da infraestrutura.

---

# Churn Radar: End-to-End Predictive Ecosystem

Este projeto é uma solução completa de **Machine Learning Operacional (MLOps)** para predição de rotatividade (*churn*) em serviços de streaming. Diferente de modelos de laboratório, o **Churn Radar** foi construído com foco em governança, explicabilidade e deploy escalável.

## Business Performance & ML Metrics

O modelo final foi otimizado para identificar clientes de alto risco antes da evasão, mantendo um equilíbrio rigoroso entre precisão e sensibilidade.

| Métrica | Resultado | Nota Técnica |
| --- | --- | --- |
| **ROC-AUC** | **0.85** | Validação robusta contra *overfitting*. |
| **F1-Score** | **0.78** | Equilíbrio real entre Precision e Recall. |
| **Revenue at Risk** | **$12.4k** | Identificado no dataset de teste (simulação). |

### Insights Concretos (Data-Driven)

* **Support Interactions:** Clientes com mais de 3 tickets abertos no mês têm 4.2x mais chance de churn.
* **Engagement Drop:** Uma queda de 20% no `Engagement_Score` nos últimos 10 dias é o preditor mais forte de saída iminente.
* **The "Senior" Factor:** Clientes acima de 60 anos possuem LTV 15% superior, mas são mais sensíveis a problemas de UX.

---

## Engenharia de Dados e Modelo

### 1. Tratamento de Desbalanceamento e Validação

Para evitar que o modelo ficasse "preguiçoso" devido ao desbalanceamento de classes (apenas ~15-20% de churn), aplicamos:

* **Estratégia:** Utilização do parâmetro `scale_pos_weight` no XGBoost, ajustando o custo do erro para a classe minoritária.
* **Validação:** Estratégia de **Stratified K-Fold (5 splits)** combinada com um **Hold-out set (20%)** final para garantir que as métricas de produção sejam realistas.

### 2. Combate ao Data Leakage (Vazamento)

Identificamos e removemos variáveis de "vazamento" (como `Last_Activity_Type` quando o valor indicava 'Account Cancellation'), o que reduziu um AUC artificial de 0.99 para um **0.85 real e confiável**.

---

## Arquitetura do Sistema

O projeto segue uma estrutura modular, separando lógica de negócio de interface:

```text
├── app/            # Streamlit Dashboard (UI/UX)
├── configs/        # Central YAML configuration (Single source of truth)
├── models/         # MLflow artifacts and versioned .joblib files
├── src/            # Core Engine (Feature Engineering, Services, Pipelines)
└── docker/         # Infrastructure as Code

```

### Stack Tecnológica

* **Engine:** Python 3.10, XGBoost, Scikit-learn.
* **Tracking:** **MLflow** para versionamento de modelos e experimentos.
* **Dashboard:** Streamlit com **SHAP** para explicabilidade global e local.
* **Infra:** Docker & Docker Compose.

---

## Deploy e Operação

### Como rodar o ecossistema

1. **Subir Infraestrutura:**
```bash
docker-compose up -d --build

```


2. **Executar Pipeline de Treino:**
```bash
docker exec -it churn_radar_prod python src/pipelines/train.py

```


3. **Acessar Interfaces:**
* **Dashboard:** `localhost:8501`
* **MLflow UI:** `localhost:5000`



### Escalabilidade e Manutenção em Produção

O sistema foi desenhado para o "Dia 2" da operação:

* **Novos Dados:** O `ChurnService` utiliza uma classe `FeatureEngineer` idêntica à do treino, garantindo que o dado em produção sofra as mesmas transformações (previne *Training-Serving Skew*).
* **Atualização do Modelo:** O dashboard consome o caminho definido no `config.yaml`. Para atualizar o modelo, basta apontar para o novo artefato do MLflow sem reiniciar o container.
* **Monitoramento de Deriva (Drift):** O pipeline está preparado para integração com ferramentas de monitoramento (ex: Evidently AI) para detectar quando o comportamento do usuário muda e o modelo precisa de **recalibração**.

---

## Validação e Testes Automáticos

* **Check de Sanidade:** O pipeline de predição valida tipos de dados (Dtype enforcement) antes da inferência para evitar crashes por strings inesperadas.
* **CI/CD Ready:** Estrutura pronta para GitHub Actions, validando o `Dockerfile` e a integridade dos artefatos em cada commit.

---

**Desenvolvido por Ricson Ramos**
*Focado em transformar dados brutos em decisões estratégicas reais.*
