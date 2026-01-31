# 🛡️ Streaming Service Churn Radar: Da Análise à Produção

Este repositório contém uma solução completa de ciência de dados para previsão e gestão de **Churn** (cancelamento de assinaturas). O projeto percorre todo o ciclo de vida de um produto de dados: desde a análise exploratória em Notebooks, passando pela engenharia de software no pipeline de treinamento, até a entrega de um Dashboard interativo.

## 📋 Sumário

* [Visão Geral do Projeto](https://www.google.com/search?q=%23-vis%C3%A3o-geral-do-projeto)
* [Estrutura do Repositório](https://www.google.com/search?q=%23-estrutura-do-reposit%C3%B3rio)
* [Arquitetura da Solução](https://www.google.com/search?q=%23-arquitetura-da-solu%C3%A7%C3%A3o)
* [Principais Insights](https://www.google.com/search?q=%23-principais-insights)
* [Como Executar](https://www.google.com/search?q=%23-como-executar)
* [A Interface Streamlit](https://www.google.com/search?q=%23-a-interface-streamlit)

---

## 🎯 Visão Geral do Projeto

O objetivo principal é reduzir o faturamento perdido por cancelamentos de assinaturas. Utilizando o algoritmo **XGBoost**, o modelo analisa comportamentos históricos para atribuir uma pontuação de risco a cada cliente, permitindo que a equipe de marketing tome decisões baseadas em dados.

---

## 📂 Estrutura do Repositório

```text
├── notebooks/           # Análise Exploratória (EDA) e prototipagem do modelo
├── data/
│   ├── raw/             # Dados brutos (imutáveis)
│   └── processed/       # Dados limpos e preparados para o modelo
├── src/
│   ├── data_cleaning.py # Funções de saneamento de dados
│   ├── finance.py       # Cálculos de ROI e métricas de negócio (LTV)
│   └── eda.py           # Funções de estilização e gráficos
├── models/
│   ├── xgboost.py       # Classe ChurnXGBoost (Pipeline Scikit-Learn + XGBoost)
│   └── churn_model.joblib # O modelo treinado finalizado
├── main.py              # Script principal de treinamento e avaliação
├── app.py               # Interface do Dashboard (Streamlit)
└── requirements.txt     # Dependências do projeto

```

---

## ⚙️ Arquitetura da Solução

O projeto foi construído sobre três pilares fundamentais:

### 1. O Pipeline de Treinamento (`main.py` + `models/`)

Utilizamos um **Scikit-Learn Pipeline** para evitar *Data Leakage* (vazamento de dados). O pipeline automatiza:

* **Imputação e Escalonamento:** Tratamento de dados numéricos.
* **One-Hot Encoding:** Transformação de variáveis categóricas (Região, Gênero, Pagamento).
* **Balanceamento de Classe:** Uso do parâmetro `scale_pos_weight` para lidar com a minoria de clientes que cancelam.

### 2. A Inteligência do Modelo (`XGBoost`)

O XGBoost foi escolhido por sua alta performance em dados tabulares e capacidade de lidar com relações não lineares complexas. O modelo não apenas prevê "quem vai sair", mas fornece a **probabilidade** (0 a 100%), permitindo segmentar clientes em risco Baixo, Médio e Alto.

---

## 📊 A Interface Streamlit

O Dashboard (`app.py`) transforma as predições técnicas em uma **ferramenta de gestão**:

* **Simulador de Negócios:** Permite ajustar o custo de retenção e ver o ROI potencial em tempo real.
* **Matriz de Priorização:** Cruza a probabilidade de Churn com o LTV (Lifetime Value), apontando quais clientes devem ser contatados primeiro.
* **Exportação de Leads:** O time comercial pode baixar um CSV filtrado apenas com os clientes de alto risco para ações imediatas.

---

## 📈 Principais Insights

Durante a análise (EDA), identificamos os principais gatilhos de cancelamento:

* **Suporte:** Clientes com mais de 3 chamados abertos têm 60% mais chance de Churn.
* **Engajamento:** Scores de engajamento abaixo de 40 pontos são fortes indicadores de saída iminente.
* **Financeiro:** O aumento no valor mensal sem oferta de upgrade é o principal motivo de churn na região Sul.

---

## 🚀 Como Executar

1. **Instalar dependências:**
```bash
pip install -r requirements.txt

```


2. **Treinar o modelo (Gera o arquivo .joblib):**
```bash
python main.py

```


3. **Rodar o dashboard:**
```bash
streamlit run app.py

```



---

## 📧 Contato

Desenvolvido por **Ricson Ramos**
