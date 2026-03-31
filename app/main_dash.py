import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 1. CONFIGURAÇÃO DE AMBIENTE E PÁGINA
st.set_page_config(
    page_title="Churn Radar 2026 | Inteligência Preditiva", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização para Alerta de Alta Prioridade
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÕES DE CARGA (Rigor de Governança)
@st.cache_resource
def load_production_model():
    """Busca o modelo 'Production' no MLflow Registry via SERVIDOR."""
    # CORREÇÃO 1: Usar servidor MLflow, não SQLite local
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    model_name = "Churn-XGB-Prod"
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        return model
    except Exception as e:
        st.error(f"Erro ao acessar Model Registry: {e}")
        return None

@st.cache_data
def get_data():
    """Carrega a base bruta para manter os IDs e metadados de exibição."""
    # CORREÇÃO 2: Caminho absoluto no container Docker
    path = "/app/data/raw/streaming.csv"
    if not Path(path).exists():
        st.error(f"Arquivo não encontrado em {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# 3. LÓGICA DE INTERFACE
st.title("🛡️ Churn Radar: Inteligência de Retenção")
st.sidebar.header("⚙️ Configurações de Negócio")

# Threshold de Risco (Gatilho para Alertas)
threshold = st.sidebar.slider(
    "Risco de Churn (Threshold)", 
    min_value=0.0, max_value=1.0, value=0.70, step=0.05,
    help="Define a probabilidade mínima para um cliente ser considerado 'Alto Risco'."
)

# NOVO: Botão para análise SHAP
show_shap = st.sidebar.checkbox("🔍 Mostrar Análise SHAP", value=False, 
                                help="Explica a importância das features para cada predição")

model = load_production_model()
df_raw = get_data()

if model and not df_raw.empty:
    # --- PROCESSAMENTO EM TEMPO REAL ---
    drop_cols = ['Churned', 'Customer_ID', 'Satisfaction_Score', 'Last_Activity']
    X_dash = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])
    
    # Ajuste de tipos para o XGBoost (Categorical Nativo)
    for col in X_dash.select_dtypes(include=['object']).columns:
        X_dash[col] = X_dash[col].astype("category")

    # Predição em Massa
    probas = model.predict_proba(X_dash)[:, 1]
    df_raw['Churn_Probability'] = probas
    
    # Segmentação
    high_risk_df = df_raw[df_raw['Churn_Probability'] >= threshold].copy()
    avg_churn = df_raw['Churn_Probability'].mean()

    # --- DASHBOARD LAYOUT ---
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Total de Clientes Analisados", f"{len(df_raw)}")
    with kpi2:
        delta = (len(high_risk_df) / len(df_raw)) * 100
        st.metric("Clientes em Alto Risco", f"{len(high_risk_df)}", f"{delta:.1f}% da base")
    with kpi3:
        st.metric("Probabilidade Média de Evasão", f"{avg_churn:.2%}")

    st.markdown("---")

    # --- ANÁLISE SHAP (NOVO) ---
    if show_shap:
        st.header("🔍 Explicabilidade SHAP")
        st.markdown("Entenda quais features contribuem para o risco de churn de cada cliente.")
        
        # Preparar dados para SHAP
        X_shap = X_dash.copy()
        
        # Converter categorias para numérico (SHAP não aceita categorias diretamente)
        for col in X_shap.select_dtypes(include=['category']).columns:
            X_shap[col] = X_shap[col].cat.codes
        
        # Calcular SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # Selecionar cliente para análise detalhada
        col_shap1, col_shap2 = st.columns([1, 2])
        
        with col_shap1:
            st.subheader("Cliente para Análise")
            # Top 10 clientes de maior risco
            top_risk_ids = high_risk_df.nlargest(10, 'Churn_Probability')['Customer_ID'].tolist()
            selected_customer = st.selectbox("Selecione um cliente de alto risco:", top_risk_ids)
            
            # Índice do cliente selecionado
            customer_idx = df_raw[df_raw['Customer_ID'] == selected_customer].index[0]
            
            st.write(f"**Probabilidade de Churn:** {df_raw.loc[customer_idx, 'Churn_Probability']:.2%}")
        
        with col_shap2:
            st.subheader(f"Contribuição das Features")
            
            # Waterfall plot para o cliente selecionado
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[customer_idx],
                    base_values=explainer.expected_value,
                    data=X_shap.iloc[customer_idx],
                    feature_names=X_shap.columns.tolist()
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        
        # Summary plot global
        st.subheader("Importância Global das Features")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("---")

    # --- GRÁFICOS ---
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.subheader("📊 Distribuição de Probabilidade")
        fig_dist = px.histogram(
            df_raw, x="Churn_Probability", nbins=50,
            color_discrete_sequence=['#1f77b4'],
            labels={'Churn_Probability': 'Probabilidade de Churn'}
        )
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_graph2:
        st.subheader("📍 Risco por Região")
        region_risk = df_raw.groupby('Region')['Churn_Probability'].mean().reset_index()
        fig_reg = px.bar(region_risk, x='Region', y='Churn_Probability', color='Churn_Probability')
        st.plotly_chart(fig_reg, use_container_width=True)

    # --- A LISTA TÁTICA ---
    st.markdown("### ⚠️ Lista de Alvos: Clientes de Alta Prioridade")
    st.info("Esta lista contém os dados identificáveis para que o time de Customer Success possa realizar o contato.")

    high_risk_display = high_risk_df.sort_values(by='Churn_Probability', ascending=False)

    cols_view = [
        'Customer_ID', 'Churn_Probability', 'Age', 'Gender', 
        'Subscription_Length', 'Monthly_Spend', 'Payment_Method', 'Region'
    ]

    st.dataframe(
        high_risk_display[cols_view].style.format({
            "Churn_Probability": "{:.2%}",
            "Monthly_Spend": "R$ {:.2f}"
        }).background_gradient(subset=['Churn_Probability'], cmap='YlOrRd'),
        use_container_width=True,
        hide_index=True
    )

    # Download dos Alvos
    csv = high_risk_display[cols_view].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exportar Lista de Alvos (CSV)",
        data=csv,
        file_name="clientes_alto_risco.csv",
        mime="text/csv",
    )

else:
    st.warning("⚠️ Aguardando pipeline de dados: Verifique se o modelo foi treinado e o MLflow está acessível em http://mlflow:5000")