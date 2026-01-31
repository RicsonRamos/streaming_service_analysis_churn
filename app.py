import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

from src.config.loader import ConfigLoader

cfg = ConfigLoader().load_all()

RAW_PATH = cfg["paths"]["data"]["raw"]
PROCESSED_PATH = cfg["paths"]["data"]["processed"]
MODEL_PATH = cfg["paths"]["models"]["churn_model"]


# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Radar de Churn - Streaming", layout="wide", page_icon="🛡️")

# 2. CARREGAMENTO DE ASSETS
@st.cache_resource
def load_assets():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSED_PATH):
        return None, None, MODEL_PATH, PROCESSED_PATH

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_PATH)

    return model, df, MODEL_PATH, PROCESSED_PATH


# Carrega os dados
model, df, m_path, d_path = load_assets()   

# VERIFICAÇÃO DE ERRO MELHORADA
if model is None or df is None:
    st.error("❌ Arquivos necessários não encontrados.")
    st.info(f"O Docker está procurando nestes locais:")
    st.code(f"Modelo: {m_path}\nDados: {d_path}")
    st.warning("Dica: Certifique-se de que as pastas 'models' e 'data' foram copiadas no Dockerfile e que os nomes estão em minúsculo.")
    st.stop()

# --- DAQUI PARA BAIXO O CÓDIGO SEGUE NORMALMENTE ---

# 3. BARRA LATERAL COM EXPLICAÇÕES
with st.sidebar:
    st.header("⚙️ Configurações de Negócio")
    
    cost_per_client = st.number_input(
        "Custo de Retenção (R$)", 
        value=50.0,
        help="Quanto a empresa gasta (descontos, marketing, ligações) para tentar manter um cliente de risco."
    )
    
    avg_clv = st.number_input(
        "LTV Médio (R$)", 
        value=1200.0,
        help="Receita total média que um cliente gera durante todo o tempo que fica na plataforma."
    )
    
    threshold = st.slider(
        "Ponto de Corte de Risco (%)", 
        0, 100, 70,
        help="Define a sensibilidade do modelo. Acima deste valor, o cliente entra na lista de 'Alto Risco'."
    ) / 100

    st.markdown("---")
    st.info("💡 **Dica:** Ajuste o Ponto de Corte para focar apenas nos casos mais críticos.")

# 4. PROCESSAMENTO
# Certifique-se de que os nomes abaixo batem com o CSV
features_modelo = [
    'Age', 'Subscription_Length', 'Support_Tickets_Raised', 
    'Satisfaction_Score', 'Monthly_Spend', 'Estimated_LTV', 'Engagement_Score',
    'Gender', 'Region', 'Payment_Method'
]

# Proteção para garantir que as colunas existem
X_input = df[features_modelo].copy()
df['Probabilidade'] = model.predict_proba(X_input)[:, 1]
df['Nivel_Risco'] = df['Probabilidade'].apply(
    lambda x: 'Alto' if x >= threshold else ('Médio' if x >= 0.4 else 'Baixo')
)

# 5. HEADER E MÉTRICAS
st.title("🛡️ Dashboard de Predição de Churn")

clientes_em_risco = df[df['Probabilidade'] >= threshold].copy()
receita_risco = clientes_em_risco['Monthly_Spend'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clientes", f"{len(df)}")
col2.metric("Em Alto Risco", f"{len(clientes_em_risco)}", f"{len(clientes_em_risco)/len(df):.1%}", delta_color="inverse")
col3.metric("Receita em Risco", f"R$ {receita_risco:,.2f}")

sucesso_est = 0.30 
roi_val = (len(clientes_em_risco) * sucesso_est * avg_clv) - (len(clientes_em_risco) * cost_per_client)
col4.metric("ROI Potencial", f"R$ {roi_val:,.2f}")

st.markdown("---")

# 6. GRÁFICOS
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎯 Matriz de Priorização")
    df['LTV_Plot'] = (df['Monthly_Spend'] * 12).abs()
    
    fig = px.scatter(
        df, x="Probabilidade", y="LTV_Plot", color="Nivel_Risco",
        size="LTV_Plot", hover_data=['Age', 'Region', 'Payment_Method'],
        color_discrete_map={'Alto': '#ef553b', 'Médio': '#fecb52', 'Baixo': '#636efa'},
        labels={'Probabilidade': 'Risco de Churn (0-1)', 'LTV_Plot': 'Valor Anual (LTV)'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📊 Distribuição de Risco")
    fig_pie = px.pie(df, names='Nivel_Risco', color='Nivel_Risco',
                     color_discrete_map={'Alto': '#ef553b', 'Médio': '#fecb52', 'Baixo': '#636efa'},
                     hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# 7. LISTA DE AÇÃO
st.subheader("📋 Lista de Clientes para Ação Imediata")
df_display = clientes_em_risco.sort_values(by=['Probabilidade', 'Monthly_Spend'], ascending=False)
cols_view = ['Probabilidade', 'Monthly_Spend', 'Subscription_Length', 'Region', 'Payment_Method']

st.dataframe(
    df_display[cols_view].style.format({'Probabilidade': '{:.1%}', 'Monthly_Spend': 'R$ {:.2f}'}),
    use_container_width=True
)

# 8. EXPORTAÇÃO
st.download_button(
    "📥 Baixar Lista para Comercial", 
    data=df_display.to_csv(index=False).encode('utf-8'), 
    file_name='prioridade_churn.csv', 
    mime='text/csv'
)