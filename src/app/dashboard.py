import streamlit as st
from src.config.loader import ConfigLoader
from src.app.services import ChurnService
import src.app.components as ui

# 1. Setup
st.set_page_config(page_title="Radar de Churn", layout="wide")
cfg = ConfigLoader().load_all()

service = ChurnService(
    model_path=cfg["paths"]["models"]["churn_model"],
    processed_path=cfg["paths"]["data"]["processed"]
)

# 2. Data Loading
model, df_raw = service.load_assets()

if model is None or df_raw is None:
    st.error("❌ Erro ao carregar ativos. Verifique os caminhos no config.")
    st.stop()

# 3. Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    threshold = st.slider("Corte de Risco", 0, 100, 70) / 100

# 4. Core Logic
df_processed = service.predict_churn(model, df_raw, threshold)

# 5. UI Rendering
st.title("🛡️ Radar de Churn - Streaming")
ui.render_metrics(df_processed, threshold)
st.markdown("---")
ui.render_charts(df_processed)

st.subheader("📋 Lista de Ação")
st.dataframe(df_processed[df_processed['Nivel_Risco'] == 'Alto'].sort_values('Probabilidade', ascending=False))