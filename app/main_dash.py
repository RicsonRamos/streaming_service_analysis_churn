"""
Main Dashboard Orchestrator for Churn Radar.
Point of entry: streamlit run main_dash.py

This module coordinates configuration loading, service initialization, 
and UI rendering using a modular component-based architecture.
"""
import sys
import os
from pathlib import Path


root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)


import streamlit as st
import pandas as pd
from src.config.loader import ConfigLoader
from app.services import ChurnService
import app.components as ui

# 1. GLOBAL PAGE SETUP
st.set_page_config(
    page_title="Churn Radar | Predictive Insights",
    layout="wide",
    page_icon="🛡️"
)

@st.cache_resource
def initialize_app():
    """
    Initializes core application configurations and model assets.
    Uses cached resource to avoid reloading the model and data on every rerun.
    
    Returns:
        tuple: (config_dict, churn_service_instance, model_object, history_df)
    """
    try:
        cfg_loader = ConfigLoader()
        cfg = cfg_loader.load_all()

        # Agora passamos o 'cfg' como o terceiro argumento exigido pelo __init__
        service = ChurnService(
            model_path=cfg["artifacts"]["current_model"],
            processed_path=cfg["data"]["final_dataset"],
            cfg=cfg  # <--- ADICIONE ESTA LINHA
        )

        model, df_history = service.load_assets()
        
        if model is None or df_history is None:
            st.error("Model or Data Assets could not be loaded. Check your paths.")
            st.stop()

        return cfg, service, model, df_history
        
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        st.stop()

# 2. ASSET LOADING
cfg, service, model, df_history = initialize_app()

if model is None or df_history is None:
    st.error("Critical Error: Model artifact or processed data not found. Check your 'configs/' paths.")
    st.stop()

# 3. SIDEBAR: SETTINGS & SIMULATOR
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    
    # Business logic threshold for risk classification
    default_threshold = cfg.get("business_logic", {}).get("risk_threshold", 0.5)
    threshold = st.slider(
        "Risk Threshold", 
        0.0, 1.0, 
        default_threshold,
        help="Adjust the probability cutoff for High Risk classification."
    )
    
    st.divider()
    
    # Strategy Simulator (Local Explanation Integrated)
    ui.render_simulator(model, service)

# 4. MAIN INTERFACE LAYOUT
st.title("🛡️ Churn Radar - Predictive Analytics")
st.caption(f"Model version: {cfg['artifacts']['current_model'].split('/')[-1]}")

# Execute bulk inference on historical data for dashboarding
df_processed = service.predict_churn(model, df_history, threshold)

# --- UI SECTIONS ---

# Section 1: Business Overview (KPIs)
ui.render_metrics(df_processed)
st.divider()

# Section 2: Visual Insights (Trends & Distributions)
ui.render_charts(df_processed)
st.divider()

# --- Section 3: Explainability (Global SHAP) ---
st.subheader("Model Interpretability (Global SHAP)")

with st.expander("View Global Feature Importance", expanded=False):
    with st.spinner("Calculating SHAP values..."):
        # USANDO A MANEIRA INTELIGENTE: 
        # O serviço cuida de alinhar, processar e calcular.
        # Limitamos a 100 amostras para não travar o Dashboard.
        shap_values, X_processed, expected_value = service.get_shap_explanation(model, df_history.head(100))
        
        # Agora passamos os dados JÁ PROCESSADOS para o componente de UI
        # Isso garante que os nomes das colunas apareçam no gráfico.
        ui.render_explainability(shap_values, X_processed)

# --- Section 4: Operational Data (Actionable List) ---
st.subheader("📋 Priority Retention List")

# Filtragem direta no DataFrame que já foi processado pelo serviço
high_risk_list = df_processed[df_processed['Risk_Level'] == 'High'].sort_values(
    by='Probability', 
    ascending=False
)

# Renderiza a tabela usando seu componente de UI
ui.render_priority_list(high_risk_list)

# 5. FOOTER
st.markdown("---")
st.caption("Churn Radar v2.0 | Data-Driven Retention Strategy")