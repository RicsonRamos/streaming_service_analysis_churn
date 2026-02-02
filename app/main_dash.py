"""
Main Dashboard Orchestrator for Churn Radar.
Point of entry: streamlit run main_dash.py
"""

import streamlit as st
from src.config.loader import ConfigLoader
from src.app.services import ChurnService
import src.app.components as ui

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
    Uses cached resource to avoid reloading the model on every interaction.
    """
    cfg_loader = ConfigLoader()
    cfg = cfg_loader.load_all()

    # Sincronizado com a nova estrutura do paths.yaml
    service = ChurnService(
        model_path=cfg["artifacts"]["current_model"],
        processed_path=cfg["data"]["final_dataset"]
    )

    model, df_history = service.load_assets()
    return cfg, service, model, df_history

# 2. ASSET LOADING
cfg, service, model, df_history = initialize_app()

if model is None or df_history is None:
    st.error("Critical Error: Missing model artifact or processed data. Run training pipeline first.")
    st.stop()

# 3. SIDEBAR & SIMULATOR
with st.sidebar:
    st.header("⚙️ Dashboard Settings")
    threshold = st.slider(
        "Risk Threshold", 
        0.0, 1.0, 
        cfg["business_logic"]["risk_threshold"], # Default from base.yaml
        help="Adjust the probability cutoff for High Risk classification."
    )
    st.divider()
    ui.render_simulator(model, service)

# 4. DATA PROCESSING
# Bulk inference for historical data
df_processed = service.predict_churn(model, df_history, threshold)

# 5. UI LAYOUT
st.title("🛡️ Churn Radar - Predictive Analytics")

# Section 1: Business Overview
ui.render_metrics(df_processed)
st.divider()

# Section 2: Visual Insights
ui.render_charts(df_processed)
st.divider()

# Section 3: Explainability (SHAP)
# We pass a sample of the processed features used by the model
feature_cols = service.expected_features # Obtained from model artifact
ui.render_explainability(model, df_processed[feature_cols].head(20))

# Section 4: Operational Data
st.subheader("📋 Priority Retention List")
high_risk_list = df_processed[df_processed['Risk_Level'] == 'High'].sort_values('Probability', ascending=False)

if not high_risk_list.empty:
    st.dataframe(high_risk_list, use_container_width=True)
else:
    st.info("No high-risk customers identified with current settings.")
