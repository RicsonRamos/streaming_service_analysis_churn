"""
Main Dashboard Orchestrator for Churn Radar.

This module serves as the entry point for the Streamlit application, 
coordinating asset loading, data processing via ChurnService, 
and UI rendering through the components module.
"""

import streamlit as st
from src.config.loader import ConfigLoader
from src.features.feature_engineering import FeatureEngineer
from src.app.services import ChurnService
import src.app.components as ui

# 1. PAGE SETUP
st.set_page_config(
    page_title="Churn Radar | Predictive Insights",
    layout="wide",
    page_icon="🛡️"
)

@st.cache_resource
def initialize_app():
    """
    Initializes core application configurations and model assets.

    Returns:
        tuple: A tuple containing (ChurnService instance, loaded model, historical DataFrame).
               Returns (None, None, None) if assets are missing.
    """
    cfg_loader = ConfigLoader()
    cfg = cfg_loader.load_all()
    fe = FeatureEngineer(cfg)
    
    service = ChurnService(
        model_path=cfg["paths"]["models"]["churn_model"],
        processed_path=cfg["paths"]["data"]["processed"]
    )
    
    model, df_history = service.load_assets()
    return service, model, df_history

# 2. ASSET LOADING
service, model, df_history = initialize_app()

if model is None or df_history is None:
    st.error("Critical Error: Model or Data assets not found. Check your config and paths.")
    st.stop()

# 3. SIDEBAR - SETTINGS & SIMULATOR
with st.sidebar:
    st.header("Settings")
    # Sensitivity threshold for risk classification
    threshold = st.slider(
        "Risk Sensitivity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.70,
        help="Adjust the probability cutoff for High Risk classification."
    )
    st.divider()
    
    ui.render_simulator(model, service)

# 4. DATA PROCESSING
# Performs batch inference and risk categorization
df_processed = service.predict_churn(model, df_history, threshold)

# 5. MAIN UI RENDERING
st.title("Churn Radar - Streaming Service Analytics")

with st.expander("📖 Quick Guide: How to use this Dashboard"):
    """Displays user guidance on dashboard functionality."""
    st.markdown("""
    This system uses **XGBoost AI** to predict customer churn.
    - **Global KPIs:** Overall health of the customer base.
    - **Prioritization Matrix:** Identifies high-value customers at risk.
    - **XAI DNA:** Explains why the model classifies a customer as 'at risk'.
    """)

# Section 1: Business Metrics (KPIs)
ui.render_metrics(df_processed)
st.divider()

# Section 2: Visual Discovery (Scatter & Pie charts)
ui.render_charts(df_processed)
st.divider()

# Section 3: Explainable AI (SHAP DNA)
# Explains model drivers using a representative sample of the data
ui.render_explainability(model, df_processed[service.expected_features].head(10))

# Section 4: Operational Data
st.subheader("Priority Retention List")
high_risk_list = df_processed[df_processed['Risk_Level'] == 'High'].sort_values('Probability', ascending=False)

if not high_risk_list.empty:
    st.dataframe(high_risk_list, use_container_width=True)
else:
    st.info("No high-risk customers identified with current threshold settings.")
