"""
Main entry point for the Churn Radar Streamlit application.
Handles asset loading, sidebar configurations, and layout orchestration.
"""

import streamlit as st
from src.config.loader import ConfigLoader
from src.app.services import ChurnService
import src.app.components as ui

# 1. Page Configuration
st.set_page_config(
    page_title="Churn Radar | Predictive Insights", 
    layout="wide", 
    page_icon="🛡️"
)

# 2. Initialization & Asset Loading
@st.cache_resource
def initialize_app():
    """Initializes configurations and loads model assets."""
    cfg = ConfigLoader().load_all()
    service = ChurnService(
        model_path=cfg["paths"]["models"]["churn_model"],
        processed_path=cfg["paths"]["data"]["processed"]
    )
    model, df_raw = service.load_assets()
    return service, model, df_raw

service, model, df_raw = initialize_app()

if model is None or df_raw is None:
    st.error("Error loading assets. Please check configuration paths.")
    st.stop()

# 3. Sidebar - Global Controls
with st.sidebar:
    st.header("Global Settings")
    # Sensitivity threshold for risk classification
    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.70, help="Adjust sensitivity for High Risk classification.")

# 4. Data Processing
# Run inference on the raw dataset
df_processed = service.predict_churn(model, df_raw, threshold)
# Fetch model-specific feature importance (SHAP)
fit_df = service.get_feature_importance(model)

# 5. Dashboard Layout
st.title("Churn Radar - Streaming Service Analytics")

# Top-level KPIs
ui.render_metrics(df_processed)
st.divider()

# Core Visualizations (Matrix & Distribution)
ui.render_charts(df_processed)
st.divider()

# Explainability Section (SHAP Drivers)
ui.render_feature_importance(fit_df)
st.divider()

# Interactive Scenario Simulator
ui.render_simulator(model, service)

# 6. Operational Action List
st.subheader("Priority Action List")
high_risk_customers = df_processed[df_processed['Risk_Level'] == 'High'].sort_values('Probability', ascending=False)

if not high_risk_customers.empty:
    st.dataframe(high_risk_customers, use_container_width=True)
else:
    st.info("No high-risk customers identified with the current threshold.")
