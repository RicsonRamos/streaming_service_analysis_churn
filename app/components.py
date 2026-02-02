"""
UI Components and visualization logic for the Churn Radar Dashboard.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import shap

def render_metrics(df: pd.DataFrame):
    """Displays main business KPIs based on customer risk levels."""
    high_risk_segment = df[df['Risk_Level'] == 'High']
    revenue_at_risk = high_risk_segment['Monthly_Spend'].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Analyzed Customers", f"{len(df):,}")
    m2.metric(
        "High Risk Customers", 
        f"{len(high_risk_segment):,}", 
        f"{len(high_risk_segment)/len(df):.1%}", 
        delta_color="inverse"
    )
    m3.metric("Revenue at Risk", f"USD {revenue_at_risk:,.2f}")

def render_charts(df: pd.DataFrame):
    """Renders risk prioritization matrix and risk distribution charts."""
    col1, col2 = st.columns([2, 1])
    color_map = {'High': '#EF553B', 'Medium': '#FECB52', 'Low': '#636EFA'}

    with col1:
        st.subheader("Prioritization Matrix")
        fig_scatter = px.scatter(
            df, x="Probability", y="Monthly_Spend", color="Risk_Level",
            color_discrete_map=color_map,
            labels={"Probability": "Churn Probability", "Monthly_Spend": "Monthly Spend ($)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{id(fig_scatter)}")

    with col2:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(
            df, names='Risk_Level', color='Risk_Level',
            color_discrete_map=color_map,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{id(fig_pie)}")

def render_explainability(model, df_sample: pd.DataFrame):
    """
    Renders SHAP-based explainability chart with professional color coding.
    """
    st.subheader("Churn Drivers (XAI)")
    
    # 1. Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_sample)
    
    # 2. Style configuration
    COLOR = "#5A5A5A" 
    plt.rcParams.update({'text.color': COLOR, 'axes.labelcolor': COLOR, 'xtick.color': COLOR, 'ytick.color': COLOR})
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 3. Generate SHAP Plot
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=None, 
            feature_names=df_sample.columns.tolist()
        ), 
        max_display=10, 
        show=False
    )

    # 4. Professional Color Sync & Label Cleanup
    blue_retention = "#008bfb"
    red_churn = "#ff0051"

    for patch in ax.patches:
        rgb = patch.get_facecolor()
        patch.set_facecolor(blue_retention if rgb[0] < rgb[2] else red_churn)

    for text in ax.texts:
        t_val = text.get_text()
        text.set_color(blue_retention if '-' in t_val else red_churn if '+' in t_val else COLOR)

    ax.grid(False)
    ax.set_xlabel("")
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig, clear_figure=True, transparent=True)
    plt.close(fig)

def render_simulator(model, service):
    """Creates a sidebar form to simulate churn probability."""
    with st.sidebar.expander("Strategy Simulator", expanded=False):
        st.write("Modify attributes to calculate risk:")

        age = st.number_input("Age", 18, 90, 30)
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        spend = st.number_input("Monthly Spend ($)", 10.0, 500.0, 50.0)
        engagement = st.slider("Engagement Score", 1, 10, 5)
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Debit Card"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        if st.button("Run Simulation", use_container_width=True):
            data = {
                'Age': age, 'Subscription_Length': tenure, 'Monthly_Spend': spend,
                'Engagement_Score': engagement, 'Gender': gender, 
                'Region': region, 'Payment_Method': method,
                'Support_Tickets_Raised': 0, 'Satisfaction_Score': 5, 
                'Last_Activity': 15, 'Estimated_LTV': spend * 12
            }

            prob = service.predict_single_customer(model, data)
            st.markdown("---")
            st.metric("Simulated Risk", f"{prob:.1%}")

            if prob > 0.7:
                st.error("Critical Risk: Immediate retention offer required.")
            else:
                st.success("Healthy Profile: Low churn probability.")
