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
            hover_data=['Age', 'Subscription_Length'],
            labels={"Probability": "Churn Probability", "Monthly_Spend": "Monthly Spend ($)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(
            df, names='Risk_Level', color='Risk_Level',
            color_discrete_map=color_map,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def render_explainability(model, df_sample: pd.DataFrame):
    """
    Renders SHAP-based explainability chart.
    Expects df_sample to contain exactly the features the model was trained on.
    """
    st.subheader("Churn Drivers (XAI DNA)")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_sample)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, df_sample, plot_type="bar", show=False)
        
        # Professional styling
        plt.title("Feature Impact on Churn Decision", loc='left', fontsize=12)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Feature importance unavailable for current view: {e}")

def render_simulator(model, service):
    """Creates a sidebar form to simulate churn probability using raw inputs."""
    with st.sidebar.expander("Strategy Simulator", expanded=False):
        st.write("Modify attributes to see risk impact:")

        age = st.number_input("Age", 18, 90, 30)
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        spend = st.number_input("Monthly Spend ($)", 10.0, 500.0, 85.0)
        tickets = st.slider("Support Tickets", 0, 20, 2)
        region = st.selectbox("Region", ["Germany", "France", "Spain", "North", "South"])
        method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Debit Card"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        if st.button("Run Simulation", use_container_width=True):
            # Input dictionary for the service
            raw_data = {
                'Age': age, 
                'Subscription_Length': tenure, 
                'Monthly_Spend': spend,
                'Support_Tickets_Raised': tickets,
                'Gender': gender, 
                'Region': region, 
                'Payment_Method': method
            }

            # Predict using the internal service logic (including Feature Engineering)
            prob = service.predict_single_customer(model, raw_data)
            
            st.divider()
            st.metric("Simulated Risk", f"{prob:.1%}")

            if prob > 0.7:
                st.error("🚨 High Risk: Retention offer recommended.")
            elif prob > 0.4:
                st.warning("⚠️ Medium Risk: Monitor activity.")
            else:
                st.success("✅ Stable: Low probability of churn.")
