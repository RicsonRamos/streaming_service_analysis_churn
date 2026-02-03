"""
UI Components and visualization logic for the Churn Radar Dashboard.
"""
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.utils.styles import CHART_THEME, apply_chart_style

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

def render_simulator(model, service):
    """Creates a sidebar form to simulate churn probability using raw inputs."""
    with st.sidebar.expander("Strategy Simulator", expanded=False):
        st.write("Modify attributes to see risk impact:")

        # Inputs Básicos
        age = st.number_input("Age", 18, 90, 30)
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        spend = st.number_input("Monthly Spend ($)", 10.0, 500.0, 85.0)
        
        # NOVOS INPUTS OBRIGATÓRIOS (Para evitar KeyError)
        ltv = st.number_input("Estimated LTV ($)", 0.0, 10000.0, 1200.0)
        engagement = st.slider("Engagement Score (0-100)", 0, 100, 50)
        
        tickets = st.slider("Support Tickets", 0, 20, 2)
        
        # Alinhamento com as categorias do modelo
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Debit Card"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        if st.button("Run Simulation", use_container_width=True):
            # Dicionário COMPLETO para o service
            raw_data = {
                'Age': age, 
                'Subscription_Length': tenure, 
                'Monthly_Spend': spend,
                'Estimated_LTV': ltv,           # Adicionado
                'Engagement_Score': engagement, # Adicionado
                'Support_Tickets_Raised': tickets,
                'Gender': gender, 
                'Region': region, 
                'Payment_Method': method
            }

            # Predict (Agora o services.py terá todos os dados para os cálculos)
            try:
                prob = service.predict_single_customer(model, raw_data)
                
                st.divider()
                st.metric("Simulated Risk", f"{prob:.1%}")

                if prob > 0.7:
                    st.error("🚨 High Risk: Retention offer recommended.")
                elif prob > 0.4:
                    st.warning("⚠️ Medium Risk: Monitor activity.")
                else:
                    st.success("✅ Stable: Low probability of churn.")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

def render_explainability(shap_values, X_processed):
    """
    Renders a minimalist SHAP bar chart for the top 3 global features.
    
    This component filters the importance matrix to highlight only the 
    primary churn drivers, reducing cognitive load for business users.
    It applies a custom theme for visual consistency across the dashboard.

    Args:
        shap_values: The SHAP values object from the explainer.
        X_processed (pd.DataFrame): The preprocessed feature matrix 
            used for inference.
    """
    st.write("### Top 3 Churn Drivers")

    try:
        # Friendly naming mapping for business clarity
        friendly_names = {
            'Support_Tickets_Raised': 'Support Interactions',
            'Monthly_Spend': 'Monthly Billing',
            'Subscription_Length': 'Tenure (Months)',
            'Age': 'Customer Age',
            'Engagement_Score': 'App Engagement'
        }

        # Calculate mean absolute SHAP values to identify top drivers
        if hasattr(shap_values, 'values'):
            mean_abs_shap = np.abs(shap_values.values).mean(0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(0)

        # Force top 3 feature selection
        top_3_indices = np.argsort(mean_abs_shap)[-3:]
        
        # Create a sliced Explanation object to avoid the 'Sum of other features' bar
        exp_top3 = shap.Explanation(
            values=shap_values.values[:, top_3_indices] if hasattr(shap_values, 'values') else shap_values[:, top_3_indices],
            data=X_processed.iloc[:, top_3_indices].values,
            feature_names=[friendly_names.get(col, col) for col in X_processed.columns[top_3_indices]]
        )

        # Plot setup with centralized styling
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=CHART_THEME["bg_color"])
        
        # Render SHAP bars with primary brand color
        shap.plots.bar(exp_top3, show=False)

        for patch in ax.patches:
            patch.set_facecolor(CHART_THEME["primary_color"])
            patch.set_edgecolor(CHART_THEME["bg_color"])

        # Apply global styling (removes spines, sets fonts and colors)
        ax = apply_chart_style(ax)

        ax.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.title(
            "Primary drivers of customer loss", 
            loc='left', 
            fontsize=CHART_THEME["font_size_title"],
            color=CHART_THEME["text_color"],
            pad=15
        )

        st.pyplot(fig, transparent=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"UI Error: Failed to render explainability component: {e}")

def render_priority_list(high_risk_df: pd.DataFrame):
    """
    Renders a formatted table of customers with high churn risk.
    """
    st.write("### 🎯 High-Risk Customers for Retention")
    
    if high_risk_df.empty:
        st.success("No high-risk customers identified with current threshold. Good job!")
        return

    # Selecionando apenas o que importa para o operacional ver
    view_cols = [
        'Customer_ID', 'Probability', 'Risk_Level', 
        'Monthly_Spend', 'Subscription_Length', 'Support_Tickets_Raised'
    ]
    
    # Verificação defensiva de colunas
    cols_to_show = [c for c in view_cols if c in high_risk_df.columns]
    
    # Formatação visual
    formatted_df = high_risk_df[cols_to_show].copy()
    formatted_df['Probability'] = (formatted_df['Probability'] * 100).map("{:.1f}%".format)
    
    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.TextColumn("Churn Probability"),
            "Risk_Level": st.column_config.TextColumn("Priority"),
            "Monthly_Spend": st.column_config.NumberColumn("Monthly ($)"),
            "Support_Tickets_Raised": st.column_config.NumberColumn("Tickets")
        }
    )
    
    st.caption(f"Total priority cases: {len(high_risk_df)}")