import streamlit as st
import plotly.express as px
import pandas as pd

def render_metrics(df: pd.DataFrame):
    """
    Displays main business KPIs based on customer risk levels.

    Args:
        df (pd.DataFrame): Processed dataframe containing 'Risk_Level' and 'Monthly_Spend'.
    """
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
    """
    Renders risk prioritization matrix and risk distribution charts.

    Args:
        df (pd.DataFrame): Dataframe with 'Probability', 'Monthly_Spend', and 'Risk_Level'.
    """
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

def render_feature_importance(fi_df: pd.DataFrame):
    """
    Displays feature importance bar chart to identify churn drivers.

    Args:
        fi_df (pd.DataFrame): Dataframe containing 'Feature' and 'Importance' columns.
    """
    st.subheader("Churn Drivers (DNA)")
    fig_fi = px.bar(
        fi_df.head(10), 
        x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Reds',
        labels={"Importance": "SHAP Impact", "Feature": "Attribute"}
    )
    st.plotly_chart(fig_fi, use_container_width=True, key="bar_importance_unique")

def render_simulator(model, service):
    """
    Creates a sidebar form to simulate churn probability for new customer scenarios.

    Args:
        model: Trained machine learning model object.
        service: Service class instance containing prediction logic.
    """
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
