import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


def run_app(cfg: dict):

    MODEL_PATH = cfg["paths"]["models"]["churn_model"]
    DATA_PATH = cfg["paths"]["data"]["processed"]

    st.set_page_config(
        page_title="Radar de Churn",
        layout="wide",
        page_icon="🛡️"
    )

    @st.cache_resource
    def load_assets():

        if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
            return None, None

        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)

        return model, df

    model, df = load_assets()

    if model is None:
        st.error("Assets not found.")
        st.stop()

    # Features
    features = [
        'Age', 'Subscription_Length', 'Support_Tickets_Raised',
        'Satisfaction_Score', 'Monthly_Spend',
        'Estimated_LTV', 'Engagement_Score',
        'Gender', 'Region', 'Payment_Method'
    ]

    X = df[features]

    threshold = cfg["business"]["churn_threshold"]

    df["Probabilidade"] = model.predict_proba(X)[:, 1]

    df["Nivel_Risco"] = df["Probabilidade"].apply(
        lambda x: "Alto" if x >= threshold else
        ("Médio" if x >= 0.4 else "Baixo")
    )

    # Header
    st.title("🛡️ Churn Dashboard")

    risk_df = df[df["Probabilidade"] >= threshold]

    col1, col2, col3 = st.columns(3)

    col1.metric("Clientes", len(df))
    col2.metric("Alto Risco", len(risk_df))
    col3.metric("Receita em Risco", f"R$ {risk_df['Monthly_Spend'].sum():,.2f}")

    # Plot
    df["LTV_Year"] = df["Monthly_Spend"] * 12

    fig = px.scatter(
        df,
        x="Probabilidade",
        y="LTV_Year",
        color="Nivel_Risco",
        size="LTV_Year"
    )

    st.plotly_chart(fig, use_container_width=True)
