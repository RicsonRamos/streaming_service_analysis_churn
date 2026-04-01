import plotly.express as px
import streamlit as st


def render_metrics(df):
    """
    Exibe os KPIs principais no topo do dashboard.

    Essas métricas incluem a quantidade total de clientes, a quantidade de clientes
    em alto risco e a probabilidade média de churn entre todos os clientes.
    """
    col1, col2, col3 = st.columns(3)
    total = len(df)
    high_risk = len(df[df["Risk_Level"] == "High"])

    # Total de Clientes
    col1.metric("Total de Clientes", f"{total}")

    # Clientes em Alto Risco
    col2.metric(
        "Clientes em Alto Risco",
        f"{high_risk}",
        delta=f"{(high_risk/total)*100:.1f}% da base",
        delta_color="inverse",
    )

    # Probabilidade Média de Churn
    col3.metric("Probabilidade Média", f"{df['Churn_Probability'].mean():.2%}")


def render_charts(df):
    """
    Renderiza as visualizações de distribuição de risco.

    Essa função renderiza dois gráficos:
    1. Um histograma de distribuição de risco, com cores representando os níveis
        de risco (baixo, médio, alto).
    2. Um gráfico de caixa com a distribuição de risco por região.

    :param df: DataFrame com as informações de risco e região.
    """
    col1, col2 = st.columns(2)
    with col1:
        # Histograma de distribuição de risco
        st.plotly_chart(
            px.histogram(
                df,
                x="Churn_Probability",
                color="Risk_Level",
                title="Distribuição de Risco",
            ),
            use_container_width=True,
        )
    with col2:
        # Gráfico de caixa com a distribuição de risco por região
        st.plotly_chart(
            px.box(df, x="Region", y="Churn_Probability", title="Risco por Região"),
            use_container_width=True,
        )


def render_priority_list(df_high_risk):
    """
    Exibe a tabela detalhada de clientes que precisam de ação imediata.

    Essa função renderiza uma tabela com as informações de cada cliente em alto risco,
    ordenada por probabilidade de churn. Caso nenhuma cliente esteja em risco, um
    aviso de sucesso é exibido.

    :param df_high_risk: DataFrame com as informações de clientes em alto risco.
    """
    if df_high_risk.empty:
        st.success("✅ Nenhum cliente identificado em nível de risco crítico.")
    else:
        # Ordenação por gravidade
        display_df = df_high_risk.sort_values(by="Churn_Probability", ascending=False)
        cols = [
            "Customer_ID",
            "Churn_Probability",
            "Monthly_Spend",
            "Subscription_Length",
        ]

        st.dataframe(
            display_df[cols].style.background_gradient(subset=["Churn_Probability"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )
