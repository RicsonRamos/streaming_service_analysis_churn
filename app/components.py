import streamlit as st
import plotly.express as px

def render_metrics(df):
    """Exibe os KPIs principais no topo do dashboard."""
    col1, col2, col3 = st.columns(3)
    total = len(df)
    high_risk = len(df[df['Risk_Level'] == 'High'])
    
    col1.metric("Total de Clientes", f"{total}")
    col2.metric("Clientes em Alto Risco", f"{high_risk}", 
                delta=f"{(high_risk/total)*100:.1f}% da base", delta_color="inverse")
    col3.metric("Probabilidade Média", f"{df['Churn_Probability'].mean():.2%}")

def render_charts(df):
    """Renderiza as visualizações de distribuição de risco."""
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="Churn_Probability", color="Risk_Level", title="Distribuição de Risco"), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df, x="Region", y="Churn_Probability", title="Risco por Região"), use_container_width=True)

def render_priority_list(df_high_risk):
    """Exibe a tabela detalhada de clientes que precisam de ação imediata."""
    if df_high_risk.empty:
        st.success("✅ Nenhum cliente identificado em nível de risco crítico.")
    else:
        # Ordenação por gravidade
        display_df = df_high_risk.sort_values(by="Churn_Probability", ascending=False)
        cols = ['Customer_ID', 'Churn_Probability', 'Monthly_Spend', 'Subscription_Length']
        
        st.dataframe(
            display_df[cols].style.background_gradient(subset=['Churn_Probability'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )