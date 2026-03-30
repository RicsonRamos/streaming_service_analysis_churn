import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_metrics(df):
    """Exibe os KPIs principais no topo do dashboard."""
    col1, col2, col3 = st.columns(3)
    
    total_customers = len(df)
    high_risk_count = len(df[df['Risk_Level'] == 'High'])
    avg_churn_prob = df['Churn_Probability'].mean()

    col1.metric("Total de Clientes", f"{total_customers}")
    col2.metric("Clientes em Alto Risco", f"{high_risk_count}", 
                delta=f"{(high_risk_count/total_customers)*100:.1f}%", delta_color="inverse")
    col3.metric("Probabilidade Média", f"{avg_churn_prob:.2%}")

def render_charts(df):
    """Renderiza as visualizações de distribuição de risco."""
    st.subheader("📊 Análise de Distribuição")
    col1, col2 = st.columns(2)

    with col1:
        # Histograma de Probabilidades
        fig_hist = px.histogram(
            df, x="Churn_Probability", 
            nbins=20, 
            title="Distribuição de Probabilidade de Churn",
            color="Risk_Level",
            color_discrete_map={'High': '#EF553B', 'Low': '#636EFA'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Gráfico de Pizza por Nível de Risco
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risco', 'Quantidade']
        fig_pie = px.pie(
            risk_counts, values='Quantidade', names='Risco',
            title="Proporção de Níveis de Risco",
            hole=0.4,
            color='Risco',
            color_discrete_map={'High': '#EF553B', 'Low': '#636EFA'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def render_priority_list(df_high_risk):
    """Exibe a tabela detalhada de clientes que precisam de ação imediata."""
    if df_high_risk.empty:
        st.success("✅ Nenhum cliente identificado em nível de risco crítico.")
    else:
        # Ordena pelos que têm maior probabilidade de sair primeiro
        display_df = df_high_risk.sort_values(by="Churn_Probability", ascending=False)
        
        # Seleciona apenas colunas relevantes para o negócio (ajuste conforme seu dataset)
        cols_to_show = ['customerID', 'Churn_Probability', 'MonthlyCharges', 'tenure']
        # Verifica se as colunas existem antes de filtrar (evita KeyError)
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        
        st.dataframe(
            display_df[cols_to_show].style.background_gradient(subset=['Churn_Probability'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )