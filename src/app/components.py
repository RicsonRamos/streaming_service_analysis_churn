import streamlit as st
import plotly.express as px

def render_metrics(df, threshold):
    risco_alto = df[df['Nivel_Risco'] == 'Alto']
    receita_risco = risco_alto['Monthly_Spend'].sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Clientes Analisados", len(df))
    m2.metric("Em Alto Risco", len(risco_alto), f"{len(risco_alto)/len(df):.1%}", delta_color="inverse")
    m3.metric("Receita em Risco", f"R$ {receita_risco:,.2f}")

def render_charts(df):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Matriz de Priorização")
        fig = px.scatter(df, x="Probabilidade", y="Monthly_Spend", color="Nivel_Risco",
                         color_discrete_map={'Alto': '#EF553B', 'Médio': '#FECB52', 'Baixo': '#636EFA'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📊 Distribuição")
        fig_pie = px.pie(df, names='Nivel_Risco', color='Nivel_Risco',
                         color_discrete_map={'Alto': '#EF553B', 'Médio': '#FECB52', 'Baixo': '#636EFA'})
        st.plotly_chart(fig_pie, use_container_width=True)