import streamlit as st
import plotly.express as px

def render_metrics(df, threshold):
    """Exibe os principais KPIs de negócio."""
    risco_alto = df[df['Nivel_Risco'] == 'Alto']
    receita_risco = risco_alto['Monthly_Spend'].sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Clientes Analisados", f"{len(df):,}")
    m2.metric("Em Alto Risco", f"{len(risco_alto):,}", f"{len(risco_alto)/len(df):.1%}", delta_color="inverse")
    m3.metric("Receita em Risco", f"R$ {receita_risco:,.2f}")

def render_charts(df):
    """Renderiza gráficos usando chaves únicas dinâmicas."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Matriz de Priorização")
        fig = px.scatter(
            df, x="Probabilidade", y="Monthly_Spend", color="Nivel_Risco",
            color_discrete_map={'Alto': '#EF553B', 'Médio': '#FECB52', 'Baixo': '#636EFA'}
        )
        # Usamos o ID do objeto para garantir que a chave nunca colida
        st.plotly_chart(fig, width='stretch', key=f"scatter_{id(fig)}")
        
    with col2:
        st.subheader("📊 Distribuição de Risco")
        fig_pie = px.pie(
            df, names='Nivel_Risco', color='Nivel_Risco',
            color_discrete_map={'Alto': '#EF553B', 'Médio': '#FECB52', 'Baixo': '#636EFA'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, width='stretch', key=f"pie_{id(fig_pie)}")

def render_feature_importance(fi_df):
    """Exibe o gráfico de barras com as causas do Churn."""
    st.subheader("🧬 DNA do Churn")
    fig_fi = px.bar(
        fi_df.head(10), 
        x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_fi, width='stretch', key="bar_importance_unique")

def render_simulator(model, service):
    """
    Cria um formulário na barra lateral para simular novos cenários.
    """
    with st.sidebar.expander("🧪 Simulador de Estratégia", expanded=False):
        st.write("Altere os dados para ver o novo risco:")
        
        # Inputs baseados nas suas features (excluindo o Satisfaction_Score viciado)
        age = st.number_input("Idade", 18, 90, 30)
        tenure = st.slider("Meses de Contrato", 1, 72, 12)
        spend = st.number_input("Gasto Mensal (R$)", 10.0, 500.0, 50.0)
        engagement = st.slider("Score de Engajamento", 1, 10, 5)
        
        # Categorias
        region = st.selectbox("Região", ["North", "South", "East", "West", "Central"])
        method = st.selectbox("Pagamento", ["Credit Card", "PayPal", "Debit Card"])
        gender = st.selectbox("Gênero", ["Male", "Female"])

        # Botão de Simulação
        if st.button("Calcular Novo Risco"):
            data = {
                'Age': age, 'Subscription_Length': tenure, 'Monthly_Spend': spend,
                'Engagement_Score': engagement, 'Gender': gender, 
                'Region': region, 'Payment_Method': method,
                # Mantemos os outros campos zerados ou com médias para não quebrar o shape
                'Support_Tickets_Raised': 0, 'Satisfaction_Score': 5, 
                'Last_Activity': 15, 'Estimated_LTV': spend * 12
            }
            
            prob = service.predict_single_customer(model, data)
            
            # Resultado Visual
            st.markdown("---")
            st.metric("Risco Simulado", f"{prob:.1%}")
            
            if prob > 0.7:
                st.error("Risco Crítico! Recomenda-se cupom de desconto.")
            else:
                st.success("Cliente Saudável.")