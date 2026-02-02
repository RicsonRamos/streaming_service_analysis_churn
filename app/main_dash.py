import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from src.config.loader import ConfigLoader
from src.features.feature_engineering import FeatureEngineer

# 1. CONFIGURAÇÃO E ESTILO
st.set_page_config(page_title="Radar de Churn 2.0", layout="wide", page_icon="🛡️")

@st.cache_resource
def load_assets():
    loader = ConfigLoader()
    cfg = loader.load_all()
    fe = FeatureEngineer(cfg)
    
    # Carregamento do modelo (Artifact)
    model_path = cfg["paths"]["models"]["churn_model"]
    if not Path(model_path).exists():
        st.error(f"Modelo não encontrado em {model_path}. Rode o treino primeiro!")
        st.stop()
        
    artifact = joblib.load(model_path)
    
    # Carregamento da base histórica para o Dashboard
    df_history = pd.read_csv(cfg["paths"]["data"]["processed"])
    return cfg, fe, artifact, df_history

cfg, fe, artifact, df_history = load_assets()
model = artifact["model"]
model_features = artifact["features"]

# --- SIDEBAR: SIMULADOR DE PREDIAÇÃO ---
with st.sidebar:
    st.header("🔍 Simular Novo Cliente")
    st.markdown("Ajuste os dados abaixo para ver a propensão ao churn em tempo real.")
    
    # Inputs com Tooltips (Help) para ajudar o usuário
    age = st.number_input("Idade", 18, 90, 30, help="Idade do cliente.")
    gender = st.selectbox("Gênero", ["Male", "Female"], help="Gênero do cliente.")
    region = st.selectbox("Região", ["North America", "Europe", "Asia", "South America", "Oceania"])
    
    sub_length = st.slider("Meses de Assinatura (Tenure)", 1, 100, 12, 
                           help="Tempo total de contrato. Clientes com mais de 12 meses tendem a ser mais leais.")
    
    monthly_spend = st.number_input("Gasto Mensal ($)", 5.0, 500.0, 50.0, 
                                   help="Valor da última fatura. Churn em clientes de alto gasto impacta mais o faturamento.")
    
    tickets = st.slider("Tickets de Suporte", 0, 20, 2, 
                        help="Quantidade de reclamações. Mais de 5 tickets por mês é um sinal de alerta crítico.")
    
    satisfaction = st.slider("Score de Satisfação", 1, 5, 3, 
                             help="Nota dada pelo cliente (1=Péssimo, 5=Excelente).")
    
    payment = st.selectbox("Método de Pagamento", ["Credit Card", "Bank Transfer", "PayPal"])
    
    st.divider()
    predict_btn = st.button("🚀 Calcular Risco de Churn", width='stretch')

# --- CONTEÚDO PRINCIPAL ---
st.title("🛡️ Radar de Churn & Insights de Retenção")

# Botão de Ajuda Geral
with st.expander("📖 Guia Rápido: O que este dashboard faz?"):
    st.markdown("""
    Este sistema utiliza **Inteligência Artificial (XGBoost)** para prever se um cliente irá cancelar o serviço de streaming.
    - **Métricas Globais:** Resumo da saúde atual da base histórica.
    - **Simulador Lateral:** Permite que o time de vendas/suporte teste perfis específicos.
    - **Gráfico de Dispersão:** Visualiza a relação entre gasto, tempo de casa e cancelamento.
    """)

# 1. MÉTRICAS DE NEGÓCIO
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Clientes Analisados", f"{len(df_history):,}")
with col2:
    churn_rate = df_history['Churned'].mean()
    st.metric("Taxa de Churn (Histórica)", f"{churn_rate:.1%}", delta="-2%" if churn_rate < 0.2 else "+1%", delta_color="inverse")
with col3:
    revenue_at_risk = df_history[df_history['Churned'] == 1]['Monthly_Spend'].sum()
    st.metric("Receita em Risco (Mensal)", f"$ {revenue_at_risk:,.2f}")

st.divider()

# 2. ÁREA DE RESULTADO DA PREDIAÇÃO
if predict_btn:
    # Criando o DataFrame com NOMES IDÊNTICOS aos que o FeatureEngineer espera
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Subscription_Length": sub_length,
        "Monthly_Spend": monthly_spend,
        "Support_Tickets_Raised": tickets,
        "Satisfaction_Score": satisfaction,
        "Payment_Method": payment
    }])
    
    # Executa Engenharia de Features
    df_enriched = fe.create_features(input_df)
    
    # Encoding Categórico (Get Dummies)
    cat_cols = cfg["model"]["features"]["categorical"]
    df_final = pd.get_dummies(df_enriched, columns=cat_cols)
    
    # Alinhamento de Colunas (Garante que o modelo não quebre por falta de colunas da Region ou Gender)
    df_final = df_final.reindex(columns=model_features, fill_value=0)
    
    # Predição
    prob = model.predict_proba(df_final)[0][1]
    
    # UI de Feedback
    st.subheader("🎯 Resultado da Simulação")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Probabilidade de Churn", f"{prob:.1%}")
    
    with res_col2:
        if prob > 0.7:
            st.error("🚨 **ALTO RISCO DE CANCELAMENTO**")
            st.info("💡 **Ação Recomendada:** O cliente está muito insatisfeito ou o valor não faz sentido. Oferecer upgrade para plano anual com 20% de desconto imediato.")
        elif prob > 0.4:
            st.warning("⚠️ **RISCO MODERADO**")
            st.info("💡 **Ação Recomendada:** Enviar conteúdo de 'Feature Discovery' para aumentar o engajamento com a plataforma.")
        else:
            st.success("✅ **CLIENTE SAUDÁVEL**")
            st.info("💡 **Ação Recomendada:** Perfil ideal para programa de indicação (Referral) ou teste de novas funcionalidades Beta.")
        
# --- EXPLICAÇÃO COM SHAP ---
    st.divider()
    st.subheader("🕵️ Por que o modelo deu esse resultado?")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_final)

    # 1. Dicionário de Tradução Abrangente
    labels_traduzidos = {
        # Colunas Originais
        'Age': 'Idade',
        'Gender': 'Gênero',
        'Subscription_Length': 'Tempo de Contrato (Meses)',
        'Monthly_Spend': 'Gasto Mensal',
        'Support_Tickets_Raised': 'Tickets de Suporte',
        'Satisfaction_Score': 'Score de Satisfação',
        'Last_Activity': 'Dias desde Última Atividade',
        'Region': 'Região',
        'Payment_Method': 'Método de Pagamento',
        
        # Colunas de Engenharia (Passo 1 e 2)
        'Estimated_LTV': 'LTV Estimado (Valor Total)',
        'Engagement_Score': 'Score de Engajamento',
        'LTV_Spend_Ratio': 'Eficiência de Gasto (Ratio)',
        'Engagement_per_Month': 'Engajamento/Mês',
        'Ticket_Engagement_Ratio': 'Tickets por Engajamento',
        
        # Colunas de Binarização (Flags)
        'Is_High_Spender': 'Cliente de Alto Gasto',
        'Is_Inactive': 'Cliente Inativo',
        'Is_Free_Trial': 'Conta em Período de Teste',
        
        # Colunas Categóricas (Após One-Hot Encoding)
        'Gender_Male': 'Gênero: Masculino',
        'Gender_Female': 'Gênero: Feminino',
        'Region_Germany': 'Região: Alemanha',
        'Region_France': 'Região: França',
        'Region_Spain': 'Região: Espanha',
        'Payment_Method_Credit Card': 'Pagamento: Cartão de Crédito',
        'Payment_Method_PayPal': 'Pagamento: PayPal',
        'Payment_Method_Bank Transfer': 'Pagamento: Transferência'
    }

    # 2. Configurações de Estilo
    COLOR = "#5A5A5A" 
    plt.rcParams.update({
        'text.color': COLOR, 'axes.labelcolor': COLOR,
        'xtick.color': COLOR, 'ytick.color': COLOR
    })
    
    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
    
    # 3. Gerar o gráfico
    # Usamos data=None para o SHAP parar de colocar "valor = nome_da_coluna" no eixo Y
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=None, 
            feature_names=[labels_traduzidos.get(col, col) for col in df_final.columns]
        ), 
        max_display=10, 
        show=False
    )

    vermelho_shap = "#ff0051"
    azul_shap = "#008bfb"

    for patch in ax_shap.patches:
        # Pega a cor atual da barra
        current_color = patch.get_facecolor()
        
        # Se for "avermelhado", vira azul. Se for "azulado", vira vermelho.
        # (Usamos uma lógica simples de checar o componente R vs B do RGB)
        if current_color[0] > current_color[2]: # Mais vermelho que azul
            patch.set_facecolor(azul_shap)
        else:
            patch.set_facecolor(vermelho_shap)

    # 4. LIMPEZA FINAL DOS EIXOS
    ax_shap.set_xlabel("") # Remove a legenda do eixo X ("SHAP value")
    ax_shap.set_xticks([]) # Remove os números/ticks do eixo X para um look minimalista
    ax_shap.spines['top'].set_visible(False)
    ax_shap.spines['right'].set_visible(False)
    ax_shap.spines['bottom'].set_visible(False) # Remove a linha de baixo
    
    # Forçar cor branca/cinza nos nomes das colunas (Eixo Y)
    ax_shap.tick_params(axis='y', colors=COLOR, labelsize=11)

    # 5. Renderizar
    st.pyplot(fig_shap, clear_figure=True, transparent=True)
    plt.close(fig_shap)

    st.info("""
    **Como ler este gráfico:**
    - Barras para a **direita (vermelhas/positivas)**: Indicam atributos que **aumentam** a chance de cancelamento.
    - Barras para a **esquerda (azuis/negativas)**: Indicam atributos que **favorecem** a permanência do cliente.
    """)

# 3. VISUALIZAÇÃO DE BI (PLOTLY)
st.subheader("📊 Comportamento da Base: Gasto vs. Retenção")
fig = px.scatter(
    df_history, 
    x="Monthly_Spend", 
    y="Subscription_Length", 
    color="Churned",
    size="Monthly_Spend",
    hover_data=['Age', 'Satisfaction_Score'],
    labels={"Churned": "Cancelou?", "Monthly_Spend": "Gasto Mensal ($)", "Subscription_Length": "Meses de Casa"},
    color_continuous_scale="RdYlGn_r"
)
st.plotly_chart(fig, width='stretch')