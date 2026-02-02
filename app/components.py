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
    Renders a minimalist SHAP bar chart strictly limited to the top 3 features.
    Forces the limit by filtering data before plotting to guarantee results.
    """
    st.write("### 🧬 Top 3 Churn Drivers")
    
    friendly_names = {
        'Age': 'Customer Age',
        'Subscription_Length': 'Tenure (Months)',
        'Support_Tickets_Raised': 'Support Interactions',
        'Monthly_Spend': 'Monthly Billing',
        'Estimated_LTV': 'Lifetime Value',
        'Engagement_Score': 'App Engagement',
        'LTV_Spend_Ratio': 'Efficiency Ratio',
        'is_senior': 'Senior Citizen (60+)',
        'Is_Free_Trial': 'On Free Trial',
        'Is_High_Spender': 'High Spender Flag',
        'Satisfaction_Score': 'Satisfaction Rating'
    }

    try:
        # 1. Calcular SHAP valores
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_sample)

        # 2. Identificar as 3 colunas com maior impacto médio (Cálculo manual do Top 3)
        # lidando com o fato de shap_values poder vir como lista (multi-class) ou array
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(df_sample.columns, vals)), columns=['col','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        
        # Pegamos apenas o nome das 3 colunas mais importantes
        top_3_cols = feature_importance['col'].head(3).tolist()
        
        # 3. Filtrar os dados e os SHAP values para conter apenas essas 3
        # Localizamos os índices das colunas top 3 no dataframe original
        indices = [df_sample.columns.get_loc(c) for c in top_3_cols]
        filtered_shap_values = shap_values[:, indices]
        filtered_df = df_sample[top_3_cols].rename(columns=friendly_names)

        # 4. Plotagem com os dados já podados
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 3)) # Reduzi a altura para 3 para ficar bem focado
        
        shap.summary_plot(
            filtered_shap_values, 
            filtered_df, 
            plot_type="bar", 
            show=False, 
            color="#2E5077"
        )

        # Limpeza Storytelling
        ax.set_xlabel("")
        ax.set_xticks([]) 
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        
        ax.tick_params(axis='y', labelsize=12)
        plt.title("Primary drivers of customer loss", loc='left', fontsize=12, pad=15)

        st.pyplot(fig, transparent=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"Force limit failed: {e}")

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

def render_explainability(shap_values, X_processed, expected_value=None):
    """
    Renders SHAP visualizations with Storytelling with Data principles.
    Focuses on business-friendly labels, transparency, and high signal-to-noise ratio.
    """
    st.write("### What is driving our customers away?")
    
    if X_processed is None or len(X_processed) == 0:
        st.warning("No data available to explain.")
        return

    # 1. Map technical names to Business English
    friendly_names = {
        'Age': 'Customer Age',
        'Subscription_Length': 'Tenure (Months)',
        'Support_Tickets_Raised': 'Support Interactions',
        'Monthly_Spend': 'Monthly Billing',
        'Estimated_LTV': 'Lifetime Value',
        'Engagement_Score': 'App Engagement',
        'LTV_Spend_Ratio': 'Efficiency Ratio',
        'is_senior': 'Senior Citizen (60+)',
        'Is_Free_Trial': 'On Free Trial',
        'Is_High_Spender': 'High Spender Flag',
        'Satisfaction_Score': 'Satisfaction Rating'
    }
    
    # Create a copy with renamed columns for the plot
    X_plot = X_processed.rename(columns=friendly_names)

    try:
        # Clear any existing plots to avoid ghosting
        plt.clf()
        fig, ax = plt.subplots(figsize=(5, 3))
        
        if len(X_processed) > 1:
            # GLOBAL EXPLANATION
            shap.summary_plot(
                shap_values, 
                X_plot, 
                plot_type="bar", 
                show=False,
                color="#2E5077" # Slate Blue (Low cognitive load)
            )
        else:
            # LOCAL EXPLANATION (Simulator)
            exp = shap.Explanation(
                values=shap_values[0],
                base_values=expected_value if expected_value is not None else 0,
                data=X_plot.iloc[0].values,
                feature_names=X_plot.columns.tolist()
            )
            shap.plots.bar(exp, show=False)

        # 2. STORYTELLING CLEANUP (Decluttering)
        ax.set_xlabel("") # Remove SHAP value label
        ax.set_xticks([]) # Remove X axis numbers
        
        # Remove borders (Spines)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        
        ax.spines['left'].set_color('#CCCCCC') # Faint Y axis line
        
        # Style Y-axis labels
        ax.tick_params(axis='y', colors='#444444', labelsize=11)
        
        # 3. Action-oriented Title
        plt.title(
            "Top Predictors of Churn Risk\n(Longer bars indicate stronger impact)", 
            loc='left', 
            fontsize=12, 
            color="#6B6B6B",
            pad=20
        )

        # Force transparent background for Streamlit Dark/Light mode compatibility
        st.pyplot(fig, transparent=True)
        plt.close(fig)
            
    except Exception as e:
        st.error(f"Visualization Error: {e}")

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