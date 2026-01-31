import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
from IPython.display import display, HTML

# =====================================================
# Configurações de Estilo e Output
# =====================================================
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------
# Função de estilo clean para storytelling (tema claro)
# -----------------------------------------------------
def apply_storytelling_style():
    """Estilo baseado no livro 'Storytelling with Data' (tema claro)."""
    colors = {
        "primary": "#448aff",   # Azul: Stay / neutro
        "alert": "#ff5252",     # Vermelho: Churn / alerta
        "secondary": "#cccccc", # Cinza claro: eixos / contexto
        "highlight": "#ffd740"  # Amarelo: insights
    }

    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.edgecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "text.color": "#636363",
        "axes.titlecolor": "#636363",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#888888",
        "axes.labelsize": 11,
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.prop_cycle": plt.cycler(
            color=[colors["primary"], colors["alert"], colors["secondary"], colors["highlight"]]
        ),
        "lines.linewidth": 2.5,
        "patch.edgecolor": "#ffffff",
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.labelcolor": "#636363"
    })

apply_storytelling_style()
CHURN_PALETTE = {0: "#448aff", 1: "#ff5252", "Stay": "#448aff", "Churn": "#ff5252"}

# -----------------------------------------------------
# Função de salvamento automático
# -----------------------------------------------------
def _save_figure(title: str):
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    

# =====================================================
# Painel Executivo / KPIs
# =====================================================
def display_executive_dashboard(df: pd.DataFrame, target_col='Churned', spend_col='Monthly_Spend'):
    churn_rate = df[target_col].mean() * 100
    avg_ticket = df[spend_col].mean()
    total_revenue_at_risk = df[df[target_col] == 1][spend_col].sum()
    total_clients = len(df)

    html = f"""
    <div style="font-family:sans-serif; background:#636363; padding:20px; border-radius:8px; border:1px solid #ddd; margin-bottom:25px;">
        <h2 style="color:#444; text-align:left; margin-top:0; border-bottom:1px solid #ccc; padding-bottom:5px;">Executive Summary</h2>
        <div style="display:flex; justify-content:space-around; font-size:14px;">
            <div style="text-align:center;">
                <p style="color:#888; margin:0;">Total Clients</p>
                <b style="color:#448aff; font-size:18px;">{total_clients:,}</b>
            </div>
            <div style="text-align:center;">
                <p style="color:#888; margin:0;">Churn Rate</p>
                <b style="color:#ff5252; font-size:18px;">{churn_rate:.1f}%</b>
            </div>
            <div style="text-align:center;">
                <p style="color:#888; margin:0;">Avg Ticket (Mo)</p>
                <b style="color:#444; font-size:18px;">${avg_ticket:.2f}</b>
            </div>
            <div style="text-align:center;">
                <p style="color:#888; margin:0;">Revenue at Risk</p>
                <b style="color:#ffd740; font-size:18px;">${total_revenue_at_risk:,.2f}</b>
            </div>
        </div>
    </div>
    """
    display(HTML(html))

# =====================================================
# Gráficos Storytelling
# =====================================================
def plot_financial_impact(df: pd.DataFrame, target_col='Churned', spend_col='Monthly_Spend'):
    revenue = df.groupby(target_col)[spend_col].sum()
    labels = ['Active Revenue', 'Lost Revenue']
    colors = [CHURN_PALETTE[0], CHURN_PALETTE[1]]

    fig, ax = plt.subplots(figsize=(7,5))
    bars = ax.bar(labels, revenue, color=colors, alpha=0.9)

    ax.set_title("Financial Loss Projection", loc="left", pad=12, color="#636363")
    ax.set_ylabel("")
    ax.set_xlabel("")
    #ax.set_xticks([])
    #ax.set_yticks([])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'${height:,.0f}', ha='center', va='bottom',  color="#636363")

    ax.legend(labels, loc='upper right', frameon=False)
    _save_figure("financial_loss_projection")
    plt.show()

def plot_categorical_churn_impact(df: pd.DataFrame, cat_features: list, target_col='Churned'):
    n = len(cat_features)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]

    for ax, col in zip(axes, cat_features):
        # 1. Calcula as proporções
        prop_df = (df.groupby(col)[target_col].value_counts(normalize=True).unstack() * 100)
        
        # 2. Gera o gráfico
        prop_df.plot(kind='bar', stacked=True, ax=ax, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], alpha=0.9)

        ax.set_title(f"{col} vs Churn", loc="left", color="#636363", pad=12)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # --- CORREÇÃO DOS RÓTULOS (Anotações) ---
        for container in ax.containers:
            labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container]
            ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=8, fontweight='bold')
        # ----------------------------------------

        ax.get_legend().remove()

    plt.tight_layout()
    plt.show()

    # Cria legenda única para toda a figura, centralizada acima dos gráficos
    fig.legend(["Stay", "Churn"], loc='upper center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta espaço para a legenda ficar acima
    _save_figure("categorical_churn_impact")
    plt.show()



def plot_numerical_distributions(df: pd.DataFrame, num_features: List[str], target_col='Churned'):
    n = len(num_features)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(6*n,5))
    if n==1: axes = [axes]

    for ax, col in zip(axes, num_features):
        sns.kdeplot(data=df, x=col, hue=target_col, fill=True,
                    palette=CHURN_PALETTE, ax=ax, common_norm=False, alpha=0.5)
        ax.set_title(f"{col} Distribution", loc="left", pad=12, color="#636363")
        ax.set_ylabel("")
        ax.set_xlabel("")
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure("numerical_distributions")
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame):
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar_kws={"shrink":0.8}, ax=ax)
    ax.set_title("Correlation Matrix", loc="left", pad=12, color="#636363")
    #ax.set_xticks([])
    #ax.set_yticks([])
    _save_figure("correlation_matrix")
    plt.show()

def plot_behavior_profile(df: pd.DataFrame, target_col='Churned'):
    df_avg = df.select_dtypes(include=[np.number]).groupby(target_col).mean().T
    df_norm = (df_avg - df_avg.min()) / (df_avg.max() - df_avg.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_norm, annot=df_avg, fmt=".1f", cmap='RdYlGn_r', cbar=False, ax=ax,
                annot_kws={"size":10, "weight":"bold", "color":"#636363"})
    ax.set_title("Customer Profile: Active vs Churn", loc="left", pad=12, color="#636363")
    ##ax.set_xticks([])
    ##ax.set_yticks([])
    _save_figure("customer_behavior_profile")
    plt.show()

# =====================================================
# Função orquestradora
# =====================================================
def initial_insights_report(df: pd.DataFrame):
    display_executive_dashboard(df)
    plot_behavior_profile(df)
    plot_financial_impact(df)
    plot_correlation_matrix(df)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if df[c].nunique() < 15]
    plot_categorical_churn_impact(df, cat_cols)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    important_nums = [c for c in num_cols if c not in ['Churned', 'Customer_ID', 'id', 'score_churn']]
    plot_numerical_distributions(df, important_nums[:3])

    #display(HTML(f"<p style='color:#888;'>✅ Relatório completo. Gráficos salvos em: <b>{OUTPUT_DIR}/</b></p>"))
