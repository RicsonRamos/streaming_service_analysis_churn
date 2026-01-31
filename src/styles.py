import matplotlib.pyplot as plt


def apply_business_style():
    """
    Storytelling with Data — Tema Claro Executivo

    Foco:
    - Menos eixos, mais mensagem
    - Valores no dado (não na escala)
    - Título guia a leitura
    """

    colors = {
        "primary": "#1f77b4",
        "alert": "#d62728",
        "secondary": "#7f7f7f",
        "highlight": "#ffbf00"
    }

    plt.rcParams.update({

        # --------------------------------------------------
        # FUNDO
        # --------------------------------------------------
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # --------------------------------------------------
        # DECLUTTER TOTAL
        # --------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,

        # Remove escalas
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.labelsize": 0,
        "ytick.labelsize": 0,

        # --------------------------------------------------
        # TEXTO
        # --------------------------------------------------
        "font.family": "sans-serif",
        "text.color": "#333333",

        "axes.titlecolor": "#6e6e6e",   # cinza claro
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,

        "axes.labelcolor": "none",

        # --------------------------------------------------
        # CORES
        # --------------------------------------------------
        "axes.prop_cycle": plt.cycler(
            color=[
                colors["primary"],
                colors["alert"],
                colors["secondary"],
                colors["highlight"]
            ]
        ),

        "patch.edgecolor": "white",
        "lines.linewidth": 2.5,

        # --------------------------------------------------
        # LEGENDA
        # --------------------------------------------------
        "legend.frameon": False,
        "legend.fontsize": 11,
    })

    print("🎯 Storytelling with Data Style Applied (Executive)")
