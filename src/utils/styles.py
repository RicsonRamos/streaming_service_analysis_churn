"""
UI Styling constants and helpers for Streamlit components.
"""

# Professional Color Palette
PALETTE = {
    "high_risk": "#BE0000",
    "medium_risk": "#FECB52",
    "low_risk": "#636EFA",
    "background": "#F0F2F6",
    "text": "#262730",
}


def get_risk_color(probability: float, threshold: float = 0.7) -> str:
    """
    Returns the hex color code based on churn probability.

    Args:
        probability (float): The churn probability value.
        threshold (float, optional): The threshold value to determine high risk. Defaults to 0.7.

    Returns:
        str: The hex color code based on the churn probability.
    """
    # High risk if probability is above or equal to threshold
    if probability >= threshold:
        return PALETTE["high_risk"]
    # Medium risk if probability is above or equal to 0.4
    if probability >= 0.4:
        return PALETTE["medium_risk"]
    # Low risk otherwise
    return PALETTE["low_risk"]


import matplotlib.pyplot as plt

CHART_THEME = {
    "bg_color": "#0E1117",
    "text_color": "#E0E0E0",
    "primary_color": "#BE0000",
    "font_size_title": 12,
    "font_size_labels": 10,
}


def apply_chart_style(ax):
    """
    Standardizes plot aesthetics for the Churn Radar brand.

    The function standardizes the background color, text color, and tick label size.
    It also hides the spines for a clean look.

    Args:
        ax (matplotlib.axes.Axes): The axes object to apply the style to.

    Returns:
        matplotlib.axes.Axes: The styled axes object.
    """
    # Set background color
    ax.set_facecolor(CHART_THEME["bg_color"])

    # Set text and tick label color
    ax.xaxis.label.set_color(CHART_THEME["text_color"])
    ax.yaxis.label.set_color(CHART_THEME["text_color"])
    ax.tick_params(
        axis="both",
        colors=CHART_THEME["text_color"],
        labelsize=CHART_THEME["font_size_labels"],
    )

    # Hide spines for a clean look
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    return ax
