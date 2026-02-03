"""
UI Styling constants and helpers for Streamlit components.
"""

# Professional Color Palette
PALETTE = {
    "high_risk": "#BE0000",
    "medium_risk": "#FECB52",
    "low_risk": "#636EFA",
    "background": "#F0F2F6",
    "text": "#262730"
}

def get_risk_color(probability: float, threshold: float) -> str:
    """Returns the hex color code based on churn probability."""
    if probability >= threshold:
        return PALETTE["high_risk"]
    if probability >= 0.4:
        return PALETTE["medium_risk"]
    return PALETTE["low_risk"]

import matplotlib.pyplot as plt

CHART_THEME = {
    "bg_color": "#0E1117",
    "text_color": "#E0E0E0",
    "primary_color": "#BE0000",
    "font_size_title": 12,
    "font_size_labels": 10
}

def apply_chart_style(ax):
    """Standardizes plot aesthetics for the Churn Radar brand."""
    ax.set_facecolor(CHART_THEME["bg_color"])
    
    # Text and ticks
    ax.xaxis.label.set_color(CHART_THEME["text_color"])
    ax.yaxis.label.set_color(CHART_THEME["text_color"])
    ax.tick_params(axis='both', colors=CHART_THEME["text_color"], labelsize=CHART_THEME["font_size_labels"])
   
    
    # Hide spines for a clean look
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
        
    return ax