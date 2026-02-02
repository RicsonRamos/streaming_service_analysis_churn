"""
UI Styling constants and helpers for Streamlit components.
"""

# Professional Color Palette
PALETTE = {
    "high_risk": "#EF553B",
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
