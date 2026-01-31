import os
from pathlib import Path

# --- DIRETÓRIOS BASE ---
# Path(__file__).resolve() garante o caminho absoluto independente de onde o script é chamado
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# --- CAMINHOS DE DADOS ---
RAW_DATA_PATH = DATA_DIR / "raw" / "streaming.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "streaming_clean.csv"

# --- CONFIGURAÇÕES DE NEGÓCIO (KPIs) ---
# Centralizar aqui facilita mudar a regra de negócio para todo o projeto
CHURN_THRESHOLD = 0.5  # Score acima disso é considerado risco crítico
TARGET_COLUMN = "Churned"
LTV_MONTHS = 12        # Estimativa de projeção de receita para o acionista

# --- IDENTIDADE VISUAL (Storytelling) ---
# Define as cores da empresa para todos os gráficos do projeto
PALETTE_BUSINESS = ["#1A237E", "#E91E63", "#00C853", "#FFAB00"] # Azul Profundo, Rosa, Verde, Âmbar
FONT_LABEL_SIZE = 12
FONT_TITLE_SIZE = 16

# --- CONFIGURAÇÕES DE AMBIENTE ---
RANDOM_STATE = 42      # Garante reprodutibilidade em todas as análises