FROM python:3.10-slim

WORKDIR /app

# Instala dependências do sistema para o XGBoost/SHAP
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expõe a porta do Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "app/main_dash.py", "--server.port=8501", "--server.address=0.0.0.0"]
