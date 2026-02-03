# Python slim leve
FROM python:3.10-slim

# Evita .pyc e buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia código do projeto
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY scripts/ /app/scripts/

# Expõe porta Streamlit
EXPOSE 8501

# Variáveis de ambiente default
ENV ENV=production
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/configs
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_PROMOTION_DIR=/app/mlruns_promoted
# Opcional: definir bucket para deploy S3/MinIO
ENV DEPLOY_TARGET=""

# Script de promoção automática
# trainer service usa isso: python /app/scripts/promote_model.py
COPY scripts/promote_model.py /app/scripts/promote_model.py

# Entrypoint padrão: roda app Streamlit
CMD ["streamlit", "run", "app/main_dash.py", "--server.port=8501", "--server.address=0.0.0.0"]
