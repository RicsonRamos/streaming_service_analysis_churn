FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Instalação de dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY main.py /app/main.py

# NÃO copia data/ - usaremos volume mount em desenvolvimento
# NÃO copia mlflow.db - usamos servidor remoto

EXPOSE 8501

# Comando padrão (sobrescrito pelo docker-compose)
CMD ["streamlit", "run", "app/main_dash.py", "--server.address=0.0.0.0"]