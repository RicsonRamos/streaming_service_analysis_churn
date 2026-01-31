FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos primeiro para cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPIA EXPLICITA DAS PASTAS (Para garantir que entrem na imagem)
COPY models/ ./models/
COPY data/ ./data/
COPY app.py .
# Se houver outros arquivos necessários, copie-os aqui

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]