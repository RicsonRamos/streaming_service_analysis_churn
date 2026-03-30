# Utilizando uma imagem slim para reduzir a superfície de ataque e o tamanho
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Instalação de dependências de compilação (necessárias para o XGBoost em algumas distros)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia apenas o necessário para manter a imagem limpa
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY main.py /app/main.py

# Porta padrão do Streamlit
EXPOSE 8501

# O Entrypoint agora é flexível via Docker Compose
CMD ["python", "main.py", "--mode", "train"]