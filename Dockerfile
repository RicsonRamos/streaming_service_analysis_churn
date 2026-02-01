# Usar uma imagem leve do Python
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para o XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas o arquivo de requisitos primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# Instalar bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o projeto para dentro do container
COPY . .

# Criar pastas que o script pode precisar (caso não existam)
RUN mkdir -p models/artifacts data/processed outputs

# Definir a variável de ambiente para o Python encontrar o diretório 'src'
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Comando padrão ao rodar o container
ENTRYPOINT ["python", "run.py"]
CMD ["train"]