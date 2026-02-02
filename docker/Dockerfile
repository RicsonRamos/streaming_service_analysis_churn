# Use a lightweight stable Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential for XGBoost/Optuna if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to run the dashboard
CMD ["streamlit", "run", "main_dash.py", "--server.port=8501", "--server.address=0.0.0.0"]
