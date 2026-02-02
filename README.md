# Churn Radar: Predictive Streaming Analytics

**An end-to-end Machine Learning ecosystem for customer retention.**

  

---

## Business Value

In the streaming industry, acquiring a new customer is **5x more expensive** than retaining an existing one. **Churn Radar** identifies high-risk users with **AI-driven precision**, enabling proactive retention strategies.

### Key Features

* Predictive engine using XGBoost optimized with Optuna
* Explainable AI via SHAP
* Retention strategy simulator (What-if analysis)
* Production-ready Dockerized environment
* Automated validation and testing

---

## System Architecture

The project follows a modular, production-oriented structure.

```
churn-radar/
│
├── src/
│   ├── config/     # YAML-based centralized configuration
│   ├── data/       # Data ingestion and persistence layer
│   ├── features/   # Validation and feature engineering
│   ├── models/     # Training, tuning, and inference
│   └── app/        # Streamlit dashboard and services
│
├── scripts/         # Training and maintenance scripts
├── tests/           # Automated tests
├── docker-compose.yml
└── requirements.txt
```

---

## Configuration

All runtime configuration is centralized in:

```
src/config/config.yaml
```

This file controls:

* Data paths
* Model parameters
* MLflow tracking URI
* Feature flags
* Environment settings

No absolute paths or secrets should be hardcoded in the source code.

Example:

```yaml
data:
  raw_path: data/raw/
  processed_path: data/processed/

model:
  name: xgboost
  max_depth: 6
  learning_rate: 0.05

mlflow:
  tracking_uri: http://mlflow:5000
```

---

## Quick Start

### 1. Prerequisites

* Docker
* Docker Compose
* Python 3.10+ (local development only)

Verify:

```bash
docker --version
docker-compose --version
python --version
```

---

### 2. Clone Repository

```bash
git clone https://github.com/RicsonRamos/streaming_service_analysis_churn
cd churn-radar
```

---

### 3. Environment Setup

Create environment file:

```bash
cp .env.example .env
```

Edit `.env` if necessary:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
APP_ENV=production
```

---

### 4. Run with Docker (Recommended)

Build and start all services:

```bash
docker-compose up --build
```

Services started:

* Streamlit App: [http://localhost:8501](http://localhost:8501)
* MLflow Tracking: [http://localhost:5000](http://localhost:5000)

To run in background:

```bash
docker-compose up -d --build
```

---

## Model Training

To retrain the model inside the container:

```bash
docker exec -it churn_radar_prod python scripts/train_model.py
```

This process will:

* Load latest processed data
* Run Optuna optimization
* Train final model
* Log artifacts to MLflow
* Persist model in `models/` directory

---

## Quality Assurance

Run tests locally or inside container:

```bash
pytest tests/ -v
```

Validation guarantees:

* No invalid mathematical operations
* Strict schema enforcement (Pydantic)
* Input boundary checks
* Feature consistency

---

## 💻 Local Development (Without Docker)

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\\Scripts\\activate    # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run src/app/main.py
```

### 4. Train Model Locally

```bash
python scripts/train_model.py
```

Note: MLflow must be running separately in this mode.

---

## Tech Stack

Core

* Python 3.10
* Pandas
* NumPy
* Scikit-learn

Machine Learning

* XGBoost
* Optuna
* SHAP

Tracking & Visualization

* MLflow
* Streamlit
* Plotly

Infrastructure & Quality

* Docker / Docker Compose
* Pytest
* Pydantic
* Black
* Isort

---

## Security & Secrets

* Do not commit `.env` files
* Use environment variables for credentials
* Use Docker secrets or Vault in production

---

## Roadmap

*

---

## License

MIT License. See `LICENSE` for details.

---

Developed with focus on engineering rigor, stability, and ROI.
