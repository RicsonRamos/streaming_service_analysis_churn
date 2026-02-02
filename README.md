
#  Churn Radar: Predictive Streaming Analytics
**An end-to-end Machine Learning ecosystem for customer retention.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost 2.0](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Business Value
In the streaming industry, acquiring a new customer is **5x more expensive** than retaining an existing one. **Churn Radar** identifies high-risk users with **AI-driven precision**, allowing marketing teams to act before the cancellation happens.

### Key Features:
* **Predictive Engine:** XGBoost model optimized via Optuna Bayesian search.
* **XAI (Explainable AI):** Integration with SHAP to explain "the why" behind every prediction.
* **Strategy Simulator:** Real-time "What-If" analysis for retention offers.
* **Production Ready:** Fully containerized with Docker and validated with Pytest.

---

## System Architecture
The project follows a modular "Production-First" structure, ensuring scalability and maintainability.



* `src/config/`: Centralized YAML-based configuration (No hardcoded paths).
* `src/data/`: Data I/O abstraction layer for clean ingestion.
* `src/features/`: Robust validation (Pydantic) and Feature Engineering logic.
* `src/models/`: Model wrappers, Baseline comparison, and Hyperparameter Tuning (Optuna).
* `src/app/`: Streamlit-based Dashboard and Business Logic Services.
* `tests/`: Unit testing suite for data integrity and model consistency.

---


## Quick Start

### 1. Prerequisites
* [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
* Python 3.10+ (for local development)

### 2. Setup & Execution via Docker
Clone the repository and run the entire ecosystem (App + MLflow Tracking):

```bash
# Clone the repo
git clone [https://github.com/youruser/churn-radar.git](https://github.com/youruser/churn-radar.git)
cd churn-radar

# Build and start containers
docker-compose up --build
'''


Note: The dashboard will be available at http://localhost:8501.
___ 


The dashboard will be available at http://localhost:8501.
3. Training the Model
To retrain the model with the latest data and perform Bayesian optimization:
docker exec -it churn_radar_prod python scripts/train_model.py

Quality Assurance
We don't trust "it works on my machine". We trust automated validation.
# Run the test suite
pytest tests/ -v

Our validation layer ensures:
 * Mathematical Correctness: No divisions by zero in feature ratios.
 * Schema Integrity: Pydantic prevents corrupted data from reaching the model.
 * Boundary Safety: Rejection of impossible inputs (e.g., age > 100).
Tech Stack
 * Core: Python 3.10, Pandas, Scikit-Learn.
 * ML: XGBoost (Classifier), Optuna (Tuning), SHAP (Interpretability).
 * Tracking: MLflow.
 * Dashboard: Streamlit, Plotly.
 * Infrastructure: Docker, Docker-Compose.
 * Quality: Pytest, Pydantic, Black, Isort.
Roadmap
 * [ ] Implement CI/CD Pipeline via GitHub Actions.
 * [ ] Add support for Parquet/Avro files for improved I/O.
 * [ ] Integrate Slack/Email notifications for "High-Risk" alerts.
Developed with focus on logic, stability, and ROI. 
