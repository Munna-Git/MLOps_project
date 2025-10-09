# MLOps Project: End-to-End Pipeline for Robust Model Deployment

> **Elevator Pitch:**  
> This repository delivers a modular, production-ready MLOps pipeline built with Python, Flask, Docker, and automated CI/CD. It efficiently manages the full ML lifecycle from feature engineering and training to containerized deployment and robust validation, with integrated explainability and data quality checks for enterprise-scale reliability.

---

## Quick highlights for recruiters

- Delivers an automated pipeline for deploying and monitoring ML models at scale.
- Demonstrates mastery of cloud-native infrastructure and CI/CD best practices.
- Enables clear model explainability and robust validation for business trust.
- Integrates with industry-standard tools for reproducibility and maintainability.
- Ready for rapid adaptation to new use cases and datasets.

## Quick tech summary for engineers

- Modular architecture leveraging Python, Flask APIs, and Dockerized deployment.
- Automated CI/CD with GitHub Actions, unit tests, and data validation checks.
- Full-featured pipeline: data cleaning → feature engineering → training → inference → explainability.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why it matters](#why-it-matters)
3. [Tech stack](#tech-stack)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Feature & Training Pipeline](#feature--training-pipeline)
6. [How to run (Local Quickstart)](#how-to-run-local-quickstart)
7. [Docker usage](#docker-usage)
8. [CI/CD & Testing](#cicd--testing)
9. [Model Explainability & Validation](#model-explainability--validation)
10. [Files & Data Layout](#files--data-layout)
11. [MLOps Project — Challenges & Solutions](#mlops-project--challenges--solutions)
12. [Results & How to Interpret Them](#results--how-to-interpret-them)
13. [Roadmap / Future Enhancements](#roadmap--future-enhancements)
14. [Contributing, License, Contact](#contributing-license-contact)

---

## Project Overview

This repository provides a robust, production-ready MLOps pipeline designed for managing the entire model lifecycle — from raw data ingestion, feature engineering, model training and validation, to deployment, monitoring, and explainability. The solution integrates best-in-class tooling for reliability, scalability, and transparency, making it suitable for business-critical ML workflows.

## Why it matters

- **Operationalizes ML:** Automates the transition from research to production, reducing time-to-market.
- **Business trust:** Ensures reproducibility, transparency, and validation of model outputs.
- **Scalability:** Supports containerized deployment and CI/CD for seamless scaling in cloud or on-prem environments.
- **Compliance-ready:** Facilitates auditing and model explainability for regulatory standards.

---

## Tech stack

- **Python:** Core language for data processing, modeling, and orchestration.
- **Flask:** Serves model predictions via RESTful APIs for integration with downstream systems.
- **Docker:** Encapsulates the application for portable, reproducible deployment.
- **GitHub Actions:** Manages CI/CD pipelines for automated linting, testing, and deployment.
- **Great Expectations:** Implements data quality checks and validation throughout the pipeline.
- **SHAP:** Provides model explainability, enabling feature impact analysis for stakeholders.
- **Pydantic:** Enforces data schemas and validation in API endpoints.
- **scikit-learn, xgboost:** Advanced modeling and ML algorithms.

---

## Overall System Architecture

```
mlops_project/
│
├── app/
│   ├── train_entrypoint.py              # Entrypoint for training pipeline (UPDATED)
│   ├── inference_entrypoint.py          # Entrypoint for inference/prediction API (UPDATED)
│
├── src/
│   ├── data/
│   │   ├── clean_data.py                # Data cleaning and preprocessing logic (UPDATED)
│   │   ├── train_test_split.py          # Train-test splitting functions
│   │
│   ├── models/
│   │   ├── train_model.py               # Model training logic
│   │   ├── evaluate_models.py           # Model evaluation and metrics computation
│   │
│   ├── utils/
│   │   ├── shap_utils.py                # SHAP explainability utilities
│   │
│   ├── schemas/                          # NEW: Pydantic validation schemas
│   │   ├── __init__.py
│   │   ├── data_schemas.py              # Data validation schemas
│   │   ├── config_schemas.py            # Configuration management schemas
│   │   ├── model_schemas.py             # API & model metadata schemas
│   │
│   ├── validation/                       # NEW: Great Expectations data quality
│   │   ├── __init__.py
│   │   ├── ge_utils.py                  # Great Expectations utilities
│   │   ├── create_expectations.py       # Expectation suite definitions
│   │   ├── expectations/                # Generated expectation suites
│   │   ├── checkpoints/                 # Validation checkpoints
│   │   ├── uncommitted/                 # GE runtime files (gitignored)
│
├── data/
│   ├── raw/                             # Raw input data
│   ├── cleaned/                         # Cleaned datasets after preprocessing
│   ├── processed/                       # Processed datasets ready for modeling
│
├── models/
│   ├── xgboost_model.pkl                # Saved trained model(s)
│   ├── xgboost_model_metadata.json      # NEW: Model metadata and metrics
│
├── notebooks/
│   ├── 01_load_and_validate.ipynb           
│   ├── 02_clean_and_transform_raw_data.ipynb
│   ├── 03_train_test_split.ipynb          
│   ├── 03_train_random_forest.ipynb
│   ├── 03_train_xgb.ipynb 
│
├── test/
│   ├── feature_pipeline_test.py         # Unit tests for pipelines (UPDATED)
│
├── scripts/                              # NEW: Setup and utility scripts
│   ├── setup_great_expectations.py      # Initialize Great Expectations
│   ├── test_integration.py              # Integration testing script
│
├── .github/
│   ├── workflows/
│       ├── ci.yml 
│       ├── cd.yml 
│
├── .env                                 # Environment variables (paths, configs)
├── Dockerfile
├── .dockerignore
├── .env.docker
├── entrypoint-wrapper.sh
├── run.sh
├── requirements.txt                     # Python dependencies (UPDATED)
├── README.md                            # Project overview and documentation
├── README_PYDANTIC_GE.md               # NEW: Pydantic & GE integration guide
├── QUICKSTART.md                       # NEW: Quick start guide
├── IMPLEMENTATION_SUMMARY.md           # NEW: Implementation summary
├── INSTALLATION_GUIDE.md               # NEW: Step-by-step installation
├── .gitignore                          # Ignored files and directories (UPDATED)
└── venv/                               # Virtual environment (optional, not tracked)
```

---

## Feature & Training Pipeline

- **Data Ingestion:** Loads raw datasets from `/data` using modular loader scripts.
- **Feature Engineering:** Applies cleaning, transformation, and feature selection.
- **Training:** Trains models using scikit-learn or XGBoost, with parameter tuning and cross-validation.
- **Model Persistence:** Saves trained models to `/models` for deployment and future inference.
- **Testing:** Validates pipeline steps via unit tests in `/tests` and Great Expectations checks.

---

## How to Run

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Munna-Git/MLOps_project.git
    cd MLOps_project
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment**
    - Edit `.env` file with your paths and settings.

4. **Run the Training Pipeline**
    ```bash
    python app/train_entrypoint.py
    ```

5. **Start the Inference API**
    ```bash
    python app/inference_entrypoint.py
    ```

6. **Run Notebooks**
    - Open Jupyter and run any notebook in `notebooks/` for interactive EDA and modeling.

7. **Run Tests**
    ```bash
    pytest test/
    ```

8. **Docker Deployment**
    ```bash
    docker build -t churn-mlops .
    docker run -it --rm -p 5000:5000 ^
        -v D:/mlops_project/data:/app/data ^
        -v D:/mlops_project/models:/app/models ^
        mlops_project
    ```
*This mounts local data and model directories into the container for persistent access.*

---

## CI/CD & Testing

- **GitHub Actions:** Automates linting (isort, flake8), unit tests, and deployment flows.
- **Continuous Integration:** Every commit triggers style checks, unit tests, and data validation (Great Expectations).
- **Continuous Deployment:** On merge to main, Docker images are built and published to DockerHub using secrets for secure authentication.
- **Testing:** `/tests` includes unit tests; Great Expectations suite validates data integrity.

---

## Model Explainability & Validation

- **SHAP:** Generates feature importance and impact visualizations after training, supporting business transparency.
- **Great Expectations:** Ensures input data meets expectations before training/inference.
- **Validation:** Automated checks for feature consistency, schema compliance, and output sanity.

---

## Files & Data Layout

- `/data`: Raw and processed datasets (CSV, Parquet, etc.).
- `/models`: Saved model binaries and artifacts.
- `/src`: Source code for data processing, feature engineering, training, and serving.
- `/tests`: Unit tests and validation scripts.
- `/notebooks`: (Optional) Jupyter notebooks for exploration.
- `.env.docker`: Environment variable file for Docker deployment.
- `requirements.txt`: Python dependencies.

---

## MLOps Project — Challenges & Solutions

**1. Git & Terminal Setup**  
- *Problem:* New to using GitHub via terminal; accidentally pushed the entire venv folder despite .gitignore.  
- *Solution:* Repeated practice led to mastering terminal-based Git operations quickly.

**2. Model Pipeline Development**  
- *Problem:* Transitioning from a simple XGBoost notebook to a modular pipeline (abstract classes, data cleaning, train/test split, etc.) was challenging.  
- *Solution:* Broke the task into smaller components, implemented robust `src` structure, and ensured proper pipeline flow and testing.

**3. Entrypoint Integration**  
- *Problem:* Integrating training and inference logic caused multiple runtime issues.
  - Duplicate CustomerId columns
  - Object-type columns in numeric inputs
  - Feature mismatch between train/inference
  - Missing Flask route for predictions
  - src import errors in GitHub Actions
  - Missing CardType column in test data
- *Solution:* Debugged step-by-step and resolved all data/module-level errors.

**4. Continuous Integration (CI)**  
- *Problem:* Linting and testing failed for minor style issues and feature encoding mismatches.  
- *Solution:* Automated formatting (isort), ignored minor warnings (E203, W503), fixed test data encoding issues, and achieved a passing CI pipeline.

**5. Dockerization**  
- *Problem 1:* .sh entrypoint script failed due to line-ending mismatch (LF vs CRLF).  
  - *Solution:* Fixed with Git config:
    ```bash
    git config core.autocrlf false
    git rm --cached entrypoint-wrapper.sh
    git add entrypoint-wrapper.sh
    git commit -m "Fix LF endings for Docker"
    ```
- *Problem 2:* Missing data/model files inside Docker image.
  - *Solution:* Used `.env.docker` for container paths and Docker volumes:
    ```bash
    docker run -it --rm -p 5000:5000 \
      -v D:/mlops_project/data:/app/data \
      -v D:/mlops_project/models:/app/models \
      mlops_project
    ```
  - *Outcome:* Container built and ran successfully, with deployment to DockerHub.

**6. Continuous Deployment (CD) & GitHub Secrets**  
- *Problem:* Setting up CD for Docker push and managing environment variables.  
- *Solution:* Integrated GitHub Secrets for credentials and automated DockerHub publishing in `cd.yml`.

---

## Results & How to Interpret Them

- **Model Outputs:** Predictions can be retrieved via API or CLI, including feature attributions and confidence scores.
- **Key Metrics:**  
  - `f2_score`
- **Interpretation:**  
  - Use SHAP visualizations and Great Expectations validation results to understand model behavior and data quality.

---

## Roadmap / Future Enhancements

- Add model monitoring, drift detection, and automated retraining.
- Integrate advanced logging and alerting for production reliability.
- Scale out inference using cloud-native orchestration (Kubernetes, etc.).
- Expand feature pipeline for new data sources.
- Enhance CI with coverage reports and security scanning.

---
