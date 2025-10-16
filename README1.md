# 🚀 **MLOps Project:** End-to-End Customer Churn Prediction Pipeline

> **Elevator Pitch:**  
> This repository delivers a **modular**, **production-ready MLOps pipeline** built with **Python**, **Flask**, **Docker**, **MLflow** (for experiment tracking), and **automated CI/CD**. The solution is **deployed using Render** and efficiently manages the full ML lifecycle—from data to deployment—providing traceable, scalable, and reproducible results.

---

## 🎯 **Quick Highlights for Recruiters**

- **Production-Ready MLOps System:**  
  Delivered a **fully automated pipeline** for **data validation**, **model training**, **deployment**, and **monitoring**—showcasing **end-to-end ML engineering and DevOps integration**.

- **Cloud-Native & Scalable Infrastructure:**  
  Built using **containerized microservices (Docker)**, **MLflow for experiment tracking**, and **CI/CD workflows (GitHub Actions)**, ensuring **reproducibility**, **maintainability**, and **rapid adaptation**.

- **Model Explainability & Trust:**  
  Integrated **SHAP** for **interpretable predictions**, empowering stakeholders to **understand and trust model outcomes**.

- **Robust Architecture:**  
  Designed **modular**, **well-logged**, **API-driven components** with **clear separation of concerns** for scalability and easy integration.

- **CI/CD & Monitoring Excellence:**  
  Established **continuous integration**, **testing**, and **deployment pipelines** with **real-time monitoring**—ensuring **reliability** and **consistent model performance** at scale.

- **Experiment Tracking:**  
  **MLflow-based tracking** and comparison of all model runs, hyperparameters, and results.

- **Modern Cloud Deployment:**  
  **Production API deployed using Render**, supporting **scalability** and **easy updates**.

---

## 🛠️ **Quick Tech Summary for Engineers**

- **Modular architecture** leveraging **Python**, **Flask APIs**, and **Dockerized deployment**
- **Automated CI/CD** with **GitHub Actions**, **unit tests**, and **data validation checks**
- **Full-featured pipeline:**  
  **data cleaning** → **feature engineering** → **training** → **inference** → **explainability**
- **Experiment management** with **MLflow** for reproducible experiments and model registry
- **API and model deployment automated via Render**

---

## 📑 **Table of Contents**

- [Project Overview](#project-overview)
- [Why This Project Matters](#why-this-project-matters)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Feature & Training Pipeline](#feature--training-pipeline)
- [How to Run](#how-to-run)
- [CI/CD & Testing](#cicd--testing)
- [Model Explainability & Validation](#model-explainability--validation)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Deployment on Render](#deployment-on-render)
- [MLOps Project — Challenges & Solutions](#mlops-project-challenges--solutions)
- [Contact](#contact)

---

## 📝 **Project Overview**

This project implements a **customer churn prediction system** designed with **production-grade MLOps practices**. The system predicts whether a customer will leave a service (churn) based on behavioral and demographic data.

- **Automated data validation and quality checks**
- **Modular, testable pipeline architecture**
- **REST API for real-time predictions**
- **Docker containerization for deployment**
- **MLflow experiment tracking and model registry**
- **Production API deployed on Render**
- **CI/CD automation with GitHub Actions**
- **Model explainability via SHAP**
- **Comprehensive logging and error handling**

---

## 💡 **Why This Project Matters**

**Business Value:**  
Customer churn directly impacts **revenue and growth**. This system enables businesses to:

- **Predict churn proactively:** Identify at-risk customers before they leave
- **Optimize retention strategies:** Target interventions based on SHAP-driven insights into churn drivers
- **Reduce operational risk:** Automated validation catches data issues before they affect predictions
- **Scale efficiently:** Containerized deployment enables easy scaling across environments

---

## 🧰 **Tech Stack**

### **Core ML & Data Processing**

- **Python 3.8+:** Primary programming language
- **Pandas & NumPy:** Data manipulation and numerical operations
- **scikit-learn:** Train/test splitting, preprocessing utilities, evaluation metrics
- **XGBoost:** Gradient boosting classifier for churn prediction
- **imbalanced-learn:** Handling class imbalance in churn data
- **pytest:** Unit and integration testing framework

### **API & Deployment**

- **Flask:** REST API server for inference endpoints with health checks
- **Docker:** Containerization for reproducible deployments
- **Waitress:** Production WSGI server for Flask (optional)
- **Render:** Cloud deployment platform for API hosting

### **Model Explainability**

- **SHAP:** Feature importance and per-prediction explanations for model transparency

### **CI/CD & Infrastructure**

- **GitHub Actions:** Automated testing, linting, and Docker image builds
- **isort, flake8:** Code formatting and linting
- **python-dotenv:** Environment variable management

### **Monitoring & Experiment Tracking**

- **MLflow:** Experiment tracking, model registry, and reproducible runs

---

## 🏗️ **Project Architecture**

```
mlops_project/
│
├── app/
│   ├── train_entrypoint.py
│   ├── inference_entrypoint.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│
├── models/
│   ├── xgboost_model.pkl
│
├── notebooks/
│
├── test/
│
├── mlruns/                  # MLflow tracking directory (local runs)
│
├── .github/
│   ├── workflows/
├── .env
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

## 🔁 **Feature & Training Pipeline**

**End-to-End Pipeline Flow**  
The training pipeline is orchestrated through `app/train_entrypoint.py` and executes the following stages:

### **1. Data Loading**

- Raw CSV loaded from `data/raw/Customer-Churn-Records.csv`

### **2. Data Cleaning & Preprocessing**

- Drops unnecessary columns (*RowNumber, Surname, Complain, CustomerId for training*)
- Renames columns for consistency (*e.g., Satisfaction Score → SatisfactionScore*)
- One-hot encodes Geography and Gender
- Ordinal encodes CardType (*SILVER=0, GOLD=1, PLATINUM=2, DIAMOND=3*)

### **3. Train-Test Split**

- Stratified split to preserve churn ratio (*default: 90/10*)
- Saves both CSV and pickle formats

### **4. Model Training**

- **XGBoost classifier** with configured hyperparameters
- Training logged with comprehensive metrics

### **5. Model Evaluation**

- Computes **F1, F2, Precision, Recall, Confusion Matrix**
- Generates **ROC and Precision-Recall curves**

### **6. Model Persistence**

- Saves model as `.pkl` file

---

## ▶️ **How to Run**

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

4. **Start MLflow Tracking Server (optional for team/multi-node)**
    ```bash
    # Local tracking (default)
    mlflow ui --backend-store-uri ./mlruns
    # or for remote server
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0
    ```

5. **Run the Training Pipeline**
    ```bash
    python app/train_entrypoint.py
    ```
    - All runs and metrics will be tracked in **MLflow**.

6. **Start the Inference API**
    ```bash
    python app/inference_entrypoint.py
    ```

7. **Run Notebooks**
    - Open Jupyter and run any notebook in `notebooks/`.

8. **Run Tests**
    ```bash
    pytest test/
    ```

9. **Docker Deployment (Local)**
    ```bash
    docker build -t churn-mlops .
    docker run -it --rm -p 5000:5000 \
        -v D:/mlops_project/data:/app/data \
        -v D:/mlops_project/models:/app/models \
        churn-mlops
    ```

10. **API Deployment on Render**
    - Connect this repo to your **Render** account.
    - Create a new **Web Service**, set build and start commands:
        - **Build Command:** `pip install -r requirements.txt`
        - **Start Command:** `python app/inference_entrypoint.py`
    - Add environment variables/secrets in Render dashboard.
    - Redeploy on new commits automatically.

---

## 🔄 **CI/CD & Testing**

### **GitHub Actions Workflows**

The project uses two automated workflows:

#### **Continuous Integration (`ci.yml`)**

- **Linting** with flake8
- **Code formatting check** with isort
- **Unit tests** with pytest
- **Feature pipeline integration tests**

#### **Continuous Deployment (`cd.yml`)**

- **Triggered on pushes to main branch**
- **Build Docker image**
- **Tag with commit SHA and 'latest'**
- **Push to DockerHub using GitHub Secrets**

#### **GitHub Secrets Required:**

- `DOCKERHUB_USERNAME`: DockerHub account username
- `DOCKERHUB_TOKEN`: DockerHub access token

---

## 📊 **Model Explainability & Validation**

### **SHAP (SHapley Additive exPlanations)**

**SHAP** provides transparent, interpretable predictions by computing feature contributions:

- **Initialized** in `app/inference_entrypoint.py`
- **Calculates per-prediction feature importance**
- **Returned in API response** for `/score/<customer_id>` endpoint

**Example SHAP Output:**
```
{
  "shap_values": {
    "Age": 0.12,
    "Balance": 0.34,
    "NumOfProducts": -0.05,
    "IsActiveMember": -0.23
  }
}
```

**Use Cases:**

- **Understanding** why individual customers are predicted to churn
- **Identifying** global feature importance patterns
- **Building trust** with stakeholders through transparent predictions

---

## 📈 **Experiment Tracking with MLflow**

All model training runs, hyperparameters, metrics, and artifacts are **logged automatically to MLflow**.

- **How:**  
  MLflow is integrated into the training pipeline. All experiments are tracked under the local `mlruns/` directory or a remote server if configured.
- **Benefits:**  
  - **Compare model versions and parameters easily**
  - **Visualize metrics and download artifacts**
  - **Register and promote best models for deployment**

**Quickstart:**
```bash
mlflow ui
# then navigate to http://localhost:5000
```

---

## 🌐 **Deployment on Render**

The production inference API is **deployed and served using [Render](https://render.com/):**

- **Steps:**
  1. Connect your GitHub repository in Render dashboard.
  2. Choose **Web Service** and select your repo/branch.
  3. Set environment variables and secrets.
  4. Set the build and start commands as above.
  5. On every new commit to `main`, Render redeploys the API automatically.

- **Benefits:**
  - **Managed, scalable API deployment** with HTTPS and zero-downtime deploys
  - **No manual server/VM management required**

---

## MLOps Project — Challenges & Solutions

## 1. Git & Terminal Setup

**Problem:**  
New to using GitHub via terminal; accidentally pushed the entire `venv` folder despite `.gitignore`.

**Solution:**
- Practiced terminal-based Git operations repeatedly  
- Learned proper `.gitignore` patterns for Python projects  
- Mastered Git workflow (`add`, `commit`, `push`, `pull`) within a day  

**Key Learning:**  
Always verify `.gitignore` before the initial commit.

---

## 2. Model Pipeline Development

**Problem:**  
Transitioning from a simple XGBoost notebook to a modular pipeline was challenging — implementing abstract classes, data cleaning, train/test split logic, and proper separation of concerns.

**Solution:**
- Broke the problem into smaller, manageable components  
- Implemented robust `src/` structure with clear module boundaries:


- Used abstract base classes (`DataPipeline`) to enforce consistent preprocessing interfaces  
- Added comprehensive testing to validate each pipeline component  

**Result:**  
Clean, maintainable codebase with proper separation of training and inference logic.

---

## 3. Entrypoint Integration

**Problem:**  
Integrating training and inference logic caused multiple runtime issues that were difficult to debug.

**Solution:**  
Debugged step-by-step and resolved the following issues:

- **Duplicate `CustomerId` columns:** Fixed by ensuring `CustomerId` is dropped before training but retained during inference preprocessing  
- **Object-type columns in numeric inputs:** Added explicit type checking and validation with **Pydantic** schemas to catch type mismatches early  
- **Feature mismatch between train/inference:** Synchronized preprocessing logic between `TrainingPipeline` and `InferencePipeline` classes  
- **Missing Flask route for predictions:** Added proper routing in `inference_entrypoint.py` with both simple (`/<customer_id>`) and detailed (`/score/<customer_id>`) endpoints  
- **`src` import errors in GitHub Actions:** Fixed by adding `sys.path.append()` in entry points and configuring proper Python path in CI workflow  
- **Missing `CardType` column in test data:** Handled optional columns gracefully in schemas with `Optional` types and default values  

**Key Takeaway:**  
Systematic debugging and comprehensive logging were essential for identifying root causes.

---

## 4. Continuous Integration (CI)

**Problem:**  
Linting and testing repeatedly failed for small style issues (spacing, tabs, comments) and feature mismatches (Gender encoding inconsistencies).

**Solution:**
- Automated code formatting with **isort** for consistent import ordering  
- Configured **flake8** to ignore minor style warnings (`E203`, `W503`, `E501` for line length)  
- Fixed test data encoding issues by ensuring consistent one-hot encoding for Gender (Female/Male) across training and test datasets  
- Added **pre-commit hooks** to catch formatting issues locally before pushing  

**Result:**  
Passing CI pipeline with **zero lint errors** and **100% test pass rate**.

---

## 5. Dockerization

### Problem 1: `.sh` entrypoint script failed due to line-ending mismatch (LF ↔ CRLF)

**Solution:**  
Fixed via Git configuration:
```bash
# Configure Git to preserve line endings
git config core.autocrlf false

# Remove cached file with wrong endings
git rm --cached entrypoint-wrapper.sh

# Re-add with correct LF endings
git add entrypoint-wrapper.sh
git commit -m "Fix LF endings for Docker"
```

**Key Learning:**
Windows/Linux line-ending differences can break shell scripts in containers.

---

### Problem 2: Missing data/model files inside Docker image

**Solution:**

* Created `.env.docker` with container-specific paths (`/app/data`, `/app/models`)
* Used **Docker volumes** to mount local directories into container:

  ```bash
  docker run -it --rm -p 5000:5000 \
      -v D:/mlops_project/data:/app/data \
      -v D:/mlops_project/models:/app/models \
      mlops_project
  ```

**Outcome:**
Successfully built and ran the container, gaining hands-on experience with:

* Port mapping (`-p 5000:5000`)
* Volume mounting (`-v`)
* Environment file injection (`--env-file`)
* DockerHub deployment workflow

---

## 6. Continuous Deployment (CD) & GitHub Secrets

**Problem:**
Setting up CD for automated Docker builds and secure credential management.

**Solution:**

* Used **GitHub Secrets** to store sensitive credentials:

  * `DOCKERHUB_USERNAME`
  * `DOCKERHUB_TOKEN`
* Integrated DockerHub publishing in `cd.yml` workflow:

  * Build image on every push to `main`
  * Tag with commit SHA and `latest`
  * Push to DockerHub automatically

**Result:**
🚀 Fully automated deployment pipeline from code commit to DockerHub.

---




## Future Enhancements

---

### 🧩 Short-Term (Next 1–3 Months)

#### **Monitoring & Alerting**

* Integrate **Prometheus** for model performance tracking
* Set up alerts for prediction latency and error rates
* Track **data drift metrics** over time

#### **Model Retraining Pipeline**

* Automate retraining on new data batches
* Implement **A/B testing framework** for model comparison
* Enable **rolling deployment** strategy for smooth updates

#### **Enhanced Logging**

* Use **ELK Stack (Elasticsearch, Logstash, Kibana)** for structured logging
* Implement request/response logs for audit trails
* Add performance profiling to detect bottlenecks

#### **Data Drift Detection**

* Integrate **Evidently AI** or **Alibi Detect**
* Trigger alerts and automated retraining when drift is detected

---

### ⚙️ Medium-Term (3–6 Months)

#### **Kubernetes Deployment**

* Migrate from Docker to **Kubernetes**
* Enable **Horizontal Pod Autoscaling** for inference workloads
* Manage configuration using **Helm charts**

#### **Feature Store**

* Implement **Feast** or a custom feature store
* Centralize feature management and versioning
* Minimize feature duplication across models

#### **Advanced CI/CD**

* Multi-stage deployments (**dev → staging → prod**)
* Canary deployments for gradual rollout
* Automated rollback on performance degradation
* Integration tests in staging environment

#### **API Enhancements**

* Add **batch prediction endpoint** for bulk scoring
* Introduce **asynchronous prediction** queue using **Celery**
* Implement rate limiting and **JWT authentication**
* Generate **OpenAPI/Swagger** documentation

---

### 🚀 Long-Term (6–12 Months)

#### **Multi-Model System**

* Develop **model ensembles** for improved accuracy
* Segment-based model selection
* Integrate **AutoML** for hyperparameter optimization
* Use **shadow mode deployment** for safe A/B testing

#### **Database Integration**

* Store predictions and audit logs in **PostgreSQL**
* Use a **time-series database** for performance metrics
* Implement **Redis** for feature caching
* Enable historical prediction analysis

#### **Advanced Explainability**

* Build **interactive SHAP dashboards**
* Provide **counterfactual (“what-if”) analysis**
* Track **global and segment-specific feature importance** over time

#### **Scaling & Performance**

* Serve models with **TorchServe** or **TensorFlow Serving**
* Enable **GPU acceleration** for batch predictions
* Implement **distributed training** using **Ray** or **Spark**
* Explore **edge deployment** for low-latency use cases

#### **Data Quality Dashboard**

* Enable real-time **Great Expectations** monitoring
* Generate **data quality scorecards and reports**
* Integrate with BI tools (**Tableau**, **Power BI**)

#### **Compliance & Governance**

* Track **model versioning and lineage**
* Add **GDPR compliance** features (data deletion, right to explanation)
* Monitor **bias and fairness metrics**
* Maintain **audit logs** for regulatory compliance

---

## 🏁 Summary of Key Learnings

* Strong hands-on experience in MLOps pipeline design and debugging
* Deep understanding of CI/CD, containerization, and model deployment
* Improved problem-solving and system design thinking
* Ready for real-world production ML system implementation

---

## 📬 **Contact**

- **Author:** Munna-Git  
- **GitHub:** [Munna-Git](https://github.com/Munna-Git)  
- **Email:** *your-email@example.com* (update with your own if you wish)