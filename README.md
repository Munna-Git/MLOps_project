# **MLOps Project:** End-to-End Customer Churn Prediction Pipeline

> **Elevator Pitch:**  
> This repository delivers a **modular**, **production-ready MLOps pipeline** built with **Python**, **Flask**, **Docker**, **MLflow** (for experiment tracking), and **automated CI/CD**. The solution is **deployed using Render** and efficiently manages the full ML lifecycle-from data to deployment-providing traceable, scalable, and reproducible results.

---

## 🎯 **Quick Highlights for Recruiters**

- **Production-Ready MLOps System:**  
  Delivered a **fully automated pipeline** for **data validation**, **model training**, **deployment**, and **monitoring**-showcasing **end-to-end ML engineering and DevOps integration**.

- **Robust Data Validation:**  
  Implemented custom validation logic using `data_validator.py` and schema-driven checks with `config/data_schema.yaml` to ensure high data quality before modeling.

- **Cloud-Native & Scalable Infrastructure:**  
  Built using **containerized microservices (Docker)**, **MLflow** for experiment tracking, and **CI/CD workflows (GitHub Actions)**, ensuring **reproducibility**, **maintainability**, and **rapid adaptation**.

- **Model Explainability & Trust:**  
  Integrated **SHAP** for interpretable predictions, empowering stakeholders to understand and trust model outcomes.

- **Robust Architecture:**  
  Designed **modular**, **well-logged**, **API-driven components** with **clear separation of concerns** for scalability and easy integration.

- **CI/CD & Monitoring Excellence:**  
  Established **continuous integration**, **testing**, and **deployment pipelines** with real-time monitoring ensuring reliability and consistent model performance at scale.

- **Experiment Tracking:**  
  **MLflow-based tracking** and comparison of all model runs, hyperparameters, and results.

- **Modern Cloud Deployment:**  
  Production API deployed using **Render**, supporting scalability and easy updates.

---

## **Quick Tech Summary for Engineers**

- **Modular architecture** leveraging **Python**, **Flask APIs**, and **Dockerized deployment**
- **Automated CI/CD** with **GitHub Actions**, **unit tests**, and **data validation checks**
- **Full-featured pipeline:**  
  **data cleaning** → **feature engineering** → **training** → **inference** → **explainability**
- **Custom validation logic** in `data_validator.py` and schema in `config/data_schema.yaml`
- **Experiment management** with **MLflow** for reproducible experiments and model registry
- **API and model deployment automated via Render**

---

## **Table of Contents**

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
- [MLOps Project - Challenges & Solutions](#mlops-project-challenges--solutions)
- [Contact](#contact)

---

## **Project Overview**

This project implements a **customer churn prediction system** designed with **production-grade MLOps practices**. The system predicts whether a customer will leave a service (churn) based on behavioral and demographic data.

- Schema-driven data validation using `data_validator.py` and `config/data_schema.yaml`
- Modular, testable pipeline architecture
- REST API for real-time predictions
- Docker containerization for deployment
- MLflow experiment tracking and model registry
- Production API deployed on Render
- CI/CD automation with GitHub Actions
- Model explainability via SHAP
- Comprehensive logging and error handling

---

## **Why This Project Matters**

**Business Value:**  
Customer churn directly impacts **revenue and growth**. This system enables businesses to:

- **Predict churn proactively:** Identify at-risk customers before they leave
- **Optimize retention strategies:** Target interventions based on SHAP-driven insights into churn drivers
- **Reduce operational risk:** Automated validation catches data issues before they affect predictions
- **Scale efficiently:** Containerized deployment enables easy scaling across environments

---

## **Tech Stack**

### **Core ML & Data Processing**

- **Python 3.10:** Primary programming language
- **Pandas & NumPy:** Data manipulation and numerical operations
- **scikit-learn:** Train/test splitting, preprocessing utilities, evaluation metrics
- **XGBoost:** Gradient boosting classifier for churn prediction
- **imbalanced-learn:** Handling class imbalance in churn data
- **pytest:** Unit and integration testing framework

### **API & Deployment**

- **Flask:** REST API server for inference endpoints with health checks
- **Docker:** Containerization for reproducible deployments
- **Waitress:** Production WSGI server for Flask
- **Render:** Cloud deployment platform for API hosting

### **Data Validation**

- **Custom Python validation:** `data_validator.py`
- **Schema configuration:** `config/data_schema.yaml`

### **Model Explainability**

- **SHAP:** Feature importance and per-prediction explanations for model transparency

### **CI/CD & Infrastructure**

- **GitHub Actions:** Automated testing, linting, and Docker image builds
- **isort, flake8:** Code formatting and linting
- **python-dotenv:** Environment variable management

### **Monitoring & Experiment Tracking**

- **MLflow:** Experiment tracking, model registry, and reproducible runs

---

## **Project Architecture**

```
mlops_project/
│
├── app/
│   ├── train_entrypoint.py              # Entrypoint for training pipeline
│   ├── inference_entrypoint.py          # Entrypoint for inference/prediction API
│
├── config/
│   ├── data_schema.yaml                 # YAML schema for data validation
│
├── src/
│   ├── data/
│   │   ├── clean_data.py                # Data cleaning and preprocessing logic
│   │   ├── train_test_split.py          # Train-test splitting functions
│   │
│   ├── models/
│   │   ├── train_model.py               # Model training logic
│   │   ├── evaluate_models.py           # Model evaluation and metrics computation
│   │
│   ├── utils/
│   │   ├── shap_utils.py                # SHAP explainability utilities
│       ├── data_validator.py            # Custom data validation logic 
│
├── data/
│   ├── raw/                             # Raw input data (e.g., Customer-Churn-Records.csv)
│   ├── cleaned/                         # Cleaned datasets after preprocessing
│   ├── processed/                       # Processed datasets ready for modeling
│
├── models/
│   ├── xgboost_model.pkl                # Saved trained model(s)
│
├── notebooks/
│   ├── 01_load_and_validate.ipynb           
│   ├── 02_clean_and_transform_raw_data.ipynb
│   ├── 03_train_test_split.ipynb          
│   ├── 03_train_random_forest.ipynb
│   ├── 03_train_xgb.ipynb 
│
├── test/
│   ├── feature_pipeline_test.py           # Unit tests for model feature pipeline
│
├── .github\
│   ├── workflows\
│       ├── ci.yml 
│       ├── cd.yml 
├── .env                                 # Environment variables (paths, configs)
├── Dockerfile
├── .dockerignore
├── .env.docker
├── entrypoint-wrapper.sh
├── run.sh
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview and documentation
├── .gitignore                           # Ignored files and directories
└── venv/                                # Virtual environment (optional, not tracked)
```
---

## **Feature & Training Pipeline**

**End-to-End Pipeline Flow**  
The training pipeline is orchestrated through `app/train_entrypoint.py` and executes the following stages:

### **1. Data Validation**

- Data is validated using **custom logic in `data_validator.py`**, powered by the **schema defined in `config/data_schema.yaml`**
- Ensures data types, required columns, value ranges, and integrity before processing

### **2. Data Loading**

- Raw CSV loaded from `data/raw/Customer-Churn-Records.csv`

### **3. Data Cleaning & Preprocessing**

- Drops unnecessary columns (*RowNumber, Surname, Complain, CustomerId for training*)
- Renames columns for consistency (*e.g., Satisfaction Score → SatisfactionScore*)
- One-hot encodes Geography and Gender
- Ordinal encodes CardType (*SILVER=0, GOLD=1, PLATINUM=2, DIAMOND=3*)

### **4. Train-Test Split**

- Stratified split to preserve churn ratio (*default: 90/10*)
- Saves both CSV and pickle formats

### **5. Model Training**

- **XGBoost classifier** with configured hyperparameters
- Training logged with comprehensive metrics

### **6. Model Evaluation**

- Computes **F1, F2, Precision, Recall, Confusion Matrix**
- Generates **ROC and Precision-Recall curves**

### **7. Model Persistence**

- Saves model as `.pkl` file
---


## **How to Run**

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
    - All runs and metrics will be tracked in **MLflow**.


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

9. **API Deployment on Render**
    - Connect this repo to your **Render** account.
    - Create a new **Web Service**, set build and start commands:
        - **Build Command:** `pip install -r requirements.txt`
        - **Start Command:** `python app/inference_entrypoint.py`
    - Add environment variables/secrets in Render dashboard.
    - Redeploy on new commits automatically.

---

## **CI/CD & Testing**

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

## **Model Explainability & Validation**

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

## **Experiment Tracking with MLflow**

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

## **Deployment on Render**

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

## MLOps Project - Challenges & Solutions

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
Transitioning from a simple XGBoost notebook to a modular pipeline was challenging - implementing abstract classes, data cleaning, train/test split logic, and proper separation of concerns.

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

## **Deployment issues**
**Problem:**
While deploying the model on Render, the container couldn’t locate the model file and inference data. Initially, it appeared that the Docker image didn’t include these assets because I had mounted local directories for persistent access.

**Solution:**
After debugging and checking the Dockerfile, I discovered the issue was with .dockerignore-it excluded both the data/ folder and .pkl model files. I updated it to ignore all data by default but explicitly keep the inference file and model artifacts. Once fixed, the model deployed successfully and predictions for customer IDs worked as expected.


**Result:**
Fully automated deployment pipeline from code commit to DockerHub.

---


## **Recommended Future Enhancements**

### **Monitoring & Alerting (Short-Term Priority)**

**Why:** Critical for production ML - you can’t manage what you can’t measure.
**What to implement:**

* **Prometheus + Grafana** (or lightweight alternatives like `prometheus_client` + Render metrics)
* Track model latency, prediction volume, and errors
* Add **data drift** monitoring using **Evidently AI**

→ *Gives visibility into model behavior & ensures reliability.*

---

### **Model Retraining Pipeline**

**Why:** Customer churn data changes over time - retraining keeps your model relevant.
**What to implement:**

* A retraining script triggered manually or on a schedule (e.g. GitHub Actions + cron)
* Save model artifacts with versioning (`models/model_v2.pkl`)
* Optional: A/B testing between new and old models before replacing

→ *Adds automation and reduces manual overhead.*

---

### **Enhanced Logging**

**Why:** Essential for debugging and auditability.
**What to implement:**

* Use **structured JSON logs**
* Store logs in a simple cloud logging solution (e.g. ELK Stack or Render logs + S3 backup)
* Log prediction inputs (hashed/anonymized), outputs, and latency

→ *Helps detect issues, bias, or misuse.*

---

### **API Enhancements**

**Why:** Improve usability and scalability of your inference service.
**What to implement:**

* **Batch prediction endpoint** (`/predict-batch`)
* **JWT authentication** for API security
* **OpenAPI/Swagger** docs for clear usage
* Optional: **Celery + Redis** if you add async predictions later

→ *Makes your model API production-grade.*


---


---

## **Summary of Key Learnings**

- **Mastered end-to-end MLOps pipeline development**, from robust data engineering to scalable model deployment in cloud environments.
- **Built advanced CI/CD automation and containerization workflows**, enabling seamless, enterprise-grade ML releases.
- **Developed deep expertise in ML experiment tracking, model versioning, and reproducible research** using industry-standard tools like MLflow and Docker.
- **Refined system design and troubleshooting skills**, consistently overcoming real-world challenges in automation, scalability, and reliability.
- **Proven ability to deliver production-ready ML solutions**, positioned for immediate impact in demanding, fast-paced industry settings.
---

## **Contact**

- **Author:** Munna  
- **GitHub:** [Munna-Git](https://github.com/Munna-Git)  
- **Email:** [munna88mn@gmail.com](mailto:munna88mn@gmail.com)