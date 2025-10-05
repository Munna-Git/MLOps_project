# Project structure

```
mlops_project/
│
├── app/
│   ├── train_entrypoint.py              # Entrypoint for training pipeline
│   ├── inference_entrypoint.py          # Entrypoint for inference/prediction API
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
│   ├── 01_load_and_validate.ipynb           # Initial data exploration and EDA
│   ├── 02_clean_and_transform_raw_data.ipynb
│   ├── 03_train_test_split.ipynb          # Model experimentation and tuning
│   ├── 03_train_random_forest.ipynb
│   ├── 03_train_xgb.ipynb 
│
├── test/
│   ├── feature_pipeline_test.py           # Unit tests for model feature pipeline
│
├── .github\
│   ├── workflows\
│       ├── ci.yml 
├── .env                                 # Environment variables (paths, configs)
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview and documentation
├── .gitignore                           # Ignored files and directories
└── venv/                                # Virtual environment (optional, not tracked)
```