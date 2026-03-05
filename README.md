# **Phishing Website Detection: Enterprise MLOps & API Pipeline**

## **1. Project Overview**
This project is an end-to-end Machine Learning solution designed to identify phishing URLs with high precision. It implements a professional **MLOps lifecycle** including Cloud Data Persistence (**MongoDB Atlas**), Experiment Tracking (**MLflow/Dagshub**), and an asynchronous **FastAPI** service for batch predictions.



## **2. Technical Stack**
* **Frameworks:** FastAPI, Uvicorn (ASGI Server)
* **Database:** MongoDB Atlas (Cloud NoSQL)
* **MLOps:** MLflow + Dagshub
* **ML Libraries:** Scikit-Learn (Ensemble & Linear Classifiers)
* **Environment:** Python 3.x

## **3. System Architecture**

### **A. Data Migration & Ingestion Layer**
The system uses a two-step process to handle data professionally:
* **Initial Migration (`push_data.py`):** An ETL script that migrates raw URL features from the local CSV to **MongoDB Atlas**, creating a centralized cloud "Source of Truth."
* **Data Ingestion Component (`src/components/data_ingestion.py`):** A dedicated pipeline component that fetches data from MongoDB, handles initial formatting, and feeds the training pipeline. This ensures the training logic is decoupled from the database connection logic.

### **B. Training & Serialization Pipeline (`main.py`)**
This orchestrator manages the training components. It evaluates 6 models (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost), logging all metadata to **MLflow** via `dagshub.init()`. The "Best Model" and preprocessor are exported to the `final_models/` directory.

### **C. Inference & Batch Prediction Pipeline (`app.py`)**
A **FastAPI** service serves as the deployment layer, utilizing a `PredictionPipeline` class:
* **Local Model Serving:** Loads artifacts directly from `final_models/` for zero-latency inference.
* **Batch Efficiency:** Optimized to classify multiple URL records in a single pass via **Uvicorn**, making it suitable for large-scale security log scanning.


## **4. Project Structure**
```text
├── data_schema/         # Defines validation rules for ingestion
├── final_models/        # Serialized production binaries (.pkl)
├── Network_Data/        # Raw source datasets (CSV format)
├── prediction_output/   # Results from batch prediction jobs
├── src/                 # Source code directory
├── temp_data/           # Cache: Intermediate storage for batch processing
├── main.py              # Training pipeline & MLOps (MLflow)
├── app.py               # FastAPI REST interface for inference
├── push_data.py         # Local-to-Cloud (MongoDB Atlas) migration
├── setup.py             # Project packaging & dependency mapping
├── requirements.txt     # Comprehensive dependency manifest
└── .env                 # Encrypted secrets