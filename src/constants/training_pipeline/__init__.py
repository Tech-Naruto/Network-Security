import os
import sys
import numpy as np
import pandas as pd


"""
Common Constants
"""
PIPELINE_NAME: str = "phishing-detection"
TARGET_COLUMN: str = "CLASS_LABEL"
ARTIFACT_DIR: str = "artifacts"
RAW_FILE_NAME: str = "phishing_data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")


"""
Constants for Data Ingestion
"""

DATA_INGESTION_DATABASE_NAME: str = "phishing-detection"
DATA_INGESTION_COLLECTION_NAME: str = "phishing_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Constants for Data Validation
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"