import os, sys
import pandas as pd
from scipy.stats import ks_2samp

from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.main_utils import read_yaml, write_yaml


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info("Reading dataset")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating number of columns")
            no_of_columns = len(self._schema_config["columns"])
            return len(df.columns) == no_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_numerical_columns(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating numerical columns")
            numerical_columns = self._schema_config["numerical_columns"]

            for col in numerical_columns:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    return False
            
            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            logging.info("Detecting dataset drift")
            status = False
            report = {}

            for column in base_df.columns:
                base = base_df[column]
                curr = current_df[column]

                sample_dist = ks_2samp(base, curr)

                if sample_dist.pvalue > threshold:
                    drift_status = False
                else :
                    drift_status = True
                    status = True

                report.update({column: {
                    "p_value": float(sample_dist.pvalue),
                    "drift_status": drift_status
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = self.data_validation_config.drift_report_dir
            os.makedirs(dir_path, exist_ok=True)

            write_yaml(file_path=drift_report_file_path, content=report)

            if status:
                logging.info("Data drift detected")
            else:
                logging.info("Data drift not detected")

            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)
            
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(file_path=train_file_path)
            test_df = DataValidation.read_data(file_path=test_file_path)
            
            # Validate number of columns and numerical columns
            validation_status = self.validate_number_of_columns(df=train_df)

            if not validation_status:
                error_message = "Your train set has more or less columns than expected"
                raise Exception(error_message)

            validation_status = self.validate_number_of_columns(df=test_df)

            if not validation_status:
                error_message = "Your test set has more or less columns than expected"
                raise Exception(error_message)

            validation_status = self.validate_numerical_columns(df=train_df)

            if not validation_status:
                error_message = "Your train set has missing or non numerical columns"
                raise Exception(error_message)

            validation_status = self.validate_numerical_columns(df=test_df)

            if not validation_status:
                error_message = "Your test set has missing or non numerical columns"
                raise Exception(error_message)

            
            # Check for data drift
            data_drift_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            valid_dir_path = self.data_validation_config.valid_data_dir
            invalid_dir_path = self.data_validation_config.invalid_data_dir
            os.makedirs(valid_dir_path, exist_ok=True)
            os.makedirs(invalid_dir_path, exist_ok=True)

            if not data_drift_status:
                logging.info("Saving valid data to valid directory")
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            else:
                logging.info("Saving invalid data to invalid directory")
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
    
            data_validation_artifact = DataValidationArtifact(
                validation_status=data_drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info("Data validation completed")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
