import os, sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info("Reading dataset")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            
            processor = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation")
            train_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df  = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Getting preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessor object")
            transformed_input_train_features = preprocessor_obj.fit_transform(input_feature_train_df)
            transformed_input_test_features = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Concatenating input features and target features")
            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]

            logging.info("Saving numpy array data")
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saving preprocessor object")
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info("Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

