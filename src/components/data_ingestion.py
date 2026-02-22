import os
import sys
import pandas as pd
import numpy as np
from typing import List
import pymongo

from sklearn.model_selection import train_test_split

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


from dotenv import load_dotenv
load_dotenv()

MONGODB_URI=os.getenv("MONGODB_URI")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def read_data(self) -> pd.DataFrame:
        try:
            logging.info("Reading data from mongodb")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            mongo_client = pymongo.MongoClient(MONGODB_URI)

            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            df = df.replace("na", np.nan)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def store_data_to_feature_store(self, df: pd.DataFrame):
        try:
            logging.info("Exporting data as csv file")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index=False)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path.")

            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            df = self.read_data()
            df = self.store_data_to_feature_store(df=df)
            self.split_data_as_train_test(df=df)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            logging.info("Data Ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)