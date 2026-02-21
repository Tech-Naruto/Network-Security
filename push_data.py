import os
import sys
import json

from dotenv import load_dotenv
import certifi
import pandas as pd
import numpy as np
import pymongo

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging


load_dotenv()

MONGODB_URI=os.getenv("MONGODB_URI")

ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json(self, file_path):
        try:
            logging.info("Converting csv to json")
            data = pd.read_csv(file_path)
            data.drop(columns=['id'], inplace=True)
            records = list(json.loads(data.to_json(orient='index')).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records
            
            logging.info("Inserting data to MongoDB")
            self.mongo_client = pymongo.MongoClient(MONGODB_URI)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    FILE_PATH = "Network_Data/Phishing_Legitimate.csv"
    DATABASE = "phishing-detection"
    COLLECTION = "phishing_data"

    logging.info("Pushing data to MongoDB")
    network_obj = NetworkDataExtract()
    records = network_obj.csv_to_json(file_path=FILE_PATH)
    no_of_records = network_obj.insert_data_to_mongodb(records=records, database=DATABASE, collection=COLLECTION)
    print(f"Inserted {no_of_records} records to MongoDB")

    logging.info("Pushing data to MongoDB completed")