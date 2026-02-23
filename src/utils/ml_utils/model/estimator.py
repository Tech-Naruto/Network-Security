import os, sys

from src.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def predict(self, X):
        try:
            X_transform = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transform)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys)