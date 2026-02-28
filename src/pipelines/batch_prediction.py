import os, sys
import pandas as pd

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.main_utils import load_object
from src.utils.ml_utils.model.estimator import NetworkModel


class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_batch_prediction(self, input_file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(input_file_path)

            network_model = NetworkModel(preprocessor=self.preprocessor, model=self.model)

            y_pred = network_model.predict(X=df)

            df["prediction"] = y_pred

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)