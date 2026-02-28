import os,  sys
import pandas as pd
import certifi
from dotenv import load_dotenv
import pymongo
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.main_utils import load_object
from src.utils.ml_utils.model.estimator import NetworkModel
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.batch_prediction import PredictionPipeline
from src.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
from src.constants.prediction_pipeline import TEMP_DATA_DIR_NAME, TEMP_DATA_FILE_NAME

predictor = PredictionPipeline(model_path="./final_models/model.pkl", preprocessor_path="./final_models/preprocessor.pkl")


load_dotenv()

ca = certifi.where()

templates = Jinja2Templates(directory="./templates")

mongodb_uri = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongodb_uri, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origin = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:

        os.makedirs(TEMP_DATA_DIR_NAME, exist_ok=True)

        temp_data_file_path = os.path.join(TEMP_DATA_DIR_NAME, TEMP_DATA_FILE_NAME)
        with open(temp_data_file_path, "wb") as f:
            f.write(file.file.read())

        df = predictor.initiate_batch_prediction(input_file_path=temp_data_file_path)
        
        os.makedirs("./prediction_output", exist_ok=True)
        df.to_csv("./prediction_output/output.csv")
        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)
