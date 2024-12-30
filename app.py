import os
import sys

import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_uri=os.getenv("MONGO_DB_URL")
print(mongo_db_uri)
import pymongo
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.pipeline.training_pipeline import TrainingPipeline
from diabetes.utils.main_utils.utils import load_object
from diabetes.utils.ml_utils.model.estimator import DiabetesModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

app=FastAPI()

origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise DiabetesException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessing.pkl")
        final_model=load_object("final_model/model.pkl")
        diabetes_model = DiabetesModel(preprocess=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = diabetes_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise DiabetesException(e,sys)

if __name__=="__main__":
    app_run(app,host="localhost",port=8000)