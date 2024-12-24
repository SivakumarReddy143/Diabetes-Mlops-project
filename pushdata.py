from diabetes.logging.logger import logging
from diabetes.exception.exception import DiabetesException
import sys
import os
import pandas as pd
import numpy as np
import pymongo
import json
from dotenv import load_dotenv
load_dotenv()
import certifi
certifi.where()

MONGO_DB_URL=os.getenv('MONGO_DB_URL')

class DiabetesDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def csv_to_json_convertor(self,file_path):
        try:
            df=pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records=list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise DiabetesException(e,sys)
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise DiabetesException(e,sys)

if __name__=="__main__":
    file_path="data/diabetes.csv"
    data=DiabetesDataExtract()
    records=data.csv_to_json_convertor(file_path=file_path)
    print(records)
    database="SIVAKUMAR"
    collection="diabetes_data"
    data.insert_data_mongodb(records=records,database=database,collection=collection)
    print(len(records))
    