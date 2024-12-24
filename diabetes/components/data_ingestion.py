from diabetes.logging.logger import logging
from diabetes.exception.exception import DiabetesException
from diabetes.entity.config_entity import DataIngestionConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise DiabetesException(e,sys)
    def export_collection_as_dataframe(self)->pd.DataFrame:
        try:
            mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            collection=mongo_client[database_name][collection_name]
            dataframe=pd.DataFrame(list(collection.find()))
            if '_id' in dataframe.columns:
                dataframe=dataframe.drop(columns=['_id'],axis=1)
            dataframe.replace({'na':np.nan},inplace=True)
            return dataframe
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def export_dataframe_to_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_data,test_data=train_test_split(dataframe,
                                                  test_size=self.data_ingestion_config.train_test_split_ratio
                                                  )
            os.makedirs(self.data_ingestion_config.ingested_dir,exist_ok=True)
            train_data.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
        except Exception as e:
            raise DiabetesException(e,sys)
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_dataframe_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifacts=DataIngestionArtifact(train_file_path=self.data_ingestion_config.train_file_path,
                                                           test_file_path=self.data_ingestion_config.test_file_path
                                                           )
            print(data_ingestion_artifacts)
            return data_ingestion_artifacts
        except Exception as e:
            raise DiabetesException(e,sys)