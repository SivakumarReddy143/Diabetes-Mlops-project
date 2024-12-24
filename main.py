from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact
from diabetes.components.data_ingestion import DataIngestion
import os
import sys
import pandas as pd
import numpy as np

if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise DiabetesException(e,sys)
