from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact
from diabetes.components.data_ingestion import DataIngestion
from diabetes.components.data_validation import DataValidation
from diabetes.components.data_transformation import DataTransformation
import os
import sys
import pandas as pd
import numpy as np

if __name__=="__main__":
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifacts,data_validation_config=data_validation_config)
        data_validation_artifacts=data_validation.initiate_data_validation()
        data_transformation_config=DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=DataTransformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifacts)
        data_transformation_artifacts=data_transformation.initiate_data_transformation()
        print(data_transformation_artifacts)
    except Exception as e:
        raise DiabetesException(e,sys)
