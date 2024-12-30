import os
import sys

from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging

from diabetes.components.data_ingestion import DataIngestion
from diabetes.components.data_validation import DataValidation
from diabetes.components.data_transformation import DataTransformation
from diabetes.components.model_trainer import ModelTrainer

from diabetes.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from diabetes.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
    
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(self.training_pipeline_config)
            logging.info("Started data ingestion")
            data_ingestion=DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion Artifacts: {data_ingestion_artifacts}")
            return data_ingestion_artifacts
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def start_data_validation(self,data_ingestion_artifact):
        try:
            self.data_validation_config=DataValidationConfig(self.training_pipeline_config)
            logging.info("Started data validation")
            data_validation=DataValidation(data_ingestion_artifact,self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"Data validation Artifacts: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            DiabetesException(e,sys)
    
    def start_data_transformation(self,data_validation_artifact):
        try:
            self.data_transformation_config=DataTransformationConfig(self.training_pipeline_config)
            logging.info("Started Data transformatin")
            data_transformation=DataTransformation(data_validation_artifact,self.data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformatin : {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise DiabetesException(e,sys)
        
    def model_Trainer(self,data_transformation_artifact):
        try:
            self.model_trainer_config=ModelTrainerConfig(self.training_pipeline_config)
            logging.info("Started model training")
            model_trainer=ModelTrainer(data_transformation_artifact,self.model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info(f"model Trainer : {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def run_pipeline(self):
        try:
            data_ingestion_artfact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artfact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact)
            model_trainer_artifacts=self.model_Trainer(data_transformation_artifact)
            return model_trainer_artifacts
        except Exception as e:
            raise DiabetesException(e,sys)