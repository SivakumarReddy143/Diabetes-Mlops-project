from diabetes.logging.logger import logging
from diabetes.exception.exception import DiabetesException
from diabetes.constant import training_pipeline
import os
import sys
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        artifacts_name=training_pipeline.ARTIFACT_DIR
        self.pipeline_name=training_pipeline.PIPELINE
        self.artifacts_dir=os.path.join(artifacts_name,timestamp)
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(training_pipeline_config.artifacts_dir,
                                                   training_pipeline.DATA_INGESTION_DIR
                                                   )
        self.feature_store_dir: str = os.path.join(self.data_ingestion_dir,
                                                   training_pipeline.DATA_INGESTION_FEATURE_STORE
                                                   )
        self.feature_store_file_path: str = os.path.join(self.feature_store_dir,
                                                         training_pipeline.FILE_NAME
                                                         )
        self.ingested_dir: str = os.path.join(self.data_ingestion_dir,
                                              training_pipeline.DATA_INGESTION_INGESTED_DIR
                                              )
        self.train_file_path: str = os.path.join(self.ingested_dir,
                                                 training_pipeline.TRAIN_FILE_NAME
                                                 )
        self.test_file_path: str = os.path.join(self.ingested_dir,
                                                training_pipeline.TEST_FILE_NAME
                                                )
        self.database_name:str = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.train_test_split_ratio: float =training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        