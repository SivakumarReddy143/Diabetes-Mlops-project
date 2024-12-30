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
        self.model_dir=os.path.join("final_model")
        self.artifacts_dir=os.path.join(artifacts_name,timestamp)
        self.timestamp=timestamp
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


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifacts_dir,
                                                     training_pipeline.DATA_VALIDATION_DIR
                                                     )
        self.valid_dir:str = os.path.join(self.data_validation_dir,
                                          training_pipeline.DATA_VALIDATION_VALID_DIR
                                          )
        self.valid_train_file_path:str = os.path.join(self.valid_dir,
                                                      training_pipeline.TRAIN_FILE_NAME
                                                      )
        self.valid_test_file_path:str = os.path.join(self.valid_dir,
                                                     training_pipeline.TEST_FILE_NAME
                                                     )
        self.invalid_dir:str = os.path.join(self.data_validation_dir,
                                            training_pipeline.DATA_VALIDATION_INVALID_DIR
                                            )
        self.invalid_train_file_path:str = os.path.join(self.invalid_dir,
                                                        training_pipeline.TRAIN_FILE_NAME
                                                        )
        self.invalid_test_file_path:str = os.path.join(self.invalid_dir,
                                                       training_pipeline.TEST_FILE_NAME
                                                       )
        self.drift_report_dir:str = os.path.join(self.data_validation_dir,
                                                 training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
                                                 ) 
        self.drift_report_file_path:str = os.path.join(self.drift_report_dir,
                                                       training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
                                                       )

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir=os.path.join(training_pipeline_config.artifacts_dir,
                                                  training_pipeline.DATA_TRANSFORMATION_DIR
                                                  )
        self.data_transformed_dir=os.path.join(self.data_transformation_dir,
                                               training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR
                                               )
        self.data_transformation_object_dir=os.path.join(self.data_transformation_dir,
                                                         training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
                                                         )
        self.train_numpy_array_file_path=os.path.join(self.data_transformed_dir,
                                                      training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_PATH
                                                      )
        self.test_numpy_array_file_path=os.path.join(self.data_transformed_dir,
                                                     training_pipeline.DATA_TRANSFORMATION__TEST_FILE_PATH
                                                     )
        self.preprocessing_object_file_path=os.path.join(self.data_transformation_object_dir,
                                                         training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
                                                         )
        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir=os.path.join(training_pipeline_config.artifacts_dir,
                                            training_pipeline.MODEL_TRAINER_DIR
                                            )
        self.model_trainer_dir_file_path=os.path.join(self.model_trainer_dir,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR)
        self.model_trainer_model_path=os.path.join(self.model_trainer_dir_file_path,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.underfit_overfit_thresold=training_pipeline.MODEL_TRAINER_UNDERFITTING_OVERFITTING_THRESOLD
        self.expected_accuracy=training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        
        