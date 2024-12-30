import os
import pandas as pd
import numpy as np
import sys

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN="Outcome"
TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"
ARTIFACT_DIR="Artifacts"
PIPELINE="pipeline"
FILE_NAME="diabetes.csv"
SCHEMA_FILE_PATH= os.path.join("data_schema","schema.yaml")
SAVED_MODEL_DIR:str = "saved_models"
MODEL_FILE_NAME:str = "model.pkl"

"""
DATA INGESTION constants start with DATA INGESTION VAR NAME
"""

DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_DATABASE_NAME: str = "SIVAKUMAR"
DATA_INGESTION_COLLECTION_NAME: str = "diabetes_data"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data validation constants start with DATA-VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "valid"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

"""
Data Transformation constants start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"
DATA_TRANSFORMATION_IMPUTER_PARAMS:dict = {
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights":"uniform"
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH:str = "train.npy"
DATA_TRANSFORMATION__TEST_FILE_PATH:str = "test.npy"

"""
Model trainer related constants start with MODEL_TRAINER VAR NAME 
"""

MODEL_TRAINER_DIR:str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE:float = 0.6
MODEL_TRAINER_UNDERFITTING_OVERFITTING_THRESOLD:float = 0.05

TRAINING_BUCKET_NAME: str = "diabetesmlops"
