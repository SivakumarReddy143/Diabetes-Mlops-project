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

"""
DATA INGESTION constants start with DATA INGESTION VAR NAME
"""

DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_DATABASE_NAME: str = "SIVAKUMAR"
DATA_INGESTION_COLLECTION_NAME: str = "diabetes_data"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

