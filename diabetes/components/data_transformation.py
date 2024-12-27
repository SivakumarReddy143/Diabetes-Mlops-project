import numpy as np
import pandas as pd
import os
import sys
from diabetes.logging.logger import logging
from diabetes.exception.exception import DiabetesException
from diabetes.entity.config_entity import DataTransformationConfig
from diabetes.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from diabetes.constant.training_pipeline import TARGET_COLUMN,DATA_TRANSFORMATION_IMPUTER_PARAMS
from diabetes.utils.main_utils.utils import (save_numpy_array_data,
                                             save_object,
                                             load_numpy_array_data,
                                             load_object)
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise DiabetesException(e,sys)
    
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise DiabetesException(e,sys)
    def get_data_transformer_object(cls)->Pipeline:
        try:
            logging.info(f"Initialie KNNImputer with:{DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            preprocess:Pipeline=Pipeline([('impute',imputer)])
            return preprocess
        except Exception as e:
            DiabetesException(e,sys)
    
    def initiate_data_transformation(self):
        try:
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            # train_dataframe
            train_input_features=train_df.drop(columns=['Outcome'],axis=1)
            train_target_column=train_df['Outcome']
            
            # test_dataframe
            test_input_feature=test_df.drop(columns=['Outcome'],axis=1)
            test_target_column=test_df['Outcome']
            
            preprocess=self.get_data_transformer_object()
            preprocess_object=preprocess.fit(train_input_features)
            
            transformed_train_input_features=preprocess_object.transform(train_input_features)
            transformed_test_input_feature=preprocess_object.transform(test_input_feature)
            
            train_arr=np.c_[transformed_train_input_features,np.array(train_target_column)]
            test_arr=np.c_[transformed_test_input_feature,np.array(test_target_column)]
            
            save_numpy_array_data(self.data_transformation_config.train_numpy_array_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.test_numpy_array_file_path,test_arr)
            save_object(file_path=self.data_transformation_config.preprocessing_object_file_path,obj=preprocess_object)
            
            save_object('final_model/preprocessing.pkl',preprocess_object)
            data_transformation_artifacts=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.preprocessing_object_file_path,
                transformed_test_file_path=self.data_transformation_config.test_numpy_array_file_path,
                transformed_train_file_path=self.data_transformation_config.train_numpy_array_file_path
            )
            
            return data_transformation_artifacts
        except Exception as e:
            raise DiabetesException(e,sys)
        