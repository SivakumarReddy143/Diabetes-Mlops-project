from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.entity.config_entity import DataValidationConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from diabetes.constant.training_pipeline import SCHEMA_FILE_PATH
from diabetes.utils.main_utils.utils import read_yaml,write_yaml
from scipy.stats import ks_2samp
import os
import sys
import pandas as pd

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifacts=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise DiabetesException(e,sys)
        
    @staticmethod
    def readData(file_path)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame):
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"required number of columns:{number_of_columns}")
            logging.info(f"dataframe has columns: {len(dataframe)}")
            if(len(dataframe)==number_of_columns):
                return True
            return False
        except Exception as e:
            raise DiabetesException(e,sys)
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml(file_path=drift_report_file_path,data=report)

        except Exception as e:
            raise DiabetesException(e,sys)
    def initiate_data_validation(self):
        try:
            train_file_path=self.data_ingestion_artifacts.train_file_path
            test_file_path=self.data_ingestion_artifacts.test_file_path
            
            train_df=DataValidation.readData(train_file_path)
            test_df=DataValidation.readData(test_file_path)
            
            status=self.validate_number_of_columns(train_df)
            if not status:
                logging.info("train data does not contain all columns")
            status=self.validate_number_of_columns(test_df)
            if not status:
                logging.info("test dataframe does not contain all columns")
            self.detect_dataset_drift(train_df,test_df)
            os.makedirs(self.data_validation_config.valid_dir)
            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            data_validation_artifacts=DataValidationArtifact(
                validation_status=status,
                valid_test_file_path=self.data_ingestion_artifacts.test_file_path,
                valid_train_file_path=self.data_ingestion_artifacts.test_file_path,
                invalid_test_file_path=None,
                invalid_train_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            print(data_validation_artifacts)
            return data_validation_artifacts
        except Exception as e:
            raise DiabetesException(e,sys)