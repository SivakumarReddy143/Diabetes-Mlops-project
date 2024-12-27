import os
import sys
import pandas as pd
import numpy as np
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.entity.config_entity import ModelTrainerConfig
from diabetes.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from diabetes.utils.ml_utils.model.estimator import DiabetesModel
from diabetes.utils.ml_utils.metric.classification_metric import get_classification_score

from diabetes.utils.main_utils.utils import evaluate_models,save_numpy_array_data,save_object,load_object,load_numpy_array_data

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            print("_________________________________________")
            print(self.data_transformation_artifact.transformed_object_file_path)
            print("____________________________________________________________")
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params={
                 "Random Forest":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,128,256]
                },
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                } 
            }
            report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            best_model_score=max(sorted(report.values()))
            best_model_name=list(report.keys())[list(report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            y_train_pred=best_model.predict(X_train)
            classification_train_metric=get_classification_score(y_train,y_train_pred)
            y_test_pred=best_model.predict(X_test)
            classification_test_metric=get_classification_score(y_test,y_test_pred)
            
            preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            Diabetes_model=DiabetesModel(preprocess=preprocessor,model=best_model)
            os.makedirs(self.model_trainer_config.model_trainer_dir_file_path,exist_ok=True)
            save_object(self.model_trainer_config.model_trainer_model_path,obj=Diabetes_model)
            save_object('final_model/model.pkl',best_model)
            
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.model_trainer_model_path,
                                                        train_metric_artifact=classification_train_metric,
                                                        test_metric_artifact=classification_test_metric
                                                        )
            return model_trainer_artifact
            
        except Exception as e:
            raise DiabetesException(e,sys)
    
    def initiate_model_trainer(self):
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            model_trainer_artifact=self.train_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
            print(model_trainer_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise DiabetesException(e,sys)