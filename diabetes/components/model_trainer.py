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
from sklearn.tree import DecisionTreeClassifier

from diabetes.utils.ml_utils.model.estimator import DiabetesModel
from diabetes.utils.ml_utils.metric.classification_metric import get_classification_score

from diabetes.utils.main_utils.utils import evaluate_models,save_numpy_array_data,save_object,load_object,load_numpy_array_data

import mlflow
import dagshub
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature

# dagshub.init(repo_owner='mshivakumarreddy78', repo_name='Diabetes-Mlops-project', mlflow=True)
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/mshivakumarreddy78/Diabetes-Mlops-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]='mshivakumarreddy78'
os.environ["MLFLOW_TRACKING_PASSWORD"]='20e6c06a55dd1445e742e9f17844edbcff441e47'

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise DiabetesException(e,sys)
        
    def track_mlflow(self,best_model,classification_metric):
        try:
            mlflow.set_tracking_uri("https://dagshub.com/mshivakumarreddy78/Diabetes-Mlops-project.mlflow")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():
                f1_score=classification_metric.f1_score
                precision_score=classification_metric.precision_score
                recall_score=classification_metric.recall_score
                mlflow.log_metric('f1_score',f1_score)
                mlflow.log_metric('precission_score',precision_score)
                mlflow.log_metric('recall_score',recall_score)
                mlflow.sklearn.log_model(best_model,'model')
                # signature=infer_signature(X,best_model.predict(X))
                if tracking_url_type_store!="file":
                    mlflow.sklearn.log_model(best_model,'model',registered_model_name=best_model)
                else:
                    mlflow.sklearn.log_model(best_model,'model')
                
        except Exception as e:
            DiabetesException(e,sys)
    
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
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
            self.track_mlflow(best_model=best_model,classification_metric=classification_train_metric)
            y_test_pred=best_model.predict(X_test)
            classification_test_metric=get_classification_score(y_test,y_test_pred)
            self.track_mlflow(best_model=best_model,classification_metric=classification_test_metric)
            
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