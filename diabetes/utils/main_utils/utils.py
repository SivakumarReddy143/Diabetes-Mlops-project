import yaml
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import sys
import os
import pickle
import numpy as np

def read_yaml(file_path):
   try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        return content
   except Exception as e:
       raise DiabetesException(e,sys)

def write_yaml(file_path, data):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        raise DiabetesException(e,sys)

def save_numpy_array_data(file_path,array:np.array):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise DiabetesException(e,sys)

def load_numpy_array_data(file_path):
    try:
        with open(file_path,'rb') as file:
            return np.load(file,allow_pickle=True)
    except Exception as e:
        raise DiabetesException(e,sys)

def save_object(file_path,obj:object):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(file=file,obj=obj)
    except Exception as e:
        raise DiabetesException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise DiabetesException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            test_model_score=r2_score(y_test,y_test_pred)
            train_model_score=r2_score(y_train,y_train_pred)
            report[list(models.keys())[i]]=test_model_score
        
        return report
            
    except Exception as e:
        raise DiabetesException(e,sys)
    