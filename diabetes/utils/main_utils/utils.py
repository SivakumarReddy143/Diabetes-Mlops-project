import yaml
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
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
            return np.load(file_path)
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
        with open(file_path) as file:
            return pickle.load(file_path)
    except Exception as e:
        raise DiabetesException(e,sys)
    