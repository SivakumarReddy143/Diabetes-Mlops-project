import yaml
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
import sys

def read_yaml(file_path):
   try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        return content
   except Exception as e:
       raise DiabetesException(e,sys)

def write_yaml(file_path, data):
    try:
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        raise DiabetesException(e,sys)
    