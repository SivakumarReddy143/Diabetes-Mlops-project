import os
import sys
from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging

class DiabetesModel():
    def __init__(self,preprocess,model):
        try:
            self.preprocessor=preprocess
            self.model=model
        except Exception as e:
            raise DiabetesException(e,sys)
        
    def predict(self,x):
        try:
            x_transform=self.preprocessor.transform(x)
            y_hat=self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise DiabetesException(e,sys)