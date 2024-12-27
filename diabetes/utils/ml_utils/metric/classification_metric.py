from diabetes.exception.exception import DiabetesException
from diabetes.logging.logger import logging
from diabetes.entity.artifact_entity import ClassificationMetric
from sklearn.metrics import recall_score,precision_score,f1_score
import sys

def get_classification_score(y_true,y_pred):
    try:
        model_recall_score=recall_score(y_true,y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_f1_score=f1_score(y_true,y_pred)
        classification_metric=ClassificationMetric(f1_score=model_f1_score,
                                                   recall_score=model_recall_score,
                                                   precision_score=model_precision_score
                                                   )
        return classification_metric
    except Exception as e:
        raise DiabetesException(e,sys)