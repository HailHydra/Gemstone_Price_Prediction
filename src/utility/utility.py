import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise customexception(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            # fit() is used to train a model on a given dataset
            model.fit(x_train,y_train)

            # predict() is used to predict the output
            y_test_pred=model.predict(x_test)

            # R2 scores for train and test data is calculated
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
        
        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function')
        raise customexception(e,sys)