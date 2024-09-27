import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utility.utility import save_object,evaluate_model
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','models.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_conifg=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,:-1],
                train_array[:,:-1],
                train_array[:,:-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n','='*40,'\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n','='*40,'\n')
            logging.info(f'Best model is found, Model Name : {best_model_name}, R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_conifg.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("Exception occured at initiate_model_training")
            raise customexception(e,sys)