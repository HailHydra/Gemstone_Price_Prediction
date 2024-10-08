import numpy as np
import pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from urllib.parse import urlparse
import os
import sys
import mlflow
import mlflow.sklearn
from src.utility.utility import load_object
from src.logger.logging import logging
from src.exceptions.exceptions import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started")

    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        logging.info("Evaluation metrics captured")
        return rmse,mae,r2

    def initiate_model_evaluation(self,train_array,test_array):
        try:
            # x_test extracts all rows and all columns except the last column: 'price'
            # y_test extracts all rows but only the last column: 'price'
            x_test,y_test=(test_array[:,:-1],test_array[:,-1])
            
            # Constructs a file path to the saved 'models.pkl'
            model_path=os.path.join('artifacts','models.pkl')
            # Loading the model from 'models.pkl' using load_object
            model=load_object(model_path)

            # Setting the registry URI for MLFlow which manages the model registry
            # "" points to default tracking server's URI
            mlflow.set_registry_uri("")

            logging.info("Model is registered")
            
            # mlflow.get_tracking_uri() returns the tracking URI
            # Scheme part of a URI refers to the protocol used http,https,file
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            # This shows whether MLFlow is using a file-based system or HTTP-based server
            print(tracking_url_type_store)

            with mlflow.start_run():
                
                # The predicted values for x_test is stored in prediction variable
                prediction=model.predict(x_test)

                # prediction variable is passed as parameter to eval_metrics
                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)

                # Logging each values to respective variables
                mlflow.log_metric('rmse',rmse)
                mlflow.log_metric('r2',r2)
                mlflow.log_metric('mae',mae)

                # If the URI type is not file then the model is logged & registered as ml_model
                if tracking_url_type_store !="file":
                    mlflow.sklearn.log_model(model,'model',registered_model_name="ml_model")

                # If the URI type is file then it just logs the model
                else:
                    mlflow.sklearn.log_model(model,'model')

        except Exception as e:
            raise customexception(e,sys)