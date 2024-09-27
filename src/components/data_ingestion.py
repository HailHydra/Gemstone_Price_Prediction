import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data=pd.read_csv(r"C:\Users\Varun\Music\Python\playground-series-s3e8\train.csv")
            logging.info("Reading data frame")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)))
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw dataset is saved in artifact folder")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Train test split is completed")
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Data ingestion part completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error occured at initiate_data_ingestion")
            raise customexception(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()    
    obj.initiate_data_ingestion()