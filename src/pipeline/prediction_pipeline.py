import os
import sys
import pandas as pd
from src.exceptions.exceptions import customexception
from src.logger.logging import logging
from src.utility.utility import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:

            # Paths to artifacts
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','models.pkl')
            
            # preprocessor is loaded with preprocessor object
            preprocessor=load_object(preprocessor_path)
            # model is loaded with model object
            model=load_object(model_path)

            # Preprocessing steps like scaling, encoding is done
            scale_feat=preprocessor.transform(features)
            # Pre-trained model is used to make predictions
            pred=model.predict(scale_feat)

            return pred

        except Exception as e:
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)