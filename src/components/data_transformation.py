import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exceptions import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utility.utility import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            logging.info('Data transformation initiated')

            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline initiated')

            num_pipeline = Pipeline(
                steps=[
                    # SimpleImputer is used to replace missing values with median values
                    ('imputer', SimpleImputer(strategy='median')),
                    # StandardScalar transforms data to have a mean of 0 and a standard deviation of 1
                    ('scalar', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # OrdinalEncoder converts categorical features into numerical format
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                # Transformer parameter contains name, transformer object, columns
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            preprocessing_obj = self.initiate_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            
            # axis=0 refers to rows & axis=1 refers to columns
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # fit_transform() learns the parameters like SimpleImputer but transform() doesn't recalculate them
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            # np.c_[] allows you to combine multiple arrays into a single 2D array
            # Here we combine every other columns with price column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)