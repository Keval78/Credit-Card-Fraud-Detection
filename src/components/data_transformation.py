import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, RandomUnderSampler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")
    train_data_path: str = os.path.join('artifacts', "train1.csv")
    test_data_path: str = os.path.join('artifacts', "test1.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This function is responsible for data trnasformation
        '''
        try:
            numerical_columns = ['Amount', 'Time']
            num_pipeline = Pipeline(
                steps=[
                    ("robust_scaler", RobustScaler()),
                ]
            )
            num_transformer = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            
            oversampler = SMOTE(sampling_strategy=0.25, random_state=42)
            undersampler = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3)
            sampling_pipeline = Pipeline(
                steps = [
                    ('under', undersampler), 
                    ('over', oversampler)
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            return (
                sampling_pipeline,
                num_transformer
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            sampling_obj, preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Class"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_train_df, target_feature_train_df = sampling_obj.fit_resample(input_feature_train_df, target_feature_train_df)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            num_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            num_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            numerical_columns = ['Amount', 'Time']
            input_feature_train_df.drop(columns=numerical_columns)
            input_feature_test_df.drop(columns=numerical_columns)

            input_feature_train_df[numerical_columns] = num_feature_train_arr[:,:len(numerical_columns)]
            input_feature_test_df[numerical_columns] = num_feature_test_arr[:,:len(numerical_columns)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            input_feature_train_df.to_csv(
                self.data_transformation_config.train_data_path, index=False, header=True)

            input_feature_test_df.to_csv(
                self.data_transformation_config.test_data_path, index=False, header=True)

            train_arr = np.c_[
                np.array(input_feature_train_df), np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                np.array(input_feature_test_df), np.array(target_feature_test_df)
            ]
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
