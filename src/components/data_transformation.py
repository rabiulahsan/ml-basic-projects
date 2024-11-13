import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = 'artifacts/preprocessor.pkl'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def DataTransformerPipeline(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            category_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #?mode
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    #? ("pipeline_name", pipeline, colums which it will be applied)
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("category_pipeline", category_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            #? read the data from path
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            #? pipeline for column transforming 
            preprocessing_obj = self.DataTransformerPipeline()

            target_column_name="math_score"
            # numerical_columns = ["writing_score", "reading_score"]

            #? The code specifies math_score as the target (dependent variable), which is what the model will predict.
            input_feature_train_data=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_data=train_data[target_column_name]

            input_feature_test_data=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_data=test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            #? fit_transform the both train and test data according to the pipeline
            #? fit_transform is used for training data
            #?transform is used on test data
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_data)


            #?
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_data)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            logging.info(f"Saved preprocessing object.")

            #? for saving the file. it imports from the utility 
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )

            logging.info("saving the model has completed.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(e)
