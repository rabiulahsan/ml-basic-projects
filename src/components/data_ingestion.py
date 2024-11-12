import os 
import sys
from  src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split


class DataIngestionConfig:
    train_data_path:str = 'artifacts/train.csv'
    test_data_path:str = 'artifacts/test.csv'
    raw_data_path:str = 'artifacts/data.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion has started...")
        try :
            df = pd.read_csv(r'notebook\data\student.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index= False, header=True)

            logging.info("Train test split initiated...")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of data has been completed.")

            return self.ingestion_config.test_data_path, self.ingestion_config.train_data_path

        except Exception as e:
            raise CustomException(e)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    