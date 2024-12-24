import os
import sys
import pandas as pd
import src

from src.exception import Custom_exception
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import Data_transformation_config

from src.components.model_trainer import ModelTrainer,ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train_csv')
    test_data_path:str = os.path.join('artifacts','test_csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info('data ingestion has begun')
        
        try:
            df=pd.read_csv('notebooks\stud.csv')
            logging.info('data read as a data frame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('train / test initiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('ingestion of the data is complete')
            
            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
            
            
        except Exception as e:
            raise Custom_exception(e,sys)
    
    
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()  
    logging.info('data ingestion set')
    data_transformation= DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) 
    logging.info('data tranformation set')  
    model_trainer= ModelTrainer()
    model_trainer.initate_model_trainer(train_arr,test_arr)    
    logging.info('model training set')
            