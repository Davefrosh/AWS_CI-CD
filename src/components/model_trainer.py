import os
import sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import Custom_exception

from src.utils import save_object

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')
    
logging.info('mdoel fitting about to start')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 
        
    def initate_model_trainer(self,train_arr,test_arr):
        try:
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],
                                             train_arr[:,-1],
                                             test_arr[:,:-1],
                                             test_arr[:,-1])
            
            models = {
                'rfc':RandomForestRegressor(),
                'lr':LinearRegression(),
                'kn': KNeighborsRegressor(),
                'dtree':DecisionTreeRegressor()
            }
            
            params = {
    'rfc': {
        'n_estimators': [100, 200, 300],  
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],   
        'bootstrap': [True, False]        
    },
    'lr': {
        'fit_intercept': [True, False]      
    },
    'kn': {
        'n_neighbors': [3, 5, 7, 10],   
        'weights': ['uniform', 'distance'],  
        'metric': ['euclidean', 'manhattan', 'minkowski'] 
    },
    'dtree': {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],    
        'max_features': [None, 'sqrt', 'log2']
}
            }
            
            
            
            
            model_report,best_params = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
            
            logging.info(f"my best parameters is/are {best_params}")
            
            best_model_score =max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            preds = best_model.predict(x_test)
            r_score = r2_score(preds,y_test)
            logging.info('results ready')
            logging.info(f'Model prediction done with best model: {best_model_name}')
            logging.info(f'R^2 Score: {r_score}')
            
            logging.info('model prediction done with best model')
            
            save_object(self.model_trainer_config.trained_model_file_path,best_model)
            
            return r_score
            
        except Exception as e:
            raise Custom_exception(e,sys)
            