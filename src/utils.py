import sys
import os

import pandas as pd
import numpy as np

import dill
from sklearn.model_selection import RandomizedSearchCV

from src.exception import Custom_exception
from sklearn.metrics import r2_score,accuracy_score

def save_object(file_path,obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise Custom_exception(e,sys)
    
    
    
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
    
        for i in range((len(list(models)))):
            model =list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            rs = RandomizedSearchCV(model,para,cv=3)
            rs.fit(x_train,y_train)
            
            model.set_params(**rs.best_params_)
        
            model.fit(x_train,y_train)
        
            y_train_pred = model.predict(x_train)
        
            y_test_pred = model.predict(x_test)
        
            train_model_score = r2_score(y_train,y_train_pred)
        
            test_model_score = r2_score(y_test,y_test_pred)
        
            report[list(models.keys())[i]] = test_model_score
        
        return report,rs.best_params_
    except Exception as e:
        raise Custom_exception(e,sys)
        
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            obj = dill.load(file_obj)
            
        return obj
    
    except Exception as e:
        raise Custom_exception(e,sys)
        
        