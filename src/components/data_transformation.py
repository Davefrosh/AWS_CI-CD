import pandas as pd
import numpy as np
import os
import sys
import feature_engine

from feature_engine.encoding import OrdinalEncoder,OneHotEncoder
from feature_engine.imputation import MeanMedianImputer,RandomSampleImputer,CategoricalImputer
from dataclasses import dataclass
from src.exception import Custom_exception
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import save_object

@dataclass
class Data_transformation_config:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')
    
@dataclass
class Scaler_config:
    scaler_file_path = os.path.join('artifacts','scaler.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = Data_transformation_config()
        self.scaler_config = Scaler_config()
    def get_data_transformer_object(self):
        try:
            num_cols = ['reading score','writing score']
            cat_cols_1= ['gender','test preparation course']
            cat_cols_2 = ['race/ethnicity', 'parental level of education', 'lunch']
            
            fe1 = MeanMedianImputer(variables=num_cols,imputation_method='median')
            fe2 = CategoricalImputer(imputation_method='frequent',variables=cat_cols_1)
            fe3 = CategoricalImputer(variables=cat_cols_2)
            fe4 = OneHotEncoder(variables=cat_cols_1,drop_last=True)
            fe5 = OrdinalEncoder(variables=cat_cols_2)
            
            pipe = Pipeline([
                ('fe1',fe1),
                ('fe2',fe2),
                ('fe3',fe3),
                ('fe4',fe4),
                ('fe5',fe5)
            ]) 
            return pipe
            
        except Exception as e:
            raise Custom_exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
                
            logging.info('initiated data transformation')
                
            pipe_obj=self.get_data_transformer_object()
            
            target = 'math score'
            
            input_feature_train_df = train_df.drop(columns=[target],axis=1)
            target_tr_df = train_df[target]
            
            input_feature_test_df = test_df.drop(columns=[target],axis=1)
            target_te_df = test_df[target]
            
            
            tr_proc=pipe_obj.fit_transform(input_feature_train_df,target_tr_df)
            te_proc = pipe_obj.transform(input_feature_test_df)
            
            sc=StandardScaler()
            
            tr_process= sc.fit_transform(tr_proc)
            te_process = sc.transform(te_proc)
            
            train_arr = np.c_[tr_process,np.array(target_tr_df)]
            test_arr = np.c_[te_process,np.array(target_te_df)]
            
            save_object(self.data_transformation_config.preprocessor_file_path,pipe_obj)
            save_object(self.scaler_config.scaler_file_path,sc)
            
            
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_file_path)
            
        except Exception as e:
            raise Custom_exception(e,sys)