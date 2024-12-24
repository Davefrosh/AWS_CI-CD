import sys
import pandas as pd
import numpy as np


from src.exception import Custom_exception

from src.utils import load_object

class Predict_pipeline:
    def __init__(self):
        pass
    
    def initiate_predict_pipeline(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocess_path = 'artifacts/preprocessor.pkl'
            scaler_path = 'artifacts/scaler.pkl'
            
            model = load_object(model_path)
            preprocessor = load_object(preprocess_path)
            scaler = load_object(scaler_path)
            preprocessed_data = preprocessor.transform(features)
            scaled_data = scaler.transform(preprocessed_data)
            prediction = model.predict(scaled_data)
            
            return prediction
        except Exception as e:
            raise Custom_exception(e,sys)    
    
class New_data:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_as_dataframe(self):
        try:
            data = {'gender':[self.gender],
                    'race/ethnicity':[self.race_ethnicity],
                    'parental level of education':[self.parental_level_of_education],
                    'lunch':[self.lunch],
                    'test preparation course':[self.test_preparation_course],
                    'reading score':[self.reading_score],
                    'writing score':[self.writing_score]}
            return pd.DataFrame(data)
        
        except Exception as e:
            raise Custom_exception(e,sys)
                                                             
        
    
    