import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","models","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Time: str,
        Amount: str,
        V1_V28: str):

        self.Time = Time
        self.Amount = Amount
        self.V1_V28 = V1_V28

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Time": [self.Time],
                "Amount": [self.Amount],
                "V1_V28": [self.V1_V28],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
