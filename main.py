"""
Script for building FastAPI instance and model inference
"""
# Put the code for your API here.
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from typing import Union, Optional
from pydantic import BaseModel
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.get_pkl import get_pkl
import joblib
import uvicorn
import re



logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

root_dir = os.path.join(os.path.dirname(__file__), '')
logger.info(f"root_dir: {root_dir}")

model_path_pkl = 'model/model.pkl'
model_path = os.path.join(root_dir, model_path_pkl)
#model_path = 'model/model.pkl'
#logger.info(f"model_path: {model_path}")
model = joblib.load(model_path)
            

encoder_path_pkl = 'model/encoder.pkl'
encoder_path = os.path.join(root_dir, encoder_path_pkl)
#encoder_path = 'model/encoder.pkl'
#logger.info(f"encoder_path: {encoder_path}")
encoder = joblib.load(encoder_path)   


lb_path_pkl = 'model/lb.pkl'
lb_path = os.path.join(root_dir, lb_path_pkl)
#lb_path = 'model/lb.pkl'
#logger.info(f"lb_path: {lb_path}")
lb = joblib.load(lb_path)


#model, encoder, lb = get_pkl()
# Declare the data object
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    # Using the first row of census.csv as sample
    class Config:
        schema_extra = {
                        "example": {
                                    'age' :39,
                                    'workclass':"State-gov", 
                                    'fnlgt':77516,
                                    'education':"Bachelors",
                                    'education_num':13,
                                    'marital_status':"Never-married",
                                    'occupation':"Adm-clerical",
                                    'relationship':"Not-in-family",
                                    'race':"White",
                                    'sex':"Male",
                                    'capital_gain':2174,
                                    'capital_loss':0,
                                    'hours_per_week':40,
                                    'native_country':"United-States"        
                                    }
                        }


# Instantiate the FastAPI app.
app = FastAPI()

@app.get("/")
async def greetings():
    logger.info('Invoking http GET to get welcome message')
    return "Welcome to Census Data Classifier API"

# Use POST action to send data to the server
@app.post('/predict')
async def predict(predict: InputData):
    logger.info('Invoking http POST to do inference')
    sample_data = {  'age': predict.age,
                'workclass': predict.workclass, 
                'fnlgt': predict.fnlgt,
                'education': predict.education,
                'education-num': predict.education_num,
                'marital-status': predict.marital_status,
                'occupation': predict.occupation,
                'relationship': predict.relationship,
                'race': predict.race,
                'sex': predict.sex,
                'capital-gain': predict.capital_gain,
                'capital-loss': predict.capital_loss,
                'hours-per-week': predict.hours_per_week,
                'native-country': predict.native_country,
            }
    

    logger.info(f"sample_data: {sample_data}")

    # prepare the input data for inference as a dataframe
    sample_df = pd.DataFrame(sample_data, index=[0])

    # Convert the sample data has names with hyphens to underscore
    # sample_data = {key.replace('_', '-'): [value] for key, value in predict.__dict__.items()}
    # new_col_names = [re.sub("\\_", "-", col) for col in sample_df.columns]
    # sample_df.columns = new_col_names

    X, _, _, _ = process_data(
                    sample_df, categorical_features=cat_features, label=None, training=False,
                    encoder=encoder, lb=lb
                )
    
    y_pred = inference(model, X)

    # convert prediction to label and add to data output
    return {'Predicted Income': lb.inverse_transform(y_pred)[0]}

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=4400, reload=True)