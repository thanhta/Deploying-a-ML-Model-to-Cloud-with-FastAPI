"""
Unit test of main.py API module
"""

from fastapi.testclient import TestClient
import json
import logging
#import pytest
#import uvicorn
from main import app

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


client = TestClient(app)

def test_get():
    """
    Test welcome message for http get at root
    """
    logger.info('Invoking test_get')

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to Census Data Classifier API"


def test_predict_invalid_content():
    """
    Test the prediction of invalid conntent 
    """
    logger.info('Invoking test_predict_invalid_content')

    data = {"age": 40,
            "workclass": "Private",
            "fnlgt": 154374,
            "education": "",
            "marital_status": "Married-civ-spouse",
            "occupation": "Machine-op-inspct",
            "race": "White",
            "sex": "Male",
            "capitalgain": 0,
            "capitalloss": 0,
            "hoursperweek": 40,
            "nativecountry": "United-States"
            }
    
    response = client.post("/predict", json=json.dumps(data))
    assert response.status_code == 422
    #assert response.status_code == 307

def test_predict_over_50K():
    """
    Test the prediction of salary over 50K 
    """
    logger.info('Invoking test_predict_over_50K')

    data = {"age": 40,
            "workclass": "Private",
            "fnlgt": 154374,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Machine-op-inspct",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    
    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {"Predicted Income": ">50K"}

def test_predict_below_50K():
    """
    Test the prediction of salary less than 50K 
    """
    logger.info('Invoking Test_predict_below_50K')

    data = {"age": 28,      
            "workclass": "Private",
            "fnlgt": 338409,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Wife",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "Cuba"
            }
    
    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {"Predicted Income": "<=50K"}