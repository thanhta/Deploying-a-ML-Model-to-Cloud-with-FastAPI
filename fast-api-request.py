import json
import requests

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

"""""
response = requests.post(
    "https://fastapi-ml-predict-03185853b66f.herokuapp.com/predict", data=json.dumps(data) )
"""

response = requests.post(
    "https://fastapi-ml-predict-03185853b66f.herokuapp.com/predict", json=data )

#print(response.status_code)
#print(response.json())

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())