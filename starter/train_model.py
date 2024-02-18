"""
Functions in this file are used to train, save and evaluate the machine learning model 
"""
# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import logging
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info('Read cleaned census data')
data_path = "../data/census_cleaned.csv"
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )

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

logger.info('Processing the training data')
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logger.info('Processing the test data')
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
logger.info('Training Random Forest Classifier')
model = train_model(X_train, y_train)

logger.info('Saving Random Forest Classifier model')
model_path = '../model/model.pkl'
joblib.dump(model, model_path)
            
logger.info('Saving Encoder')
encoder_path = '../model/encoder.pkl'
joblib.dump(encoder, encoder_path)   

logger.info('Saving Lb')
lb_path = '../model/lb.pkl'
joblib.dump(lb, lb_path) 

# Evaluate the model
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
logger.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")