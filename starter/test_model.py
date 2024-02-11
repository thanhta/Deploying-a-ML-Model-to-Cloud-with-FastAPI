import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, inference

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


def test_process_data():
    """ Test process data """

    df = pd.read_csv('./data/census_cleaned.csv')
    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape[0] == y_train.shape[0]


def test_ml_model():
    """ Test Random Forest model """

    model = joblib.load('./model/model.pkl')
    assert isinstance(model, RandomForestClassifier)


def test_ml_inference():
    """
    Test inference of model
    """
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape