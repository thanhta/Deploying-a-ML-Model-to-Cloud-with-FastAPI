import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics

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

def test_model():
    """ Test Random Forest model """

    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """
    Test inference of model
    """
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape

def test_compute_model_metrics():
    """
    Test compute_model_metrics
    """
    y = [1, 1, 1, 0]
    y_preds = [1, 0, 1, 0]
    out_put = compute_model_metrics(y, y_preds)
    assert len(out_put) == 3
    for i in out_put:
        assert type(i)==np.float64