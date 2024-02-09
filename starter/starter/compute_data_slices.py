"""
Functions in this file is used to computes performance on model slices. 
I.e. a function that computes the performance metrics 
when the value of a given feature is held fixed.
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import joblib
from ml.data import process_data
from ml.model import compute_model_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

cat_features = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
    ]

logger.info('Read cleaned census data')
census_data = pd.read_csv("../data/census_cleaned.csv")

logger.info('Split Train and Test data')
train, test = train_test_split(census_data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=census_data['salary']
                                )

logger.info('Retrieve Model')
model_path = '../model/model.pkl'
model = joblib.load(model_path)

logger.info('Retrieve Encoder')
encoder_path = '../model/encoder.pkl' 
encoder = joblib.load(encoder_path)

logger.info('Retrieve Lb')
lb_path = '../model/lb.pkl'
lb = joblib.load(lb_path)

slice_metrics = {'feature': [], 'category': [], 'precision': [], 'recall': [], 'fbeta': []}

logger.info('Compute slice metrics')
for cat in cat_features:
    for cls in test[cat].unique():
        df_temp = test[test[cat] == cls]
        X_test, y_test, _, _ = process_data(
            df_temp, categorical_features=cat_features, label='salary', training=False,
            encoder=encoder, lb=lb
        )

        y_pred = model.predict(X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        slice_metrics['feature'].append(cat)
        slice_metrics['category'].append(cls)
        slice_metrics['precision'].append(precision)
        slice_metrics['recall'].append(recall)
        slice_metrics['fbeta'].append(fbeta)
    
slice_df = pd.DataFrame.from_dict(slice_metrics)
slices_path = './slices_output.txt'
slice_df.to_csv(slices_path, index=False)
