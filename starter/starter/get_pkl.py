"""
Script for FastAPI instance and model inference
"""
# Put the code for your API here.
import os
import sys
import logging
import pandas as pd
import joblib



def get_pkl():
    """
    get model.pkl, encoder.pkl, lb.pkl
    """ 
    
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    root_dir = os.path.join(os.path.dirname(__file__), '')
    logger.info(f"root_dir: {root_dir}")

    logger.info('Retrieve Model pkl')
    model_path_pkl = 'model/model.pkl'
    #model_path = 'model/model.pkl'
    #model_path = os.path.join(file_dir, model_path_pkl)
    #model_path = os.path.join(root_dir, model_path_pkl)
    #logger.info(f"model_path: {model_path}")

    model_path = './model/model.pkl'
    model = joblib.load(model_path)
            
    logger.info('Retrieve Encoder pkl')
    #encoder_path_pkl = 'model/encoder.pkl'
    #encoder_path = 'model/encoder.pkl'
    #encoder_path = os.path.join(file_dir, encoder_path_pkl)
    #encoder_path = os.path.join(root_dir, encoder_path_pkl)

    encoder_path = './model/encoder.pkl'
    encoder = joblib.load(encoder_path)   

    logger.info('Retrieve LabelBinarizer pkl')
    #lb_path_pkl = 'model/lb.pkl'
    #lb_path = 'model/lb.pkl'
    #lb_path = os.path.join(file_dir, lb_path)
    #lb_path = os.path.join(root_dir, lb_path_pkl)

    lb_path = './model/lb.pkl'
    lb = joblib.load(lb_path) 

    return model, encoder, lb