import sys
import os
import joblib
import pandas as pd

# Add src to sys.path to import custom modules
sys.path.insert(0, 'src')
from src.exception import CustomException
from src.logger import logging

# Define the local directory where artifacts are stored after download from S3
LOCAL_ARTIFACTS_DIR = "artifacts"

class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Loading model and preprocessor for prediction...")
            
            # Define paths to the downloaded artifacts in the temporary directory
            model_path = os.path.join(LOCAL_ARTIFACTS_DIR, "model.pkl")
            preprocessor_path = os.path.join(LOCAL_ARTIFACTS_DIR, "preprocessor.pkl")

            self.model = joblib.load(model_path)
            preprocessor_obj = joblib.load(preprocessor_path)
            
            self.mlb_transformer = preprocessor_obj['mlb_transformer']
            self.disease_mapping = preprocessor_obj['disease_mapping']
            logging.info("Model and preprocessor loaded successfully from local temp storage.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, user_symptoms: list):
        try:
            symptoms_to_transform = [user_symptoms]
            input_vector = self.mlb_transformer.transform(symptoms_to_transform)
            prediction_code = self.model.predict(input_vector)[0]
            disease_name = self.disease_mapping.get(prediction_code, "Unknown condition")
            
            return disease_name
        except Exception as e:
            raise CustomException(e, sys)
