import sys
import os

# Add src to system path to allow component imports
sys.path.insert(0, 'src')
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def run(self):
        logging.info("--- Training Pipeline Started ---")
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("--- Training Pipeline Completed Successfully! ---")
            logging.info(f"Model saved locally at: {model_path}")
            logging.info(f"Preprocessor saved locally at: {preprocessor_path}")

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
