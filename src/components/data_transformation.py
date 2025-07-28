import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Add src to sys.path to import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation started")
        try:
            # Fill NaN values with an empty string to avoid errors during processing
            train_df = pd.read_csv(train_path).fillna('')
            test_df = pd.read_csv(test_path).fillna('')

            target_column_name = 'disease'
            
            # Identify all columns that contain symptom information
            symptom_cols = [col for col in train_df.columns if 'symptom' in col]

            # Helper function to clean and collect all symptoms from a row into a list
            def get_symptoms_from_row(row):
                # We strip whitespace and replace underscores from each symptom string
                return [s.strip().replace('_', ' ') for s in row if s]
            
            train_symptoms_list = train_df[symptom_cols].apply(get_symptoms_from_row, axis=1)
            test_symptoms_list = test_df[symptom_cols].apply(get_symptoms_from_row, axis=1)

            # Use MultiLabelBinarizer to perform one-hot encoding on the lists of symptoms
            mlb = MultiLabelBinarizer()
            
            # Fit the binarizer on the training data and transform both sets
            input_feature_train_arr = mlb.fit_transform(train_symptoms_list)
            input_feature_test_arr = mlb.transform(test_symptoms_list)
            logging.info("Successfully one-hot encoded symptom data.")

            # Process the target (disease) column
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]
            
            # Create a mapping from numerical code back to disease name for later use
            target_categories = target_feature_train_df.astype('category').cat
            disease_mapping = {i: label for i, label in enumerate(target_categories.categories)}
            target_feature_train_arr = target_categories.codes
            
            # Use the same categories for the test set to ensure consistency
            target_feature_test_arr = pd.Categorical(target_feature_test_df, categories=target_categories.categories).codes
            
            # Save the fitted binarizer and mappings together as our preprocessor object
            logging.info("Creating and saving preprocessor object.")
            preprocessor_obj = {
                'mlb_transformer': mlb,  # The fitted one-hot encoder
                'disease_mapping': disease_mapping
            }
            joblib.dump(preprocessor_obj, self.data_transformation_config.preprocessor_obj_file_path)

            # Combine the processed features and target into final numpy arrays for the model
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Data Transformation completed")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
