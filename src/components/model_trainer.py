import os
import sys
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

# Add src to sys.path to import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Model Training started")
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            model.fit(X_train, y_train)
            logging.info("Model training complete.")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")

            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            logging.info("Model saved as model.pkl")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)