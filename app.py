import os
import sys
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import joblib
import numpy as np

# Add src to system path to allow imports
sys.path.insert(0, 'src')
from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils import get_gemini_response
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# --- Load Data Once on Startup from Local Storage ---
try:
    logging.info("Flask app starting up...")
    # Load artifacts directly from the local 'artifacts' folder
    preprocessor = joblib.load('artifacts/preprocessor.pkl')
    
    VALID_SYMPTOMS_DISPLAY = list(preprocessor['mlb_transformer'].classes_)
    
    # Load all dataframes from the root project folder
    description_df = pd.read_csv('symptom_Description.csv')
    description_df.columns = [col.strip().lower() for col in description_df.columns]
    description_df['disease'] = description_df['disease'].str.strip().str.lower()

    precaution_df = pd.read_csv('symptom_precaution.csv')
    precaution_df.columns = [col.strip().lower() for col in precaution_df.columns]
    precaution_df['disease'] = precaution_df['disease'].str.strip().str.lower()

    severity_df = pd.read_csv('Symptom-severity.csv')
    severity_df.columns = [col.strip().lower() for col in severity_df.columns]
    severity_dict = severity_df.set_index('symptom')['weight'].to_dict()

    dataset_df = pd.read_csv('dataset.csv').fillna('')
    dataset_df.columns = [col.strip().lower() for col in dataset_df.columns]
    
    prediction_pipeline = PredictionPipeline()
    logging.info("All resources loaded successfully from local storage.")
except FileNotFoundError as e:
    error_msg = f"ERROR: Could not find a required file: {e}. Please run 'python src/pipeline/train_pipeline.py' first."
    logging.error(error_msg)
    print(error_msg)
    sys.exit(1)
# ------------------------------------

@app.route('/')
def home():
    session.clear()
    session['symptoms'] = []
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_message = data['message'].lower()
        logging.info(f"Received message: {user_message}")

        current_symptoms = session.get('symptoms', [])
        
        potential_symptoms = [s.strip() for s in user_message.split(',')]
        
        added_symptoms = []
        unrecognized_symptoms = []

        for symptom in potential_symptoms:
            if symptom in VALID_SYMPTOMS_DISPLAY:
                if symptom not in current_symptoms:
                    current_symptoms.append(symptom)
                    added_symptoms.append(symptom)
            elif symptom:
                unrecognized_symptoms.append(symptom)
        
        session['symptoms'] = current_symptoms
        
        bot_response = ""
        suggested_symptoms = []

        if added_symptoms:
            logging.info(f"Added symptoms: {added_symptoms}. Current list: {current_symptoms}")
            predicted_disease = prediction_pipeline.predict(current_symptoms)
            logging.info(f"ML Model Prediction: {predicted_disease}")
            
            symptom_weights = [severity_dict.get(s.replace(' ', '_'), 0) for s in current_symptoms]
            avg_severity = np.mean(symptom_weights) if symptom_weights else 0
            severity_level = "Low"
            if avg_severity > 4:
                severity_level = "High"
            elif avg_severity > 2:
                severity_level = "Moderate"
            logging.info(f"Calculated severity: {avg_severity:.2f} ({severity_level})")

            try:
                disease_symptoms = set()
                disease_rows = dataset_df[dataset_df['disease'] == predicted_disease]
                symptom_cols = [f'symptom_{i}' for i in range(1, 18)]
                for _, row in disease_rows.iterrows():
                    for col in symptom_cols:
                        symptom = row[col].strip().replace('_', ' ')
                        if symptom and symptom not in current_symptoms:
                            disease_symptoms.add(symptom)
                suggested_symptoms = list(disease_symptoms)[:3]
                logging.info(f"Suggesting new symptoms: {suggested_symptoms}")
            except Exception as e:
                logging.warning(f"Could not generate symptom suggestions: {e}")

            desc = "No description available."
            precaution_list = []
            try:
                lookup_disease = predicted_disease.strip().lower()
                desc = description_df.loc[description_df['disease'] == lookup_disease, 'description'].iloc[0]
                precs = precaution_df.loc[precaution_df['disease'] == lookup_disease]
                precaution_list = [row[f'precaution_{i+1}'] for _, row in precs.iterrows() for i in range(4) if pd.notna(row[f'precaution_{i+1}'])]
            except (IndexError, KeyError) as e:
                logging.warning(f"Could not find description/precaution for {predicted_disease}: {e}")

            prompt = f"""
            Analyze the following medical information and provide a concise summary in HTML format.

            **Input Symptoms:** {", ".join(current_symptoms)}
            **Predicted Condition:** {predicted_disease}
            **Calculated Severity:** {severity_level}
            **Full Description:** {desc}
            **Precautions:** {', '.join(precaution_list)}

            **Your Task:**
            Generate an HTML response. Do NOT wrap it in markdown.
            1.  Start with "**Predicted Condition:**" followed by the disease name.
            2.  Add a "**Severity Level:**" section. If severity is "High", add a strong recommendation to see a doctor.
            3.  Create a "**Description:**" section with 2-3 short bullet points.
            4.  Create a "**Suggested Precautions:**" section with bullet points for each precaution.
            5.  Use `<ul>` and `<li>` tags for bullet points and `<br>` for spacing.
            """
            bot_response = get_gemini_response(prompt)
        
        if unrecognized_symptoms:
            unrec_str = ", ".join([f"'{s}'" for s in unrecognized_symptoms])
            error_message = f"I don't recognize the following symptom(s): {unrec_str}. Please check for typos."
            bot_response = (bot_response + "<br><br>" + error_message) if added_symptoms else error_message
        
        elif not added_symptoms and not unrecognized_symptoms:
             bot_response = "Please enter a symptom."

        return jsonify({'answer': bot_response, 'symptoms': session.get('symptoms', []), 'suggestions': suggested_symptoms})

    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'answer': 'Sorry, an internal error occurred.', 'symptoms': [], 'suggestions': []}), 500


@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    session['symptoms'] = []
    logging.info("Session cleared.")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    try:
        if not os.path.exists('artifacts/model.pkl'):
            error_msg = "ERROR: Model artifact not found. Please run 'python src/pipeline/train_pipeline.py' first."
            logging.error(error_msg)
            print(error_msg)
        else:
            app.run(host="0.0.0.0", port=5010)
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED DURING APPLICATION STARTUP ---\n{e}")
        logging.error(f"Application startup failed: {e}")
