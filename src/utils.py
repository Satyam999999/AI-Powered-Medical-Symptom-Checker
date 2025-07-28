import os
import sys
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Add src to sys.path to import custom modules
sys.path.insert(0, 'src')
from src.exception import CustomException
from src.logger import logging

# Load environment variables from a .env file
load_dotenv()

# Configure the Gemini API with your key
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.warning("GEMINI_API_KEY not found. The app will run without generative features.")
    else:
        genai.configure(api_key=gemini_api_key)
except Exception as e:
    logging.error(f"Error configuring Gemini: {e}")

def get_gemini_response(prompt: str) -> str:
    """
    Sends a prompt to the Gemini model and returns the text response,
    cleaning it of any markdown code fences.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini API key not configured. Cannot provide detailed explanation."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # --- FIX: Clean the response to remove markdown code blocks ---
        # This regex removes ```html ... ``` and ``` ... ``` wrappers.
        cleaned_text = re.sub(r'```(html)?\n(.*)\n```', r'\2', response.text, flags=re.DOTALL)
        
        return cleaned_text.strip()

    except Exception as e:
        raise CustomException(e, sys)
