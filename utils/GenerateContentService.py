import json
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted
from config import Config
from ..utils import GeminiResponseEditor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def generate_content(prompt, system_instruction=" ", json_response=False, generation_config={} ):
    GOOGLE_API_KEY = Config.GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

    GENERATION_CONFIG = {
        "temperature": 1,
        #"top_p": 0.95,
        #"top_k": 0,
    }
    if json_response:
        GENERATION_CONFIG["response_mime_type"] = "application/json"

    combined_generation_config = GENERATION_CONFIG | generation_config
    SAFETY_SETTINGS = [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    model = genai.GenerativeModel(
        "models/gemini-2.0-flash-latest",
        system_instruction=system_instruction,
        generation_config=combined_generation_config,
        safety_settings=SAFETY_SETTINGS,
    )

    max_attempts = 3  # Adjust this number as needed
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            #print(response)
            formatted_text = GeminiResponseEditor.remove_special_chars(response.text)
            return formatted_text  # If successful, return the result
        except DeadlineExceeded:
            logger.info("Deadline exceeded. Retrying in 1 second...")
            time.sleep(1)
        except json.JSONDecodeError:
            logger.info("JSON decode error. Retrying in 1 second...")
            time.sleep(1)
        except ResourceExhausted:
            logger.info("Resource exhausted. Retrying in 10 seconds...")
            time.sleep(10)
        except ValueError:
            logger.info("A value error occurred. Retrying in 1 second...")
            if response is not None:
                logger.info(response.prompt_feedback)
                logger.info(response.candidates[0].finish_reason)
                logger.info(response.candidates[0].safety_ratings)
            time.sleep(1)
        except Exception as e:
            logger.error(e)

    # If all retries fail
    logger.info(f"Failed to generate content after {max_attempts} attempts.")
    return None
