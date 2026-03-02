# llm_client.py
# ---------------------------------------------------------------------------
# Handles communication with the LLM endpoint.
# 1. Builds the prompt using the template from prompt.py.
# 2. Sends the request and validates the JSON response against schema.py.
# ---------------------------------------------------------------------------


import os
import json
import requests
from dotenv import load_dotenv
from prompt import EXTRACTION_PROMPT
from schema import ProjectExtraction


def extract_project_metadata(text: str) -> dict:
    """
    Send raw project-requirement text to the LLM and return validated
    structured metadata as a plain dict.

    Returns a dict with an 'error' key when extraction or validation fails,
    so the caller can display a user-friendly message.
    """

    # --- 1. Build the prompt ------------------------------------------------
    # .format() replaces {text} with the user input; literal braces in the
    # template are already doubled {{ }} to avoid KeyError.

    prompt = EXTRACTION_PROMPT.format(text=text)
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-5.1")
    max_tokens = int(os.getenv("OPENAI_LLM_MAX_TOKENS", "4096"))

    if not openai_api_key:
        return {
            "error": "Invalid OpenAI API Key",
            "details": "Set OPENAI_API_KEY in .env",
            "raw_output": ""
        }

    endpoint_url = f"{openai_api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": openai_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    response = requests.post(endpoint_url, json=data, headers=headers, timeout=120)


    # --- 4. Handle HTTP-level errors ----------------------------------------
    if response.status_code == 200:
        raw = response.json()["choices"][0]["message"]["content"].strip()
        # Remove markdown code blocks if the model accidentally includes them
        raw = raw.replace("```json", "").replace("```", "").strip()
    else:
        raise Exception(f"Error Generating Answer (HTTP {response.status_code}): {response.text}")

    # --- 5. Parse & validate the LLM output against the Pydantic schema -----
    try:
        parsed = json.loads(raw)                 # raw JSON string → dict
        validated = ProjectExtraction(**parsed)   # validate with Pydantic
        # Use .model_dump() for Pydantic v2 compatibility
        return validated.model_dump()                   
    except json.JSONDecodeError as e:
        # LLM returned something that is not valid JSON
        return {
            "error": "JSON parsing failed",
            "details": f"JSONDecodeError: {e}",
            "raw_output": raw
        }
    except Exception as e:
        # Pydantic validation or any other unexpected error
        return {
            "error": "Extraction failed",
            "details": str(e),
            "raw_output": raw
        }
