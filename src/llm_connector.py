import json
import requests

from src.config import API_URL, ORACLE_MODEL
from src.utils import debug_print

# ============================================================
# LLM CONNECTOR
# ============================================================
def query_rwth_server(prompt, max_tokens=50, model_to_use=ORACLE_MODEL):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }

    session = requests.Session()
    session.trust_env = False

    try:
        response = session.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        debug_print("API_ERROR", f"Connection failed: {e}")
        return ""