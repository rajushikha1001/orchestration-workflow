# app/llm_client.py
"""
Small wrapper to call Ollama's local REST API.
Assumes an Ollama server running at http://ollama:11434 (when dockerized),
or http://localhost:11434 when running locally.
"""

import os
import requests
from typing import Dict, Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # containerized will use service name

def generate_with_ollama(model: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Call Ollama to generate text from a prompt.
    Returns the text result (string).
    """
    endpoint = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "parameters": {"temperature": 0.0}
    }
    resp = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns 'text' chunks (this may vary by version)
    return data.get("text", "")
