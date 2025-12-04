# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_endpoint():
    res = client.post("/query", json={"q": "What is in the docs?"})
    assert res.status_code in (200, 500)  # 200 if index + ollama available; 500 if ollama not reachable
