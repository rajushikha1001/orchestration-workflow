# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import answer_query, ensure_index
import uvicorn

app = FastAPI(title="RAG Orchestration Example")

class Query(BaseModel):
    q: str

@app.on_event("startup")
def startup_event():
    # Ensure index is ready at startup (build if missing)
    ensure_index()

@app.post("/query")
def query_endpoint(payload: Query):
    try:
        answer = answer_query(payload.q)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
