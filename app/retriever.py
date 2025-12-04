# app/retriever.py
"""
Query orchestration: do retrieval with LlamaIndex then post-process with Ollama.
"""

from indexer import load_index, build_index, INDEX_PATH
from llm_client import generate_with_ollama

DEFAULT_MODEL = "openassistant/large"  # replace with a model you have in Ollama or on HF

def ensure_index():
    idx = load_index()
    if idx is None:
        idx = build_index()
    return idx

def answer_query(query: str, top_k: int = 3) -> str:
    idx = ensure_index()
    # LlamaIndex query API — basic similarity search to fetch top docs
    response = idx.query(query, similarity_top_k=top_k)
    # response may be an object — get the text snippet(s)
    docs_text = str(response)  # quick conversion; improve as needed
    # Compose a prompt for final generation
    prompt = f"Use the following retrieved documents to answer the question:\n\n{docs_text}\n\nQuestion: {query}\n\nAnswer concisely:"
    # Call Ollama to finalize/generate
    result = generate_with_ollama(DEFAULT_MODEL, prompt)
    return result
