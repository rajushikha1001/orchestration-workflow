# app/indexer.py
"""
Creates a LlamaIndex index from text files in ./data/sample_docs
Saves/loads the index to disk (./index.json).
"""

import os
from pathlib import Path
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding

INDEX_PATH = Path(__file__).resolve().parents[1] / "index.json"
DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"

def build_index(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # Read documents
    reader = SimpleDirectoryReader(str(DOCS_DIR))
    docs = reader.load_data()

    # Create embeddings via HuggingFace integration
    hf_embed = HuggingFaceEmbedding(model_name=model_name)

    # Create index (vector) using the embedding and default LLM predictor
    # Note: GPTSimpleVectorIndex is an older API name; main idea is: build and persist
    index = GPTSimpleVectorIndex.from_documents(docs, embed_model=hf_embed)
    index.save_to_disk(str(INDEX_PATH))
    return index

def load_index():
    if INDEX_PATH.exists():
        return GPTSimpleVectorIndex.load_from_disk(str(INDEX_PATH))
    return None
