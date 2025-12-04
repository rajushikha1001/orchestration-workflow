# tests/test_indexer.py
from app.indexer import build_index, load_index, INDEX_PATH
import os

def test_build_and_load_index(tmp_path):
    # Build index (it will write to the repo-level index.json path)
    idx = build_index()
    assert idx is not None
    loaded = load_index()
    assert loaded is not None
