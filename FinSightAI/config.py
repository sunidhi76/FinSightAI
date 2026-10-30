
"""Application configuration for FinSight."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"
BM25_FILE = INDEX_DIR / "bm25.json"
INDEX_INFO_FILE = INDEX_DIR / "index_info.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_K = 5
RRF_K = 60

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
# Streamlit Cloud secrets are supported without ever printing the secret.
try:
    import streamlit as _st
    if hasattr(_st, "secrets"):
        GEMINI_API_KEY = str(_st.secrets.get("GEMINI_API_KEY", GEMINI_API_KEY)).strip()
        GEMINI_MODEL = str(_st.secrets.get("GEMINI_MODEL", GEMINI_MODEL)).strip()
except Exception:
    pass
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.12"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "604800"))

for directory in (RAW_DIR, INDEX_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
