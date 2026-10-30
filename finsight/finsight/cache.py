
"""Disk cache used to protect free-tier LLM quotas."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
from config import CACHE_DIR, CACHE_TTL
from utils import text_hash

def _path(key: str) -> Path:
    return CACHE_DIR / f"{text_hash(key)}.json"

def get(key: str) -> Any | None:
    path=_path(key)
    if not path.exists(): return None
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
        if CACHE_TTL > 0 and time.time()-data["created"] > CACHE_TTL:
            path.unlink(missing_ok=True)
            return None
        return data["value"]
    except (OSError, ValueError, KeyError):
        return None

def set(key: str, value: Any) -> None:
    path=_path(key)
    path.write_text(json.dumps({"created":time.time(),"value":value}, ensure_ascii=False), encoding="utf-8")


def make_key(query: str, retrieval_settings: dict, index_id: str) -> str:
    normalized_query = " ".join(query.casefold().split())

    return json.dumps(
        {
            "q": normalized_query,
            "settings": retrieval_settings,
            "index": index_id,
        },
        sort_keys=True,
    )
