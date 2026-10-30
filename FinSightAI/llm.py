
"""Gemini-first LLM client with Ollama fallback."""
from __future__ import annotations
import json
import logging
import time
from typing import Any
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER, OLLAMA_HOST, OLLAMA_MODEL

LOGGER=logging.getLogger("finsight.llm")
OLLAMA_AVAILABLE=False

def check_ollama() -> tuple[bool, str]:
    global OLLAMA_AVAILABLE
    try:
        r=requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        r.raise_for_status()
        names={m.get("name","") for m in r.json().get("models",[])}
        OLLAMA_AVAILABLE=OLLAMA_MODEL in names or any(n.split(":")[0]==OLLAMA_MODEL.split(":")[0] for n in names)
        return OLLAMA_AVAILABLE, ("model available" if OLLAMA_AVAILABLE else f"{OLLAMA_MODEL} not installed")
    except requests.RequestException as exc:
        OLLAMA_AVAILABLE=False
        return False, f"Ollama unavailable: {exc.__class__.__name__}"

def _gemini_generate(prompt: str, system: str, temperature: float, json_mode: bool) -> str:
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        raise RuntimeError("Gemini API key or model is not configured.")
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=GEMINI_API_KEY)
    config=types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    response=client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
    return response.text or ""

def _ollama_generate(prompt: str, system: str, temperature: float, json_mode: bool) -> str:
    global OLLAMA_AVAILABLE
    ok, reason=check_ollama()
    if not ok:
        raise RuntimeError(reason)
    payload={
        "model": OLLAMA_MODEL,
        "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
        "stream":False,
        "options":{"num_ctx":8192,"temperature":0},
    }
    if json_mode: payload["format"]="json"
    r=requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("message",{}).get("content","")

def generate(prompt: str, system: str, temperature: float=0, json_mode: bool=False) -> tuple[str,str]:
    """Generate with Gemini first; fallback to Ollama after transient/quota failures."""
    if LLM_PROVIDER=="ollama":
        return _ollama_generate(prompt, system, temperature, json_mode), "Ollama"
    last=None
    for attempt in range(3):
        try:
            return _gemini_generate(prompt, system, temperature, json_mode), "Gemini"
        except Exception as exc:
            last=exc
            msg=str(exc).lower()
            transient=any(x in msg for x in ("429","quota","timeout","timed out","connection","network","503","502","unavailable"))
            if not transient: break
            if attempt<2: time.sleep(2**attempt)
    LOGGER.warning("Gemini failed; attempting Ollama fallback (%s)", last.__class__.__name__ if last else "unknown")
    try:
        return _ollama_generate(prompt, system, temperature, json_mode), "Ollama fallback"
    except Exception:
        raise RuntimeError("Gemini failed and Ollama fallback is unavailable.") from last
