
"""Verify Gemini and Ollama setup without exposing secrets."""
from __future__ import annotations
import os, time, requests
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent/".env")

def p(name,status,msg):
    print(f"[{status}] {name}: {msg}")

def main():
    key=os.getenv("GEMINI_API_KEY","").strip()
    model=os.getenv("GEMINI_MODEL","").strip()
    if not key:
        p("Gemini key","FAIL","Set GEMINI_API_KEY in .env.")
    else:
        p("Gemini key","PASS",f"key is present (ends ...{key[-4:]})")
    available=[]
    if key:
        try:
            from google import genai
            client=genai.Client(api_key=key)
            for m in client.models.list():
                mid=getattr(m,"name","") or ""
                actions=getattr(m,"supported_actions",[]) or []
                if "generateContent" in actions or not actions:
                    available.append(mid.replace("models/","",1))
            if available:
                p("Gemini models","PASS",", ".join(available))
            else: p("Gemini models","FAIL","No generateContent model IDs returned.")
        except Exception as exc:
            p("Gemini models","FAIL",f"{exc.__class__.__name__}: check key/network.")
    if model and available and model not in available:
        p("GEMINI_MODEL","FAIL",f"'{model}' is not available.")
        choices=[x for x in available if "pro" not in x.lower() and "2.5" not in x.lower()]
        suggestion=next((x for x in choices if "flash-lite" in x.lower()),next((x for x in choices if "flash" in x.lower()),choices[0] if choices else ""))
        print(f"Suggested model: {suggestion}")
    elif model: p("GEMINI_MODEL","PASS",model)
    else: p("GEMINI_MODEL","FAIL","Set GEMINI_MODEL in .env; use a listed non-Pro, non-2.5 model.")
    if key and model:
        try:
            from google import genai
            client=genai.Client(api_key=key); t=time.perf_counter()
            r=client.models.generate_content(model=model,contents="Say hello in 5 words")
            p("Gemini test","PASS",f"{(time.perf_counter()-t):.2f}s — {r.text.strip()[:100]}")
        except Exception as exc: p("Gemini test","FAIL",f"{exc.__class__.__name__}: check model/key.")
    host=os.getenv("OLLAMA_HOST","http://localhost:11434").rstrip("/")
    omodel=os.getenv("OLLAMA_MODEL","llama3.2:3b")
    try:
        r=requests.get(f"{host}/api/tags",timeout=3); r.raise_for_status()
        names=[m.get("name","") for m in r.json().get("models",[])]
        if omodel in names or any(n.split(":")[0]==omodel.split(":")[0] for n in names):
            p("Ollama","PASS",f"{omodel} is installed")
            try:
                t=time.perf_counter()
                rr=requests.post(f"{host}/api/chat",json={"model":omodel,"messages":[{"role":"user","content":"Say hello in 5 words"}],"stream":False,"options":{"num_ctx":8192,"temperature":0}},timeout=60)
                rr.raise_for_status(); p("Ollama test","PASS",f"{(time.perf_counter()-t):.2f}s")
            except Exception as exc: p("Ollama test","FAIL",f"{exc.__class__.__name__}: run ollama serve.")
        else: p("Ollama","FAIL",f"{omodel} is not installed. Run: ollama pull {omodel}")
    except Exception as exc: p("Ollama","FAIL",f"{exc.__class__.__name__}: start Ollama.")
if __name__=="__main__": main()
