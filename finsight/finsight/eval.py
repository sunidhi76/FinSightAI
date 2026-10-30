
"""Evaluate retrieval and grounded answers on six questions."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR
from retriever import Retriever
from generator import answer_question,REFUSAL
from llm import LLM_PROVIDER
import cache
import llm

def keyword_hit(answer,gold):
    if not gold: return True
    low=answer.lower()
    return any(k.lower() in low for k in gold)

def run(provider=None):
    if provider:
        llm.LLM_PROVIDER = provider
    questions=json.loads(Path("test_questions.json").read_text(encoding="utf-8"))
    model=SentenceTransformer("all-MiniLM-L6-v2")
    retr=Retriever(model=model)
    rows=[]
    for q in questions:
        settings={"mode":q.get("mode","Hybrid"),"k":5,"company":q.get("company")}
        results=retr.retrieve(q["question"],5,settings["mode"],q.get("company"),q.get("year"))
        hit=any(r["page"] in q.get("expected_pages",[]) for r in results)
        key=cache.make_key(q["question"],settings,retr.index_id)
        cached=cache.get(key)
        t=time.perf_counter()
        if cached: data,used=cached["data"],cached["provider"]+" (cache)"
        else:
            data,used=answer_question(q["question"],results)
            cache.set(key,{"data":data,"provider":used})
        latency=time.perf_counter()-t
        rows.append({"id":q["id"],"hit@5":hit,"correct":keyword_hit(data["answer"],q.get("gold_keywords",[])),
                     "citation":bool(data["citations"]),"refusal":(data["answer"]==REFUSAL)==q.get("unanswerable",False),
                     "latency_s":round(latency,3),"provider":used})
        time.sleep(1)
    md=Path("EVALUATION.md")
    old=md.read_text(encoding="utf-8") if md.exists() else "# Evaluation\n"
    table=["\n## Latest run", "|ID|Hit@5|Correct|Citation|Refusal|Latency|Provider|","|---|---:|---:|---:|---:|---:|---|"]
    table += [f"|{r['id']}|{'✓' if r['hit@5'] else '✗'}|{'✓' if r['correct'] else '✗'}|{'✓' if r['citation'] else '✗'}|{'✓' if r['refusal'] else '✗'}|{r['latency_s']}|{r['provider']}|" for r in rows]
    md.write_text(old+"\n".join(table)+"\n",encoding="utf-8")
    print("\n".join(map(str,rows)))
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--provider",choices=["gemini","ollama"],default=None)
    args=ap.parse_args()
    run(args.provider)
