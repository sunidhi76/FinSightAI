
"""Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion."""
from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from config import INDEX_DIR, DEFAULT_K, RRF_K

class Retriever:
    def __init__(self,index_dir:Path=INDEX_DIR,model=None):
        self.index_dir=index_dir
        self.model=model
        self.index=faiss.read_index(str(index_dir/"faiss.index"))
        self.metadata=json.loads((index_dir/"metadata.json").read_text(encoding="utf-8"))
        corpus=json.loads((index_dir/"bm25.json").read_text(encoding="utf-8"))
        self.bm25=BM25Okapi(corpus)
        self.index_id=json.loads((index_dir/"index_info.json").read_text())["index_id"]

    def _allowed(self, company=None, year=None):
        return {i for i,m in enumerate(self.metadata)
                if (not company or m["company"]==company) and (not year or str(m["year"])==str(year))}

    def retrieve(self,query,k=DEFAULT_K,mode="Hybrid",company=None,year=None):
        allowed=self._allowed(company,year)
        if not allowed: return []
        candidates={}
        if mode in ("Dense only","Hybrid"):
            q=self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
            scores,idxs=self.index.search(q,min(len(self.metadata),max(k*4,20)))
            rank=0
            for score,idx in zip(scores[0],idxs[0]):
                if idx<0 or idx not in allowed: continue
                rank+=1; candidates.setdefault(idx,{})["dense_rank"]=rank; candidates[idx]["dense_score"]=float(score)
        if mode in ("BM25 only","Hybrid"):
            scores=self.bm25.get_scores(query.lower().split())
            order=np.argsort(scores)[::-1]
            rank=0
            for idx in order:
                if idx not in allowed: continue
                rank+=1; candidates.setdefault(int(idx),{})["bm25_rank"]=rank; candidates[int(idx)]["bm25_score"]=float(scores[idx])
                if rank>=max(k*4,20): break
        fused=[]
        for idx,d in candidates.items():
            rrf=sum(1/(RRF_K+d[key]) for key in ("dense_rank","bm25_rank") if key in d)
            fused.append((rrf,idx,d))
        fused.sort(reverse=True)
        results=[]
        for score,idx,d in fused[:k]:
            m=dict(self.metadata[idx])
            m.update({"score":float(score),"dense_score":d.get("dense_score"),"bm25_score":d.get("bm25_score"),"rank":len(results)+1})
            results.append(m)
        return results
