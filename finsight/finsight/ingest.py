
"""PDF ingestion, cleaning, chunking and FAISS/BM25 index construction."""
from __future__ import annotations
import io, json, logging, re
from pathlib import Path
from typing import Callable, BinaryIO
import numpy as np
import fitz
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import *
from utils import clean_page_text, infer_company_year, infer_section, recursive_chunks, file_hash, text_hash

LOGGER=logging.getLogger("finsight.ingest")

def extract_pages(data: bytes) -> list[tuple[int,str]]:
    pages=[]
    try:
        doc=fitz.open(stream=data, filetype="pdf")
        for i,page in enumerate(doc):
            pages.append((i+1, clean_page_text(page.get_text("text"))))
        doc.close()
    except Exception:
        pages=[]
    if not any(t.strip() for _,t in pages):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages=[(i+1, clean_page_text(p.extract_text() or "")) for i,p in enumerate(pdf.pages)]
        except Exception as exc:
            LOGGER.warning("pdfplumber fallback failed: %s", exc.__class__.__name__)
    return pages

def build_records(pdf_items: list[tuple[str,bytes,str|None,str|None]], page_ranges: dict[str,tuple[int,int]]|None=None,
                  progress: Callable[[float,str],None]|None=None) -> list[dict]:
    records=[]
    total=len(pdf_items)
    for fi,(filename,data,company,year) in enumerate(pdf_items):
        inferred_company,inferred_year=infer_company_year(filename)
        company=company or inferred_company
        year=year or inferred_year
        pages=extract_pages(data)
        if not any(text for _,text in pages):
            LOGGER.warning("No extractable text in %s; it may be scanned.", filename)
            continue
        selected=page_ranges.get(filename) if page_ranges else None
        for page,text in pages:
            if selected and not (selected[0] <= page <= selected[1]): continue
            if not text: continue
            section=infer_section(text)
            for ci,chunk in enumerate(recursive_chunks(text,CHUNK_SIZE,CHUNK_OVERLAP)):
                records.append({
                    "id": f"{file_hash(data)[:10]}-p{page}-c{ci}",
                    "text":chunk,"company":company,"year":year,"page":page,
                    "section":section,"source":filename
                })
        if progress: progress((fi+1)/total, f"Ingested {filename}: {len(pages)} pages")
    return records

def embed_records(records:list[dict], model:SentenceTransformer, progress=None) -> np.ndarray:
    texts=[r["text"] for r in records]
    vectors=model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype("float32")

def save_index(records:list[dict], vectors:np.ndarray, destination:Path=INDEX_DIR) -> str:
    destination.mkdir(parents=True,exist_ok=True)
    index=faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index,str(destination/"faiss.index"))
    (destination/"metadata.json").write_text(json.dumps(records,ensure_ascii=False),encoding="utf-8")
    tokenized=[r["text"].lower().split() for r in records]
    (destination/"bm25.json").write_text(json.dumps(tokenized),encoding="utf-8")
    index_id=text_hash(json.dumps([(r["id"],r["text"]) for r in records],sort_keys=True))[:16]
    (destination/"index_info.json").write_text(json.dumps({"index_id":index_id,"count":len(records),"dim":vectors.shape[1]}),encoding="utf-8")
    return index_id

def load_pdfs_from_raw() -> list[tuple[str,bytes,None,None]]:
    return [(p.name,p.read_bytes(),None,None) for p in sorted(RAW_DIR.glob("*.pdf"))]

def ingest_pdf_items(pdf_items, model, destination=INDEX_DIR, progress=None):
    records=build_records(pdf_items,progress=progress)
    if not records: raise ValueError("No extractable text was found. Check that the PDFs are text-based and not scanned.")
    vectors=embed_records(records,model)
    return save_index(records,vectors,destination)
