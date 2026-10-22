
"""Shared utilities for FinSight."""
from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("finsight")

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_page_text(text: str) -> str:
    text = normalize_text(text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"(page\s*)?\d{1,4}", ln, re.I)]
    if len(lines) > 4:
        # Repeated one-line headers/footers are common in annual reports.
        counts = {}
        for ln in lines:
            counts[ln] = counts.get(ln, 0) + 1
        repeated = {ln for ln, n in counts.items() if n >= 3 and len(ln) < 100}
        lines = [ln for ln in lines if ln not in repeated]
    return normalize_text("\n".join(lines))

def infer_company_year(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    year_match = re.search(r"(?:FY|FY-|20)\s*(20\d{2})", stem, re.I)
    year = year_match.group(1) if year_match else "Unknown"
    company = re.sub(r"[_-]*(?:FY|20\d{2})[_-]*", " ", stem, flags=re.I)
    company = re.sub(r"[_-]+", " ", company).strip() or "Unknown Company"
    return company.title(), year

def infer_section(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ("management discussion", "management's discussion", "md&a")):
        return "MD&A"
    if any(x in low for x in ("consolidated statement", "balance sheet", "income statement", "cash flow")):
        return "Financial Statements"
    if any(x in low for x in ("risk factors", "risks and uncertainties")):
        return "Risk Factors"
    if any(x in low for x in ("directors' report", "board's report")):
        return "Directors' Report"
    return "General"

def recursive_chunks(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text recursively while preferring paragraph/sentence boundaries."""
    text = normalize_text(text)
    if len(text) <= size:
        return [text] if text else []
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    pieces = [text]
    for sep in separators:
        if max(map(len, pieces)) <= size:
            break
        new=[]
        for piece in pieces:
            if len(piece) <= size:
                new.append(piece)
                continue
            parts=piece.split(sep)
            cur=""
            for part in parts:
                candidate=(cur+sep+part).strip() if cur else part.strip()
                if len(candidate) <= size:
                    cur=candidate
                else:
                    if cur: new.append(cur)
                    cur=part.strip()
            if cur: new.append(cur)
        pieces=new
    chunks=[]
    for piece in pieces:
        if not piece: continue
        if not chunks:
            chunks.append(piece[:size])
        else:
            prev=chunks[-1]
            prefix=prev[-overlap:] if overlap else ""
            if len(piece) + len(prefix) <= size:
                chunks.append((prefix+" "+piece).strip())
            else:
                chunks.append(piece[:size])
                rest=piece[size-overlap:] if len(piece)>size else ""
                if rest: chunks.append(rest)
    return [c.strip() for c in chunks if c.strip()]

def keyword_highlight(text: str, query: str, max_len: int = 700) -> str:
    terms=[t for t in re.findall(r"[A-Za-z0-9₹$%.-]{3,}", query) if t.lower() not in {"what","when","where","which","does","did","the","this","that","from"}]
    snippet=text
    if len(snippet)>max_len:
        low=snippet.lower()
        positions=[low.find(t.lower()) for t in terms if low.find(t.lower())>=0]
        start=max(0,(min(positions)-180) if positions else 0)
        snippet=snippet[start:start+max_len]
        if start: snippet="…"+snippet
    for term in sorted(set(terms), key=len, reverse=True):
        snippet=re.sub(re.escape(term), lambda m: f"**{m.group(0)}**", snippet, flags=re.I)
    return snippet
