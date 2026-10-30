
"""Grounded answer generation and structured output validation."""
from __future__ import annotations

import json
import re

from llm import generate
from config import RETRIEVAL_THRESHOLD

REFUSAL = "I don't know based on the provided reports."

SYSTEM = """
You are FinSight, a financial-report question-answering assistant.

RULES:
1. Answer only using the supplied report context.
2. If the context does not support the answer, return exactly:
   I don't know based on the provided reports.
3. Never invent financial figures, dates, names, or facts.
4. For numerical questions, verify the requested fiscal year,
   reporting period, currency/unit, and whether the figure is
   consolidated or standalone.
5. Never substitute a figure from another fiscal year.
6. For comparisons, provide both figures only when the context
   supports both. Calculate percentage change only when the
   required figures are available.
7. Use the supplied source IDs for citations.
8. Return valid JSON with exactly these keys:
   {
     "answer": "Your answer",
     "citations": ["source_id"],
     "confidence": "high"
   }
9. Use confidence "high", "medium", or "low".
10. Keep answers concise and explain figures in complete sentences.
"""


def _parse(raw: str, valid_ids: set[str]) -> dict:
    """Parse structured model output and validate citation IDs."""
    try:
        data = json.loads(raw)

        answer = str(data.get("answer", "")).strip()
        citations = [
            str(c)
            for c in data.get("citations", [])
            if str(c) in valid_ids
        ]

        confidence = str(data.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        if not answer:
            raise ValueError("Empty answer")

        if REFUSAL.lower() in answer.lower():
            return {
                "answer": REFUSAL,
                "citations": [],
                "confidence": "low",
            }

        # An answer without valid citations must not be marked high
        # confidence. The UI may still display the retrieved sources.
        if not citations:
            confidence = "low"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
        }

    except (ValueError, TypeError, json.JSONDecodeError):
        # Do not silently treat arbitrary plain text as a validated
        # high-confidence answer.
        text = raw.strip()

        if not text or REFUSAL.lower() in text.lower():
            return {
                "answer": REFUSAL,
                "citations": [],
                "confidence": "low",
            }

        return {
            "answer": text,
            "citations": [],
            "confidence": "low",
        }


def answer_question(
    query: str,
    results: list[dict],
    provider_hint: str | None = None,
    threshold: float = RETRIEVAL_THRESHOLD,
) -> tuple[dict, str]:

    if not results:
        return {
            "answer": REFUSAL,
            "citations": [],
            "confidence": "low",
        }, "No LLM call"

    # IMPORTANT:
    # Do not reject results solely because their similarity score
    # is below the threshold. Financial reports often contain
    # relevant tables whose retrieval scores are numerically small.
    #
    # The threshold remains available for diagnostics/UI display.
    try:
        top_score = float(results[0].get("score", 0.0))
    except (TypeError, ValueError):
        top_score = 0.0

    # Keep a bounded context to avoid unnecessarily large prompts.
    context_results = results[:4] if provider_hint == "Ollama" else results[:5]

    blocks = []
    for r in context_results:
        blocks.append(
            f"[{r['id']}] "
            f"Company: {r.get('company', 'Unknown')} | "
            f"FY: {r.get('year', 'Unknown')} | "
            f"Page: {r.get('page', 'Unknown')} | "
            f"Section: {r.get('section', 'General')}\n"
            f"{r.get('text', '')}"
        )

    prompt = f"""
QUESTION:
{query}

TOP RETRIEVAL SCORE:
{top_score:.6f}

CONFIGURED RETRIEVAL THRESHOLD:
{threshold:.6f}

REPORT CONTEXT:
{"".join([])}

{chr(10).join(blocks)}

Answer the question using only the report context.
For financial figures, match the exact fiscal year requested.
If the requested fact cannot be established from the context,
return the specified refusal in the JSON answer.
"""

    try:
        raw, provider = generate(
            prompt,
            SYSTEM,
            temperature=0,
            json_mode=True,
        )

        data = _parse(raw, {r["id"] for r in context_results})

        if data["answer"].strip().lower() == REFUSAL.lower():
            data["answer"] = REFUSAL
            data["confidence"] = "low"

        return data, provider

    except Exception:
        # Do not fabricate a response if both generation and fallback
        # fail. Preserve a clear, safe refusal instead.
        return {
            "answer": REFUSAL,
            "citations": [],
            "confidence": "low",
        }, "No LLM call"
