# FinSight Evaluation

The benchmark has six questions: three direct facts, one rephrased fact, one cross-company comparison and one intentionally unanswerable question.

| Metric | Meaning |
|---|---|
| Retrieval Hit@5 | Expected page appears in top five retrieved chunks |
| Answer correctness | Expected keywords appear in the generated answer |
| Citation present | At least one retrieved chunk ID is cited |
| Refusal correct | Unanswerable question receives the exact refusal |
| Latency | Answer-generation wall time |
| Provider | Gemini, Ollama, fallback or cache |

> Populate `expected_pages` and `gold_keywords` in `test_questions.json` after selecting real annual reports. The starter benchmark deliberately does not invent company-specific facts.

## Failure analysis

### Failure case 1
- **Observed:** ____________________
- **Cause:** table split across chunks / BM25 vs dense disagreement / unit mismatch
- **Fix attempted:** ____________________
- **Result:** ____________________

### Failure case 2
- **Observed:** ____________________
- **Cause:** table split across chunks / BM25 vs dense disagreement / unit mismatch
- **Fix attempted:** ____________________
- **Result:** ____________________

## Gemini vs Ollama

| Question | Gemini correctness | Gemini latency | Ollama correctness | Ollama latency |
|---|---:|---:|---:|---:|
| q1 | — | — | — | — |
| q2 | — | — | — | — |
| q3 | — | — | — | — |
| q4 | — | — | — | — |
| q5 | — | — | — | — |
| q6 | — | — | — | — |

Run the same benchmark with:
```bash
python eval.py --provider gemini
python eval.py --provider ollama
```

## Surprise question checklist
- [ ] Is the question answerable from the uploaded reports?
- [ ] Does the retriever return the correct company?
- [ ] Does it return the correct fiscal year?
- [ ] Does the top-5 context contain the needed page?
- [ ] Is the financial unit preserved?
- [ ] Is the answer grounded only in context?
- [ ] Are citation IDs present?
- [ ] Does the source card expose the exact passage?
- [ ] If absent, does the app refuse instead of guessing?
- [ ] Can I explain why Hybrid retrieval helped?
