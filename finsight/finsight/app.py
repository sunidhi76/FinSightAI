"""FinSight Streamlit application."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

# Must be the first Streamlit command.
st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide",
)

from sentence_transformers import SentenceTransformer

from config import *
from utils import configure_logging, infer_company_year, keyword_highlight
from ingest import ingest_pdf_items, load_pdfs_from_raw
from retriever import Retriever
from generator import answer_question, REFUSAL
from llm import check_ollama
import cache


# --------------------------------------------------
# PAGE STYLING
# --------------------------------------------------

configure_logging()

st.markdown(
    """
    <style>
    :root {
        --accent: #8b5cf6;
    }

    .block-container {
        padding-top: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827, #312e81);
        color: white;
        margin-bottom: 1rem;
    }

    .card {
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        margin: 0.5rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-weight: 700;
    }

    .good {
        background: #dcfce7;
        color: #166534;
    }

    .bad {
        background: #fee2e2;
        color: #991b1b;
    }

    .small {
        opacity: 0.75;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODELS AND SESSION STATE
# --------------------------------------------------

@st.cache_resource
def embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def disk_retriever(index_id):
    return Retriever(model=embedding_model())


def ensure_state():
    defaults = {
        "messages": [],
        "api_calls": 0,
        "index_id": None,
        "retriever": None,
        "last_search": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


ensure_state()


# --------------------------------------------------
# INGESTION
# --------------------------------------------------

def rebuild(items):
    progress_bar = st.progress(0, text="Starting ingestion...")

    def update_progress(fraction, message):
        progress_bar.progress(
            min(float(fraction), 1.0),
            text=message,
        )

    index_id = ingest_pdf_items(
        items,
        embedding_model(),
        progress=update_progress,
    )

    st.session_state.index_id = index_id
    st.session_state.retriever = Retriever(
        model=embedding_model()
    )

    st.session_state.messages = []
    st.session_state.last_search = None

    progress_bar.progress(1.0, text="Ingestion complete")


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

def answer(query, mode, k, threshold, company=None, year=None):
    retriever = st.session_state.retriever

    # Treat every submitted question independently.
    # This prevents dashboard questions from inheriting
    # the previous question's wording.
    rewritten_query = query
    follow_up = False

    results = retriever.retrieve(
        rewritten_query,
        k,
        mode,
        company,
        year,
    )

    settings = {
        "mode": mode,
        "k": k,
        "company": company,
        "year": year,
        "threshold": threshold,
    }

    cache_key = cache.make_key(
        rewritten_query,
        settings,
        retriever.index_id,
    )

    cached = cache.get(cache_key)

    if cached:
        data = cached["data"]
        provider = cached["provider"] + " (cache)"
    else:
        data, provider = answer_question(
            rewritten_query,
            results,
            threshold=threshold,
        )

        if (
            provider not in ("No LLM call", "")
            and "cache" not in provider.lower()
        ):
            st.session_state.api_calls += 1

        cache.set(
            cache_key,
            {
                "data": data,
                "provider": provider,
            },
        )

    st.session_state.last_search = {
        "original": query,
        "rewritten": rewritten_query,
        "mode": mode,
        "results": results,
        "follow_up": follow_up,
    }

    return data, provider, results


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("FinSight")
st.sidebar.caption(
    "Financial Report Analyser • Query-time RAG"
)

st.sidebar.metric(
    "API calls used this session",
    st.session_state.api_calls,
)

if GEMINI_API_KEY and GEMINI_MODEL:
    st.sidebar.success(
        f"Gemini configured: {GEMINI_MODEL}"
    )
else:
    st.sidebar.warning(
        "Gemini is not fully configured. "
        "Check GEMINI_API_KEY and GEMINI_MODEL in .env."
    )

ollama_ok, ollama_reason = check_ollama()

if ollama_ok:
    st.sidebar.info(
        f"Fallback ready: Ollama • {OLLAMA_MODEL}"
    )
else:
    st.sidebar.caption(
        "Ollama fallback unavailable; Gemini can still run."
    )


# --------------------------------------------------
# UPLOAD REPORTS
# --------------------------------------------------

uploads = st.sidebar.file_uploader(
    "Add annual reports",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploads:
    metadata = []

    st.sidebar.markdown("**Uploaded report metadata**")

    for uploaded_file in uploads:
        company_name, fiscal_year = infer_company_year(
            uploaded_file.name
        )

        company_name = st.sidebar.text_input(
            f"Company — {uploaded_file.name}",
            company_name,
            key=f"company_{uploaded_file.name}",
        )

        fiscal_year = st.sidebar.text_input(
            f"Fiscal year — {uploaded_file.name}",
            fiscal_year,
            key=f"year_{uploaded_file.name}",
        )

        metadata.append(
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                company_name,
                fiscal_year,
            )
        )

    if st.sidebar.button(
        "Build session index",
        use_container_width=True,
    ):
        try:
            rebuild(metadata)
            st.rerun()
        except Exception as exc:
            st.sidebar.error(
                f"Ingestion failed: {exc}"
            )


if st.sidebar.button(
    "Load sample reports",
    use_container_width=True,
):
    raw_reports = load_pdfs_from_raw()

    if not raw_reports:
        st.sidebar.warning(
            "No PDFs in data/raw/. Add public annual reports there."
        )
    else:
        try:
            rebuild(raw_reports)
            st.rerun()
        except Exception as exc:
            st.sidebar.error(
                f"Ingestion failed: {exc}"
            )


# --------------------------------------------------
# RETRIEVAL SETTINGS
# --------------------------------------------------

st.sidebar.divider()

mode = st.sidebar.radio(
    "Retrieval mode",
    ["Hybrid", "Dense only", "BM25 only"],
)

k = st.sidebar.slider(
    "Chunks to retrieve",
    3,
    10,
    DEFAULT_K,
)

threshold = st.sidebar.slider(
    "Grounding threshold",
    0.0,
    0.5,
    RETRIEVAL_THRESHOLD,
    0.01,
)

compare = st.sidebar.toggle(
    "Compare two companies"
)

if st.session_state.retriever:
    companies = sorted(
        {
            item["company"]
            for item in st.session_state.retriever.metadata
        }
    )

    years = sorted(
        {
            str(item["year"])
            for item in st.session_state.retriever.metadata
        }
    )

    selected_company = st.sidebar.selectbox(
        "Company filter",
        ["All"] + companies,
    )

    selected_year = st.sidebar.selectbox(
        "Fiscal year",
        ["All"] + years,
    )

    company = (
        None if selected_company == "All"
        else selected_company
    )

    year = (
        None if selected_year == "All"
        else selected_year
    )
else:
    company = None
    year = None


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📊 FinSight</h1>
        <p>
            Ask long financial reports questions in plain English
            — grounded, cited, and comparable.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not st.session_state.retriever:
    st.info(
        "Add reports to begin. Use the sidebar to upload PDFs "
        "or load PDFs from data/raw/."
    )

    st.markdown(
        "**Why annual reports?** They are public, dense, numeric, "
        "and long — exactly the kind of documents where RAG "
        "saves manual searching."
    )

else:
    tabs = st.tabs(["Ask", "Key metrics"])

    # --------------------------------------------------
    # ASK TAB
    # --------------------------------------------------

    with tabs[0]:
        examples = [
            "What was the latest-year revenue?",
            "What are the main risk factors?",
            "How did operating margin change?",
            "What guidance did management provide?",
        ]

        columns = st.columns(4)

        for index, question_text in enumerate(examples):
            if columns[index].button(
                question_text,
                key=f"example_{index}",
            ):
                st.session_state.pending = question_text
                st.session_state.pop("last_search", None)

        question = st.chat_input(
            "Ask a question about the reports..."
        )

        if not question:
            question = st.session_state.pop(
                "pending",
                None,
            )

        if question:
            if compare:
                available_companies = sorted(
                    {
                        item["company"]
                        for item in st.session_state.retriever.metadata
                    }
                )

                if len(available_companies) >= 2:
                    first_company = available_companies[0]
                    second_company = available_companies[1]

                    first_results = (
                        st.session_state.retriever.retrieve(
                            question,
                            k,
                            mode,
                            first_company,
                            None,
                        )
                    )

                    second_results = (
                        st.session_state.retriever.retrieve(
                            question,
                            k,
                            mode,
                            second_company,
                            None,
                        )
                    )

                    combined_results = (
                        first_results + second_results
                    )

                    data, provider = answer_question(
                        question,
                        combined_results,
                    )

                    st.session_state.last_search = {
                        "original": question,
                        "rewritten": question,
                        "mode": mode,
                        "results": combined_results,
                    }

                    st.session_state.messages.append(
                        {
                            "question": question,
                            "answer": data["answer"],
                        }
                    )

                    st.markdown(
                        f"**Using {provider}**"
                    )
                    st.markdown(data["answer"])

                    comparison_rows = []

                    for company_name, company_results in [
                        (first_company, first_results),
                        (second_company, second_results),
                    ]:
                        comparison_rows.append(
                            {
                                "Company": company_name,
                                "Answer": (
                                    data["answer"]
                                    if not company_results
                                    else "See cited context"
                                ),
                                "Top relevance": (
                                    round(
                                        float(
                                            company_results[0]["score"]
                                        ),
                                        4,
                                    )
                                    if company_results
                                    else 0
                                ),
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(comparison_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

                else:
                    st.warning(
                        "Compare mode needs at least two companies."
                    )

            else:
                data, provider, results = answer(
                    question,
                    mode,
                    k,
                    threshold,
                    company,
                    year,
                )

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": data["answer"],
                    }
                )

                # --------------------------------------------------
                # ANSWER STATUS
                # --------------------------------------------------

                answer_text = str(
                    data.get("answer", "")
                ).strip()

                citations = data.get("citations", [])
                has_citations = bool(citations)

                is_refusal = (
                    answer_text.casefold()
                    == REFUSAL.strip().casefold()
                )

                if is_refusal:
                    badge = "Not in documents"
                    badge_class = "bad"

                elif has_citations:
                    badge = "Grounded"
                    badge_class = "good"

                else:
                    badge = "Answer generated · verify sources"
                    badge_class = "bad"

                st.markdown(
                    f'<span class="badge {badge_class}">'
                    f'{badge}</span> &nbsp; '
                    f'**Using {provider}**',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"### {answer_text}"
                )

                confidence = str(
                    data.get("confidence", "low")
                ).lower()

                confidence_values = {
                    "high": 1.0,
                    "medium": 0.65,
                    "low": 0.25,
                }

                st.progress(
                    confidence_values.get(
                        confidence,
                        0.25,
                    ),
                    text=f"Confidence: {confidence.title()}",
                )

                with st.expander("How I searched"):
                    search_info = (
                        st.session_state.last_search
                    )

                    st.write(
                        "Original query:",
                        search_info["original"],
                    )

                    st.write(
                        "Rewritten query:",
                        search_info["rewritten"],
                    )

                    st.write(
                        "Retrieval:",
                        search_info["mode"],
                    )

                    st.write(
                        "Chunks used:",
                        len(results),
                    )

                st.subheader("Sources")

                if not results:
                    st.info(
                        "No source passages were retrieved."
                    )

                for source_index, result in enumerate(results, 1):
                    if (
                        citations
                        and result["id"] not in citations
                    ):
                        continue

                    with st.container(border=True):
                        st.markdown(
                            f"**[{source_index}] "
                            f"{result.get('company', 'Unknown')} • "
                            f"FY {result.get('year', 'Unknown')} • "
                            f"p. {result.get('page', 'Unknown')}**"
                        )

                        st.caption(
                            f"Relevance: "
                            f"{float(result.get('score', 0.0)):.4f} "
                            f"• {result.get('section', 'General')}"
                        )

                        st.markdown(
                            keyword_highlight(
                                result.get("text", ""),
                                question,
                            )
                        )

        # --------------------------------------------------
        # CHAT EXPORT
        # --------------------------------------------------

        if st.session_state.messages:
            chat_markdown = "# FinSight Chat Export\n\n"

            for message in st.session_state.messages[-20:]:
                chat_markdown += (
                    f"**Q:** {message['question']}\n\n"
                    f"**A:** {message['answer']}\n\n"
                    "---\n"
                )

            st.download_button(
                "Download chat as Markdown",
                chat_markdown,
                "finsight_chat.md",
                "text/markdown",
            )

    # --------------------------------------------------
    # KEY METRICS TAB
    # --------------------------------------------------

    with tabs[1]:
        available_companies = sorted(
            {
                item["company"]
                for item in st.session_state.retriever.metadata
            }
        )

        selected_company = st.selectbox(
            "Company",
            available_companies,
        )

        if st.button("Extract key metrics"):
            prompt = (
                "Extract revenue, net profit, EPS and YoY growth "
                "from the provided context. Return JSON with "
                "metric, value, unit, fiscal_year, citation_id."
            )

            results = st.session_state.retriever.retrieve(
                (
                    f"revenue net profit EPS earnings per share "
                    f"YoY growth {selected_company}"
                ),
                8,
                "Hybrid",
                selected_company,
                None,
            )

            from llm import generate

            context = "\n\n".join(
                (
                    f"[{result['id']}] "
                    f"FY {result.get('year', 'Unknown')} "
                    f"p.{result.get('page', 'Unknown')}\n"
                    f"{result.get('text', '')}"
                )
                for result in results[:4]
            )

            raw, provider = generate(
                prompt + "\nCONTEXT:\n" + context,
                "Extract ONLY from context. Return JSON.",
                json_mode=True,
            )

            try:
                parsed = json.loads(raw)

                if isinstance(parsed, list):
                    rows = parsed
                else:
                    rows = parsed.get("metrics", [])

            except (json.JSONDecodeError, TypeError):
                rows = []

            if rows:
                dataframe = pd.DataFrame(rows)

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                numeric_values = pd.to_numeric(
                    dataframe.get(
                        "value",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                )

                if numeric_values.notna().any():
                    figure = px.bar(
                        dataframe,
                        x="metric",
                        y=numeric_values,
                        title=(
                            f"{selected_company} key metrics"
                        ),
                    )

                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                    )

            else:
                st.warning(
                    "The model did not return structured metrics. "
                    "Ask a direct question for cited retrieval instead."
                )