"""Streamlit entry point for the AI Research Agent (V1).

Run with:
    streamlit run app.py

The UI is intentionally simple: one input box, a progress bar, and
clearly labelled result sections (reformulated query -> articles ->
structured evidence -> grounded summary -> disclaimer).
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import streamlit as st

# Ensure local modules import cleanly when Streamlit launches from any cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import SETTINGS  # noqa: E402
from src.pipeline import Pipeline, PipelineResult  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent - Health Literature",
    page_icon=":microscope:",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "Does intermittent fasting improve insulin sensitivity in adults "
    "with type 2 diabetes?",
    "Is metformin associated with reduced cancer incidence in patients "
    "with prediabetes?",
    "What is the effect of resistance training on bone mineral density "
    "in postmenopausal women?",
    "Does vitamin D supplementation reduce respiratory infections in "
    "older adults?",
]


def _init_state() -> None:
    """Initialize session state once per browser tab."""
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "sort_mode" not in st.session_state:
        st.session_state["sort_mode"] = "Relevance"


_init_state()


# ---------------------------------------------------------------------------
# Sidebar - configuration + about
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.write(f"**LLM provider:** `{SETTINGS.llm_provider}`")
    if SETTINGS.llm_provider == "groq":
        st.write(f"**Model:** `{SETTINGS.groq_model}`")
        if not SETTINGS.groq_api_key:
            st.warning(
                "No Groq API key found. Add GROQ_API_KEY to your .env file "
                "for AI-powered reformulation, extraction, and summary. "
                "Otherwise the app falls back to a keyword-only demo."
            )
        else:
            st.success("Groq API key detected.")
    elif SETTINGS.llm_provider == "ollama":
        st.write(f"**Model:** `{SETTINGS.ollama_model}`")
        st.write(f"**Endpoint:** `{SETTINGS.ollama_base_url}`")
    elif SETTINGS.llm_provider == "openai":
        st.write(f"**Model:** `{SETTINGS.openai_model}`")
        if not SETTINGS.openai_api_key:
            st.warning("No OPENAI_API_KEY found.")
    else:
        st.info("Running in no-LLM mode. AI-generated outputs are disabled.")

    st.write(f"**Max articles per query:** {SETTINGS.max_articles}")
    if SETTINGS.ncbi_email:
        st.caption(f"NCBI contact: {SETTINGS.ncbi_email}")
    else:
        st.caption(
            "Set NCBI_EMAIL in .env to be polite to NCBI rate limits "
            "(strongly recommended)."
        )

    st.divider()
    st.header("About")
    st.markdown(
        "**Version 1** of an AI Research Agent for scientific evidence "
        "extraction in health literature. Turns a plain-English health "
        "question into a PubMed search, extracts structured evidence "
        "from each abstract, and produces a grounded summary."
    )
    st.caption(
        "Research-support tool. Not medical advice. "
        "Always verify findings against the cited sources."
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("AI Research Agent")
st.subheader("Scientific evidence extraction from health literature")

st.warning(SETTINGS.disclaimer, icon=":material/info:")

# Example question chips
st.caption("Try an example question:")
chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
for idx, ex in enumerate(EXAMPLE_QUESTIONS):
    if chip_cols[idx].button(
        ex[:55] + ("..." if len(ex) > 55 else ""),
        key=f"ex_{idx}",
        use_container_width=True,
    ):
        st.session_state["question"] = ex

with st.form("query_form"):
    question = st.text_area(
        "Ask a health or biomedical research question",
        value=st.session_state["question"],
        placeholder=(
            "e.g. Does intermittent fasting improve insulin sensitivity "
            "in adults with type 2 diabetes?"
        ),
        height=120,
        max_chars=500,
    )
    col_left, col_right = st.columns([1, 1])
    with col_left:
        submitted = st.form_submit_button(
            "Run research pipeline", type="primary", use_container_width=True
        )
    with col_right:
        reset = st.form_submit_button("Reset", use_container_width=True)

if reset:
    st.session_state["result"] = None
    st.session_state["question"] = ""
    st.rerun()

if submitted:
    if not question or not question.strip():
        st.error("Please enter a question first.")
    else:
        st.session_state["question"] = question
        st.session_state["running"] = True
        progress = st.progress(0.0, text="Starting...")

        def _cb(message: str, fraction: float) -> None:
            progress.progress(min(max(fraction, 0.0), 1.0), text=message)

        pipeline = Pipeline()
        try:
            result: PipelineResult = pipeline.run(question, progress=_cb)
        except Exception as exc:  # pragma: no cover - last-resort guard
            st.session_state["running"] = False
            st.exception(exc)
            st.stop()
        st.session_state["result"] = result
        st.session_state["running"] = False
        progress.empty()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
_STUDY_TYPE_COLOR = {
    "Meta-analysis": "violet",
    "Systematic review": "violet",
    "RCT": "green",
    "Clinical trial": "green",
    "Cohort": "blue",
    "Case-control": "blue",
    "Review": "orange",
    "Other": "gray",
}


def _badge(label: str) -> str:
    """Return a Streamlit-coloured badge string for the given study type."""
    color = _STUDY_TYPE_COLOR.get(label, "gray")
    return f":{color}-badge[{label}]"


def _result_to_json(result: PipelineResult) -> str:
    payload = {
        "question": result.question,
        "reformulated_query": (
            result.processed_query.reformulated
            if result.processed_query
            else ""
        ),
        "summary": (
            result.summary_with_footer or (result.summary.text if result.summary else "")
        ),
        "grounding_warnings": result.grounding_warnings,
        "evidence": [
            {
                "pmid": e.pmid,
                "title": e.title,
                "study_type": e.study_type,
                "population": e.population,
                "intervention": e.intervention,
                "outcome": e.outcome,
                "key_finding": e.key_finding,
                "supporting_quote": e.supporting_quote,
                "status": e.status,
                "error": e.error,
            }
            for e in result.evidence
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _result_to_csv(result: PipelineResult) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "pmid",
            "title",
            "year",
            "journal",
            "top_study_type",
            "extracted_study_type",
            "population",
            "intervention",
            "outcome",
            "key_finding",
            "supporting_quote",
            "url",
            "status",
        ]
    )
    for art, ev in zip(result.selected_articles, result.evidence):
        writer.writerow(
            [
                art.pmid,
                art.title,
                art.year,
                art.journal,
                art.top_study_type,
                ev.study_type,
                ev.population,
                ev.intervention,
                ev.outcome,
                ev.key_finding,
                ev.supporting_quote,
                art.url,
                ev.status,
            ]
        )
    return buf.getvalue()


def _sort_indices(result: PipelineResult, mode: str):
    pairs = list(zip(result.selected_articles, result.evidence))
    if mode == "Year (newest)":
        return sorted(
            range(len(pairs)),
            key=lambda i: (pairs[i][0].year or "0000"),
            reverse=True,
        )
    if mode == "Study type":
        order = list(_STUDY_TYPE_COLOR.keys())
        return sorted(
            range(len(pairs)),
            key=lambda i: order.index(pairs[i][0].top_study_type)
            if pairs[i][0].top_study_type in order
            else len(order),
        )
    return list(range(len(pairs)))  # Relevance (already sorted)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(result: PipelineResult) -> None:
    if not result.safety.allowed:
        st.error(result.safety.blocked_reason)
        return

    for warning in result.safety.warnings:
        st.warning(warning, icon=":material/warning:")

    for err in result.errors:
        st.warning(err, icon=":material/warning:")

    if result.processed_query:
        with st.expander("Reformulated PubMed query", expanded=True):
            pq = result.processed_query
            st.code(pq.reformulated, language="text")
            if pq.used_llm:
                st.caption("Generated by the LLM.")
            else:
                st.caption(pq.notes or "Generated by the keyword fallback.")

    if result.summary:
        st.subheader("Grounded summary")
        for w in result.grounding_warnings:
            st.warning(w, icon=":material/warning:")
        st.markdown(result.summary_with_footer or result.summary.text)
        if not result.summary.used_llm:
            st.caption(
                "This summary was produced without an LLM (fallback mode)."
            )

    if result.evidence:
        st.subheader("Structured evidence")
        col_sort, col_dl1, col_dl2 = st.columns([2, 1, 1])
        with col_sort:
            st.session_state["sort_mode"] = st.selectbox(
                "Sort by",
                options=["Relevance", "Year (newest)", "Study type"],
                index=["Relevance", "Year (newest)", "Study type"].index(
                    st.session_state.get("sort_mode", "Relevance")
                ),
            )
        with col_dl1:
            st.download_button(
                "Download JSON",
                data=_result_to_json(result),
                file_name="evidence.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Download CSV",
                data=_result_to_csv(result),
                file_name="evidence.csv",
                mime="text/csv",
                use_container_width=True,
            )

        order = _sort_indices(result, st.session_state["sort_mode"])
        for i in order:
            art = result.selected_articles[i]
            ev = result.evidence[i]
            with st.container(border=True):
                badge = _badge(art.top_study_type)
                st.markdown(f"{badge} **{art.title or '(untitled)'}**")
                meta_bits = []
                if art.authors_short():
                    meta_bits.append(art.authors_short())
                if art.year:
                    meta_bits.append(art.year)
                if art.journal:
                    meta_bits.append(art.journal)
                if meta_bits:
                    st.caption(" - ".join(meta_bits))
                st.caption(f"PMID: [{art.pmid}]({art.url})")

                if ev.status == "ok":
                    cols = st.columns(5)
                    cols[0].markdown(
                        f"**Study type**\n\n{ev.study_type}"
                    )
                    cols[1].markdown(
                        f"**Population**\n\n{ev.population}"
                    )
                    cols[2].markdown(
                        f"**Intervention**\n\n{ev.intervention}"
                    )
                    cols[3].markdown(f"**Outcome**\n\n{ev.outcome}")
                    cols[4].markdown(
                        f"**Key finding**\n\n{ev.key_finding}"
                    )
                    if ev.supporting_quote:
                        st.markdown(
                            f"> _Supporting quote (verbatim from "
                            f"abstract):_ \"{ev.supporting_quote}\""
                        )
                else:
                    st.warning(
                        f"Evidence extraction failed: {ev.error}",
                        icon=":material/warning:",
                    )
                with st.expander("Abstract"):
                    st.write(art.abstract or "(no abstract available)")

    if result.retrieved_articles:
        with st.expander(
            f"Full retrieved article list "
            f"({len(result.retrieved_articles)})",
            expanded=False,
        ):
            for art in result.retrieved_articles:
                st.markdown(
                    f"- [{art.pmid}]({art.url}) - "
                    f"{art.title or '(untitled)'}"
                )


result = st.session_state.get("result")
if result is not None:
    _render_result(result)
else:
    st.info(
        "Type a health research question above, or click one of the "
        "example chips to begin."
    )
