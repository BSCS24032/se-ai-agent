@echo off
REM Convenience launcher for Windows users.
REM Usage:
REM   tasks.cmd install   - create venv and install deps
REM   tasks.cmd test      - run pytest
REM   tasks.cmd run       - launch the Streamlit UI
REM   tasks.cmd cli "your question here"
REM   tasks.cmd check     - run tests + import smoke

IF "%1"=="install" (
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    GOTO end
)

IF "%1"=="test" (
    call venv\Scripts\activate
    python -m pytest -q
    GOTO end
)

IF "%1"=="run" (
    call venv\Scripts\activate
    python -m streamlit run app.py
    GOTO end
)

IF "%1"=="cli" (
    call venv\Scripts\activate
    python -m src.cli %2 %3 %4 %5 %6 %7 %8 %9
    GOTO end
)

IF "%1"=="check" (
    call venv\Scripts\activate
    python -m pytest -q
    python -c "import sys; sys.path.insert(0, '.'); import config; from src import llm_client, query_processor, pubmed_retriever, relevance_filter, evidence_extractor, summary_generator, safety_layer, pipeline, http_utils, text_utils, cli; print('all modules import cleanly')"
    GOTO end
)

echo Unknown target: %1
echo Available: install, test, run, cli, check

:end
