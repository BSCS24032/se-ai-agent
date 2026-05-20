# AI Research Agent for Scientific Evidence Extraction

**Version 1 - Coursework Prototype**

An AI-powered research assistant that turns a plain-English health
question into a structured, source-grounded scientific evidence
summary backed by PubMed literature.

> Research-support tool only. Not a doctor, not a diagnostic system,
> and not a substitute for medical advice. All findings should be
> verified against the original sources.

---

## What it does

1. You type a health or biomedical research question in plain English.
2. The system reformulates it into a PubMed-friendly query using an
   LLM.
3. It retrieves up to 10 PubMed abstracts via NCBI Entrez E-Utilities,
   with automatic retry on transient failures.
4. It extracts structured evidence from each abstract - study type,
   population, intervention/exposure, outcome, key finding, and a
   verbatim supporting quote.
5. It generates a concise, grounded summary that cites papers by PMID
   and is checked for citation coverage and obvious hallucination.
6. Everything is displayed in a clean Streamlit interface, or via a
   CLI for scripting.

## Quick start

```bash
# 1. Create a virtual environment
python -m venv venv
# Windows:   venv\Scripts\activate
# Unix:      source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env       # Unix
copy .env.example .env     # Windows
# Open .env and add your Groq API key (free at https://console.groq.com/keys)

# 4. Run the UI
streamlit run app.py

# OR run from the terminal
python -m src.cli "Does intermittent fasting improve insulin sensitivity?"
```

Convenience targets are also provided:
- Unix / macOS: `make install`, `make test`, `make run`, `make cli Q="..."`
- Windows: `tasks.cmd install`, `tasks.cmd test`, `tasks.cmd run`, `tasks.cmd cli "..."`

The full step-by-step setup guide is in [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Architecture (at a glance)

```
User question
   |
   v
[Safety classifier] --> blocks crisis queries; warns on personal-advice
   |
   v
[Query Processor]    --> LLM reformulates -> PubMed-friendly query
   |
   v
[PubMed Retriever]   --> esearch / efetch (retry on 429/5xx)
   |
   v
[Relevance Filter]   --> drop empty / non-English; rank by overlap +
   |                     study-type boost
   v
[Evidence Extractor] --> LLM -> structured JSON + verbatim quote
   |
   v
[Summary Generator]  --> LLM -> grounded summary with PMIDs
   |
   v
[Safety grounding]   --> warns on missing / foreign / low-coverage citations
   |
   v
[Streamlit UI / CLI]
```

A Mermaid version of the same diagram lives in
[docs/architecture.mmd](docs/architecture.mmd).

Each layer is isolated so future versions can swap or extend a single
piece (V2 adds storage, V3 adds full-text/vector search, V4 adds
orchestration) without rewriting the rest. See
[docs/architecture.md](docs/architecture.md) for the full design
write-up that maps to Assignment 3.

## Project layout

```
SE AI Agent/
  app.py                     # Streamlit entry point
  config.py                  # Environment / settings loader
  requirements.txt
  .env.example
  README.md
  SETUP_GUIDE.md             # Morning-after step-by-step
  CHANGELOG.md
  LICENSE
  Makefile                   # Unix convenience targets
  tasks.cmd                  # Windows convenience targets
  src/
    cli.py                   # Terminal runner
    http_utils.py            # Retry/backoff wrapper
    llm_client.py            # LLM abstraction (Groq / Ollama / OpenAI / stub)
    query_processor.py       # Query validation + reformulation
    pubmed_retriever.py      # NCBI Entrez E-Utilities client
    relevance_filter.py      # Filter + rank abstracts
    evidence_extractor.py    # Structured JSON extraction + verbatim quote
    summary_generator.py     # Grounded summary synthesis
    safety_layer.py          # Disclaimers + safety + grounding checks
    pipeline.py              # Orchestrates the full flow
    text_utils.py            # Shared tokenizer / stopwords / English check
  tests/                     # pytest suite (57 tests, all passing)
  docs/                      # architecture + usage docs + mermaid diagram
```

## V1 scope

In scope:
- Single health question per session
- PubMed abstract retrieval (up to 10 results) with retry/backoff
- Structured evidence extraction (study type, population, intervention,
  outcome, key finding, verbatim supporting quote)
- Grounded summary with PMID references and grounding checks
- Streamlit UI with downloads, sorting, and badges
- CLI with JSON output for scripting
- Disclaimer and safety classification on every input

Out of scope (planned for later versions):
- Full PDF ingestion (V3)
- User accounts and saved history (V2 / V5)
- Vector search / RAG (V3)
- Multi-agent reasoning (V4)
- Citation graph reasoning (V4 / V5)

## License

MIT - see [LICENSE](LICENSE).
