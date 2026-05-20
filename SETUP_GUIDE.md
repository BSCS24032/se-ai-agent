# Morning-after Setup Guide

Hi. This guide walks you through everything you need to do on your own
machine to get the AI Research Agent V1 running. I built and polished
the code while you slept; this file is the runbook to bring it to
life.

Each step has an estimated time and a note about what to do if it
fails.

---

## What's already done

- Full project scaffolded in `C:\Users\gummy\OneDrive\Desktop\New folder\SE AI Agent`.
- 57 unit/integration tests, all green.
- Architecture, usage, and changelog docs.
- A working git history with per-task commits, bundled separately (see
  step 8 below — the OneDrive folder cannot host the .git directory
  directly from my sandbox).

## What you need to do

1. Install Python (5 min)
2. Open a terminal in the project folder (1 min)
3. Create a virtual environment and install dependencies (3 min)
4. Create your `.env` and get a free Groq API key (5 min)
5. Run the app and try a question (1 min)
6. (Optional) Set up NCBI rate-limit credentials (3 min)
7. (Optional) Deploy to Streamlit Community Cloud (10 min)
8. (Optional) Restore the git history (2 min)
9. Sanity check before submitting (3 min)

Total: ~15-30 minutes depending on how many optional steps you do.

---

## 1. Install Python (5 minutes)

You need Python 3.9 or newer. On Windows the easiest path is:

1. Open Microsoft Store.
2. Search for "Python 3.12" (or 3.11; both fine).
3. Click *Get* / *Install*.

To verify, open PowerShell and run:

```powershell
python --version
```

You should see `Python 3.x.x`. If `python` is not recognized, log out
and back in once.

> Alternative: download from <https://www.python.org/downloads/>. Be
> sure to tick "Add Python to PATH" in the installer.

## 2. Open a terminal in the project folder (1 minute)

In File Explorer, navigate to:

```
C:\Users\gummy\OneDrive\Desktop\New folder\SE AI Agent
```

Click the address bar, type `powershell`, and press Enter. Confirm with:

```powershell
dir
```

You should see `app.py`, `config.py`, `requirements.txt`, `src\`,
`tests\`, `docs\`, `README.md`, `SETUP_GUIDE.md`, `Makefile`,
`tasks.cmd`, `.env.example`, etc.

## 3. Create a virtual environment and install dependencies (3 minutes)

Pick one of the two paths:

### Option A: convenience script (recommended)

```powershell
tasks.cmd install
tasks.cmd test
```

You should see `57 passed`.

### Option B: manual

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

Same expected result. If you see import errors, check that the
virtual environment is activated (the prompt should start with
`(venv)`).

## 4. Create `.env` and get a free Groq API key (5 minutes)

The default LLM provider is Groq because it offers a generous free
tier and runs Llama 3.x at very high speed.

1. Copy the example:
   ```powershell
   copy .env.example .env
   ```
2. Get a Groq API key:
   - Visit <https://console.groq.com/keys>
   - Sign in with Google or GitHub (free, no credit card).
   - Click *Create API Key*, copy the key (starts with `gsk_...`).
3. Open `.env` in Notepad (or VS Code) and fill in:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_paste_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   NCBI_EMAIL=gummybear112256@gmail.com
   ```

### Why Groq and Llama?

The brief asked for a free / least-cost option using Llama or another
strong open-source model. Groq hosts the open-weights Llama 3.x family
on its inference platform with a free developer tier. That gives us
the open-model story without anyone having to run a 70B model
locally. If you ever want to demo without any network access, set
`LLM_PROVIDER=ollama` and pull a smaller Llama model (`ollama pull
llama3.1`) - same code, no API key required.

### What happens if you skip this step?

The app still runs. With no API key it falls back to a keyword-only
PubMed search and prints a non-AI summary. Useful as a degraded demo;
not what you want to show off to a grader.

## 5. Run the app and try a question (1 minute)

```powershell
tasks.cmd run
```

(or `streamlit run app.py` if you set up manually)

A browser tab opens at `http://localhost:8501`. The home page now
shows four example-question chips - clicking one fills the input box.

To use the CLI instead:

```powershell
tasks.cmd cli "Does intermittent fasting improve insulin sensitivity?"
```

Or for JSON output:

```powershell
python -m src.cli --json "your question here"
```

To stop the Streamlit app: press `Ctrl+C` in the terminal window.

## 6. (Optional) NCBI rate-limit credentials (3 minutes)

Without credentials, NCBI caps you at 3 requests/second. With a free
API key the cap rises to 10/sec.

1. Visit <https://www.ncbi.nlm.nih.gov/account/settings/>.
2. Scroll to "API Key Management" and click *Create an API Key*.
3. Copy the key into `.env`:
   ```
   NCBI_API_KEY=your_long_hex_string
   ```

Save, restart Streamlit.

## 7. (Optional) Deploy to Streamlit Community Cloud (10 minutes)

For a publicly viewable demo to share with your teacher:

1. Create a free GitHub account if you don't have one.
2. Push this folder to a new (private or public) GitHub repo. If you
   restored the git bundle in step 8, you already have a history;
   just `git remote add origin <url>` and `git push -u origin main`.
3. Go to <https://share.streamlit.io>, sign in with GitHub.
4. Click *New app*, pick the repo, branch `main`, file `app.py`.
5. Under *Advanced settings -> Secrets*, paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "llama-3.3-70b-versatile"
   LLM_PROVIDER = "groq"
   NCBI_EMAIL = "gummybear112256@gmail.com"
   ```
6. Click *Deploy*. You get a public URL like
   `https://your-app.streamlit.app/`.

Streamlit Community Cloud is free for one app per account on a small
container. Plenty for a coursework demo.

## 8. (Optional) Restore the git history (2 minutes)

I committed each task as I built it, but the .git directory cannot
live inside your OneDrive folder from my sandbox. The full history is
delivered as `git_history.bundle` at the root of the project folder.

To restore it:

```powershell
# In the SE AI Agent folder
git init
git fetch .\git_history.bundle main:main
git checkout main
```

If you are not using git, you can ignore this entirely. The current
files in the folder are the final state of the project.

## 9. Sanity check before submitting (3 minutes)

```powershell
tasks.cmd check
tasks.cmd run
```

`tasks.cmd check` runs the full test suite plus an import smoke check.

Then try a question end-to-end in the UI. Check:

- [ ] You see the reformulated PubMed query in a code block.
- [ ] You see a grounded summary with `[PMID: ...]` citations.
- [ ] Each evidence card has values populated (not all "unknown").
- [ ] Evidence cards show a study-type badge (e.g. "RCT") and
      sometimes a verbatim supporting quote.
- [ ] The footer shows reviewed PMIDs and the disclaimer.
- [ ] The Download JSON and Download CSV buttons produce a file with
      the evidence in it.

If any of those fail, the most likely cause is the LLM provider.
Check `.env`, restart Streamlit, and re-try.

---

## Decisions I made for you (and how to change them)

| Decision                                | Why                                                                           | How to change                                                                |
| --------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Default LLM = Groq + Llama 3.3 70B      | Free tier, open-model story, very fast inference                              | Edit `.env`: switch `LLM_PROVIDER` to `ollama`, `openai`, or `none`         |
| Streamlit for the UI                    | Roadmap recommended it; fastest path to a polished demo                       | A React + FastAPI frontend can be added later without touching the modules |
| No database in V1                       | Roadmap explicitly defers persistence to V2                                   | V2 adds SQLite / Supabase; this is a deliberate scope decision              |
| `requests` instead of `biopython`       | One less heavy dependency; cleaner XML parsing in <250 lines                  | Swap `_efetch` for `Bio.Entrez.efetch` if you prefer Biopython              |
| Tests use a stubbed LLM and mocked HTTP | Tests run in <1s and never hit the network or burn quota                      | Real integration tests can be added in V2 with `pytest-vcr` or similar     |
| Per-task git commits                    | You asked for this in the kickoff message                                     | n/a                                                                          |

---

## If you get stuck

1. **`streamlit: command not found`** - Activate the virtual environment first, or use `tasks.cmd run`.
2. **`ModuleNotFoundError: No module named 'src'`** - Run from the project root folder, not from inside `src/`.
3. **`PubMedError: HTTP 429`** - You're rate-limited. Add `NCBI_API_KEY` to `.env`.
4. **Summary lacks PMID citations** - The grounding warning will tell you. It is an LLM quality issue; try a different question or switch to a larger Groq model in `.env`.
5. **App is slow** - First load is slow because Streamlit imports things lazily. Subsequent runs are fast.

Good luck. When you wake up the project is ready to demo.
