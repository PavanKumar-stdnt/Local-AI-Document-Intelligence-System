# RAG Chatbot — 100% Free, Fully Local

Ask questions about your own documents using **Gemma3:4b**, **ChromaDB**, and **Streamlit**.  
No API keys. No cloud. No cost.

---

## What it does

Upload any PDF, TXT, or Markdown file → the chatbot reads it, indexes it locally, and answers questions about it — with source citations showing exactly which document the answer came from.

```
User: "What are the leave policy rules?"
Bot:  "According to HR_Policy.pdf, employees are entitled to..."
      📚 Sources: HR_Policy.pdf
```

---

## Tech Stack (all free)

| Component       | Tool                        | Cost |
|----------------|-----------------------------|------|
| LLM             | Gemma3:4b via Ollama (local) | $0   |
| Embeddings      | nomic-embed-text via Ollama | $0   |
| Vector Database | ChromaDB (on disk)          | $0   |
| Web UI          | Streamlit                   | $0   |
| Evaluation      | RAGAS                       | $0   |

---

## Project Structure

```
rag-chatbot/
│
├── app.py                  ← Day 3: Streamlit web app (main entry point)
├── ingest.py               ← Day 1: CLI to load & index documents
├── cli_chat.py             ← Day 2: Terminal chatbot for testing
├── evaluate.py             ← Day 3: Run quality evaluation
├── requirements.txt        ← All Python dependencies
├── .env.example            ← Configuration template
├── .gitignore
│
├── src/                    ← Core pipeline modules
│   ├── __init__.py
│   ├── config.py           ← Central settings (reads from .env)
│   ├── logger.py           ← Logging setup
│   ├── document_loader.py  ← Load PDFs / TXTs from disk or upload
│   ├── text_splitter.py    ← Chunk documents with overlap
│   ├── vector_store.py     ← Embed + persist to ChromaDB
│   ├── retriever.py        ← MMR retriever from ChromaDB
│   ├── chain.py            ← ConversationalRetrievalChain + ask()
│   └── evaluator.py        ← Evaluation helpers + RAGAS
│
├── tests/                  ← Unit tests
│   ├── __init__.py
│   └── test_pipeline.py
│
├── scripts/                ← Helper scripts
│   ├── setup.sh            ← Linux/macOS one-shot setup
│   ├── setup_windows.bat   ← Windows one-shot setup
│   └── reset.sh            ← Wipe DB and re-index
│
└── docs/                   ← Put YOUR documents here (gitignored)
```

---

## Quickstart

### Step 1 — Install Ollama

Download from **https://ollama.com** and install for your OS.

Then pull the models (one-time, ~3.6 GB total):
```bash
ollama pull gemma3:4b         # the LLM (~3.3 GB) — fast, runs on 8 GB RAM
ollama pull nomic-embed-text  # the embedding model (~274 MB)
```

### Step 2 — Clone and set up

```bash
git clone <your-repo-url>
cd rag-chatbot

# Linux / macOS
chmod +x scripts/setup.sh
./scripts/setup.sh

# Windows — double-click scripts\setup_windows.bat
```

Or manually:
```bash
python -m venv rag-env
source rag-env/bin/activate        # Windows: rag-env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
mkdir docs
```

### Step 3 — Add your documents

```bash
cp your_policy.pdf docs/
cp your_notes.txt  docs/
# Any PDF, TXT, or MD file works
```

### Step 4 — Index the documents (Day 1)

```bash
python ingest.py
```

Output:
```
[INFO] STEP 1/3 — Loading documents…
[INFO] STEP 2/3 — Splitting into chunks…
[INFO] STEP 3/3 — Embedding and storing in ChromaDB…
[INFO] Ingestion complete!  Chunks stored: 142
```

### Step 5 — Test in the terminal (Day 2)

```bash
python cli_chat.py
```

```
You: What is the refund policy?
Bot: According to the document, refunds must be requested within 30 days...
     📚 Sources: policy.pdf
```

### Step 6 — Launch the web UI (Day 3)

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Day-by-Day Learning Guide

### Day 1 — Document Ingestion Pipeline
**Goal:** Populated vector database ready for querying.

1. `ingest.py` — CLI entry point
2. `src/document_loader.py` — load PDFs/TXTs
3. `src/text_splitter.py` — chunk with overlap
4. `src/vector_store.py` — embed + store in ChromaDB

**Key concept:** Chunk size matters. Smaller chunks (300–500 tokens) = precise retrieval but less context. Larger chunks (800–1000) = more context but noisier retrieval. Start with 500, tune based on evaluation scores.

### Day 2 — Retrieval Chain
**Goal:** Working CLI chatbot with memory.

1. `src/retriever.py` — MMR retriever
2. `src/chain.py` — prompt + LLM + memory → chain
3. `cli_chat.py` — interactive REPL for testing

**Key concept:** MMR (Max Marginal Relevance) fetches 8 candidates and keeps the 4 most diverse — avoiding returning 4 chunks that all say the same thing.

### Day 3 — Web UI + Evaluation
**Goal:** Polished web app with upload, citations, and quality metrics.

1. `app.py` — Streamlit chat UI with file uploader
2. `evaluate.py` — RAGAS evaluation runner
3. `src/evaluator.py` — faithfulness + answer relevancy metrics

**Key concept:** RAGAS faithfulness measures whether the answer is supported by the retrieved context (catches hallucinations). Target > 0.7 for production use.

---

## Configuration

Edit `.env` to customise behaviour:

```env
# Switch models depending on your machine's RAM
LLM_MODEL=gemma3:4b             # default — fast, ~3.3 GB, runs on 8 GB RAM
LLM_MODEL=gemma3:1b             # lightest — ~815 MB, very fast on low-end machines
LLM_MODEL=mistral               # alternative — good quality, ~4 GB
LLM_MODEL=llama3                # most capable — ~4.7 GB, needs 16 GB RAM

# Tune chunk size (lower = more precise, higher = more context)
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Retriever — how many chunks to fetch
RETRIEVER_K=4
RETRIEVER_FETCH_K=8
```

---

## Commands Reference

| Command | What it does |
|---------|-------------|
| `python ingest.py` | Index all files in ./docs |
| `python ingest.py --reset` | Wipe DB and re-index |
| `python ingest.py --docs path/` | Index a custom folder |
| `python cli_chat.py` | Terminal chatbot |
| `streamlit run app.py` | Web UI at localhost:8501 |
| `python evaluate.py` | Basic evaluation |
| `python evaluate.py --ragas` | Full RAGAS metrics |
| `pytest tests/ -v` | Run unit tests |
| `./scripts/reset.sh` | Wipe DB and re-index (Linux/macOS) |

---

## Troubleshooting

**`Connection refused` / Ollama errors**
```bash
# Make sure Ollama is running
ollama serve        # start the Ollama server
ollama list         # check which models are downloaded
```

**Slow responses**
```bash
# Switch to a lighter model in .env
LLM_MODEL=llama3.2:1b
```

**`No documents found`**
```bash
ls docs/    # make sure files are in the docs folder
# Supported: .pdf  .txt  .md
```

**Re-index after adding new files**
```bash
python ingest.py        # appends new files to existing DB
python ingest.py --reset  # full re-index from scratch
```

**`ModuleNotFoundError`**
```bash
source rag-env/bin/activate   # activate the virtual environment first
```

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| Python | 3.10 | 3.11 / 3.12 |
| OS | Linux / macOS / Windows | Linux / macOS |

> **Tip:** On a machine with less than 8 GB RAM, use `LLM_MODEL=llama3.2:1b` in your `.env` for faster responses.

---

## License

MIT — free to use, modify, and distribute.
