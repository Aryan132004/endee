# 🧠 Personal Knowledge Assistant

> Upload your PDFs, notes, and documents — then ask questions and get instant AI-powered answers.  
> Built on **[Endee](https://github.com/endee-io/endee)** vector database + **Gemini 2.0 Flash** + local semantic embeddings.

[![Endee](https://img.shields.io/badge/Vector_DB-Endee-blue)](https://github.com/endee-io/endee)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-orange)](https://aistudio.google.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

> 🍴 Built on top of [endee-io/endee](https://github.com/endee-io/endee)

---

## 📌 Problem Statement

Students, researchers, and professionals accumulate hundreds of PDFs, notes, and documents. Finding specific information means manually searching through everything — slow, frustrating, and inefficient.

**Personal Knowledge Assistant** solves this by letting you chat with your documents in plain English. Upload your files, ask questions, and get grounded answers with source citations — in seconds.

---

## 🎯 Features

- 📄 Upload **PDF, TXT, and Markdown** files
- 🔍 **Semantic search** — finds meaning, not just keywords
- 🤖 **AI-generated answers** grounded in your actual documents
- 📚 **Source citations** — see exactly which chunk answered your question
- 🖥️ **Web UI** via Streamlit + **CLI** for terminal users
- 🆓 **Completely free** — local embeddings + free Gemini API

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│                                                             │
│  PDF / TXT / MD  →  Text Chunks  →  Embeddings  →  Endee   │
│  (pdfplumber)       (500 words)    (MiniLM-L6)   (cosine)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         QUERYING                            │
│                                                             │
│  User Question  →  Embed  →  Endee Search  →  Top-K Chunks │
│                                    ↓                        │
│                             Gemini LLM  →  Grounded Answer  │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Steps

| Step | What Happens |
|------|-------------|
| **1. Parse** | Extract raw text from PDF/TXT/MD files |
| **2. Chunk** | Split text into overlapping 500-word chunks |
| **3. Embed** | Convert each chunk to a 384-dim vector using `all-MiniLM-L6-v2` (runs locally) |
| **4. Store** | Upsert vectors + metadata into Endee index (cosine similarity) |
| **5. Query** | Embed user question → search Endee → retrieve top-K similar chunks |
| **6. Answer** | Send chunks + question to Gemini → receive grounded answer |

---

## 🔧 How Endee Is Used

[Endee](https://github.com/endee-io/endee) is the **core vector database** that powers semantic retrieval in this project.

```python
from endee import Endee

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create a cosine similarity index
client.create_index(
    name="knowledge_base",
    dimension=384,           # matches all-MiniLM-L6-v2
    space_type="cosine",
    precision="float32"
)

index = client.get_index(name="knowledge_base")

# Store document chunks as vectors
index.upsert([{
    "id": "notes.pdf::chunk0",
    "vector": [0.1, 0.2, ...],   # 384-dim embedding
    "meta": {
        "text":   "Gradient descent minimizes the loss function...",
        "source": "notes.pdf",
        "chunk":  0
    }
}])

# Semantic search at query time
results = index.query(vector=[...], top_k=5)
```

Endee's HNSW indexing enables fast approximate nearest-neighbour search across thousands of document chunks, returning the most semantically relevant passages in milliseconds.

---

## 📁 Project Structure

```
personal-knowledge-assistant/
│
├── utils/
│   ├── __init__.py          # Package init
│   ├── embedder.py          # Local sentence-transformer embeddings (free, no API)
│   ├── pdf_parser.py        # PDF/TXT/MD parser + overlapping text chunker
│   ├── vector_store.py      # Endee wrapper — create index, upsert, search
│   └── llm.py               # Gemini / Groq / OpenAI LLM caller
│
├── config.py                # Loads all settings from .env
├── ingest.py                # CLI: parse, embed, upload documents → Endee
├── ask.py                   # CLI: ask questions (one-shot & interactive)
├── app.py                   # Streamlit web UI
│
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template (safe to commit)
├── .env                     # Your actual secrets — NEVER commit this
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Docker (to run Endee)
- Free Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikeys)

---

### Step 1 — Star & Fork Endee ⭐ (Mandatory)

1. Go to → https://github.com/endee-io/endee
2. Click ⭐ **Star**
3. Click 🍴 **Fork** → select your account → **Create fork**
4. Clone your fork:
```bash
git clone https://github.com/<your-username>/endee.git
cd endee
```

---

### Step 2 — Start Endee Vector Database

```bash
# In the root endee/ folder, Endee is already configured via docker-compose.yml
docker compose up -d
```

Verify it works → open **http://localhost:8080** in your browser.

---

### Step 3 — Set Up the Python Project

```bash
cd personal-knowledge-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your real values:

```env
# Get free key at https://aistudio.google.com/app/apikeys
GEMINI_API_KEY=AIzaSy_xxxxxxxxxxxxxxxxxxxx
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash

ENDEE_BASE_URL=http://localhost:8080/api/v1
ENDEE_AUTH_TOKEN=

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
INDEX_NAME=knowledge_base
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

### Step 5 — Ingest Documents

```bash
# Single file
python ingest.py your_notes.pdf

# Multiple files
python ingest.py lecture1.pdf lecture2.txt notes.md

# Supported formats: .pdf  .txt  .md
```

Expected output:
```
📄 Ingesting: your_notes.pdf
   ✔ Split into 42 chunks
   ✔ Embeddings generated (42 vectors)
   ✔ Stored 42 chunks in Endee
```

---

### Step 6 — Ask Questions

**Web UI (recommended for demo):**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**CLI — one-shot:**
```bash
python ask.py "What is gradient descent?"
python ask.py "Summarise the key points about neural networks"
```

**CLI — interactive mode:**
```bash
python ask.py
# Type questions, press Enter, type 'exit' to quit
```

---

## 💡 Example Use Cases

| Document Type | Example Question |
|---|---|
| 📚 Semester notes | "Explain dynamic programming with an example" |
| 📄 Research papers | "What methodology did the authors use?" |
| 📋 Technical docs | "How do I configure authentication?" |
| 🏥 Medical documents | "What are the side effects mentioned?" |
| 📰 Any PDF | "Summarise the main points of this document" |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Vector Database** | Endee (HNSW, cosine) | Fast semantic search at scale |
| **Embeddings** | `all-MiniLM-L6-v2` | Local, free, 384-dim, high quality |
| **LLM** | Gemini 2.0 Flash | Fast, free tier, strong reasoning |
| **PDF Parser** | pdfplumber | Reliable text extraction |
| **Web UI** | Streamlit | Quick, clean interface |
| **CLI** | Rich | Beautiful terminal output |

---

## 🤔 Why Vector Search?

Traditional keyword search finds exact word matches. Vector search finds **meaning**.

For example, asking *"How does the brain learn?"* can retrieve a chunk about *"synaptic plasticity and Hebbian learning"* — because Endee understands that these concepts are semantically related in the 384-dimensional embedding space, even though no words overlap.

This is why Endee is the most critical component of this RAG system.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Your Gemini API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `ENDEE_BASE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DIM` | `384` | Vector dimensions |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping words between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |

---

## 🔗 Links

| Resource | URL |
|---|---|
| 📦 This Project | https://github.com/\<your-username\>/endee/tree/master/personal-knowledge-assistant |
| 🍴 Forked Endee Repo | https://github.com/\<your-username\>/endee |
| ⭐ Original Endee Repo | https://github.com/endee-io/endee |
| 📖 Endee Docs | https://docs.endee.io |
| 🤖 Gemini API Keys | https://aistudio.google.com/app/apikeys |

---

## 📄 License

MIT — free to use, modify, and distribute.
