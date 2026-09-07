*[Leer esto en español](README.es.md)*

# RAG over Colombian and Latin American History

A FastAPI application that answers natural-language questions over a corpus of history books (primarily Colombian and Latin American), using an LLM to generate answers grounded exclusively in the actual content of those books — with guaranteed source citation and explicit handling of cases where the system lacks sufficient information to answer.

Portfolio project built to demonstrate **AI Engineering** skills: integrating LLMs via API, designing RAG (Retrieval-Augmented Generation) pipelines, working with vector databases, and making engineering decisions that were validated with real data rather than copied from a tutorial.

> **Current status:** corpus loading complete — 48 of 50 target books indexed. See [Corpus status](#corpus-status) for details.

---

## Why this project

As a recently graduated systems engineer with a backend development background (Laravel, .NET), I wanted a project that builds on that foundation — building software, not training models from scratch — while demonstrating the specific skill set the current **AI Engineer** market actually asks for: integrating existing language models into real applications via APIs, vector databases, and retrieval architectures.

I chose a history corpus (rather than, say, generic technical documentation) because it forces real data-quality problems to be solved: books hundreds of pages long, PDFs with problematic encodings, corrupted files, and multilingual content — conditions much closer to a real-world use case than an already-clean toy dataset.

---

## Architecture

The system is split into two independent programs with separate responsibilities:

```
┌─────────────────┐         ┌──────────────────────┐
│   ingesta.py       │        │      main.py            │
│  (offline, run       │       │   (production API)       │
│  when books are      │       │                          │
│  added)               │       │                          │
├─────────────────┤         ├──────────────────────┤
│ 1. Reads PDFs         │       │ 1. Receives user's        │
│ 2. Extracts text       │       │    question (POST)        │
│ 3. Splits into chunks   │       │ 2. Embeds the question     │
│ 4. Generates embeddings │       │ 3. Searches for relevant   │
│ 5. Stores in Chroma      │──────▶│    chunks in Chroma        │
└─────────────────┘         │ 4. Filters by relevance      │
                              │    threshold                  │
                              │ 5. Builds prompt with          │
                              │    context + question           │
                              │ 6. Calls the LLM                  │
                              │ 7. Returns answer +                │
                              │    cited sources                    │
                              └──────────────────────┘
```

This separation isn't arbitrary: ingestion is a heavy, slow, quota-limited process run sporadically. The query API needs to be fast and always available. Mixing both responsibilities into one program would have unnecessarily coupled their lifecycles.

---

## Tech stack

| Component | Technology | Why |
|---|---|---|
| Backend / API | **FastAPI** | Native async, automatic validation via Pydantic, self-generated interactive docs. The de facto standard in the AI-application ecosystem — deliberately chosen over Django, which brings unnecessary overhead (heavy ORM, admin panel) for a pure API. |
| Language | **Python 3.12** | Used 3.12 instead of a newer version (3.14) installed in parallel, due to better ecosystem compatibility maturity for AI/ML libraries against freshly released Python versions. |
| LLM (generation) | **Google Gemini API** (`gemini-flash-lite-latest`) | Genuinely free tier, no credit card required, with a daily quota sufficient for development and demos (unlike Anthropic/OpenAI, which only offer a one-time trial credit). The `-latest` alias is used instead of pinning an exact version, to avoid depending on a model name that could be deprecated without notice. |
| Embeddings | **Google Gemini API** (`gemini-embedding-001`) | With explicit `task_type` differentiation (`RETRIEVAL_DOCUMENT` for the corpus, `RETRIEVAL_QUERY` for user questions) — an asymmetric optimization that improves semantic search precision. |
| Vector database | **ChromaDB** (persistent, local) | No infrastructure cost, appropriate for the project's scale (dozens of books). No external service required. |
| PDF extraction | **pypdf** | Pure Python library, no compiled dependencies, permissive license (BSD) — relevant for a public GitHub portfolio project, unlike faster alternatives that carry an AGPL license. |

---

## Key technical decisions (and why they matter)

**1. Conversation memory and RAG are independent mechanisms.**
Conversation history (what the user and model have said to each other) is kept separate from the context retrieved via semantic search. Retrieved RAG context is injected only into that turn's specific call — it is never permanently stored in the history. This prevents the conversation from accumulating irrelevant context from previous questions, which would otherwise make every subsequent turn slower and more expensive for no benefit.

**2. The relevance threshold was calibrated empirically, not guessed.**
Rather than fixing an arbitrary vector-distance cutoff to decide which chunks count as "relevant enough," the system was tested against a varied set of questions (clearly relevant, tangential, and completely unrelated to the corpus), and the actual distances returned by ChromaDB were measured. The threshold was tuned against that data, and **is documented as a parameter that requires recalibration as the corpus grows** — with more books and more topical diversity, the optimal cutoff shifts, which was validated in practice during development (moved from 0.7 to 0.65 after expanding the corpus from 6 to 15 books).

**3. Explicit API quota handling with retries and granular resumability.**
The free tier of the embeddings API imposes per-minute and per-day limits. The ingestion pipeline implements retries with progressive backoff on 429 errors, and — more importantly — **resumability at the individual chunk level, not just the book level**: if the process is cut off mid-book due to daily quota exhaustion, the next day it resumes exactly at the chunk where it left off, instead of reprocessing the whole book and burning through the quota again without making progress.

**4. Explicit grounding against hallucinations.**
The prompt directly instructs the model to state when the context doesn't contain the answer, rather than inventing information. Additionally, the `sources` field in the response is computed deterministically from ChromaDB's metadata — it does not rely on the model correctly mentioning its sources in free text.

**5. Differentiated error handling.**
Failures in vector search and failures in LLM generation are caught separately, returning appropriate HTTP status codes (503) with clear messages, instead of a generic 500. The ChromaDB collection is fetched lazily (inside the function, not at module load), so the server can start even if ingestion hasn't been run yet.

**6. Resilience against faulty input data.**
Ingestion isolates each PDF's reading step in its own error handling: a corrupted file (a real case was found — a truncated PDF, unrecoverable even with specialized tools like `qpdf`) is logged and skipped, without halting processing of the rest of the corpus.

---

## Corpus status

**48 books loaded** (2 out of an initial target of 50 were discarded due to file corruption — one PDF was truncated beyond repair, even with specialized tools like `qpdf`; this is logged automatically by the ingestion script rather than silently skipped). Corpus covers Colombian and Latin American history, plus several general world history references.

**Known limitation:** some books include summaries or excerpts in other languages (e.g. Catalan, in academic documents with multilingual abstracts). Gemini's embeddings are multilingual and handle this reasonably well, but no language filtering was implemented — this is documented as a future improvement, not a hidden defect.

---

## Local installation and usage

### Prerequisites
- Python 3.12
- A free API key from [Google AI Studio](https://aistudio.google.com/)

### Setup

```powershell
# Clone the repository
git clone <repo-url>
cd mi-proyecto-rag

# Create and activate a virtual environment
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file with:
# GEMINI_API_KEY=your_key_here
```

### Load the corpus

Place PDFs in the `libros/` folder and run:

```powershell
python ingesta.py
```

The script is safe to interrupt and resume — it respects the free daily API quota and automatically picks up where it left off.

### Run the API

```powershell
uvicorn app.main:app --reload
```

Interactive docs available at `http://127.0.0.1:8000/docs`.

### Example usage

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué fue el M-19?"}'
```

```json
{
  "respuesta": "El M-19 fue un movimiento guerrillero colombiano...",
  "fuentes": ["carlos-pizarro-leongomez-de-guerrillero-a-candidato-presidencial-varios-autores.pdf"]
}
```

*(Request/response fields are in Spanish, matching the corpus language.)*

---

## Roadmap / future improvements

- [ ] Recalibrate the relevance threshold periodically if new books are added
- [ ] Per-chunk language detection/filtering
- [ ] Minimal frontend (decision pending)
- [ ] Per-user conversation sessions (history is currently global, appropriate for a single-user demo)

---

## Author

Built by Carlos Escobar as part of a portfolio for AI Engineer positions.