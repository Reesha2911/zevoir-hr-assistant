# Zevoir HR Assistant — RAG Chatbot for Email & WhatsApp Assistance

A retrieval-augmented chatbot that answers HR policy questions and drafts emails and WhatsApp messages, using only the documents in `knowledge_base/`. It runs fully locally with Ollama.

> **The five policy documents in `knowledge_base/` are fictional sample data written for this project. They are not real company policies.**

Same stack as Zevoir Genie: **FAISS + Ollama (`nomic-embed-text` + `llama3`) + FastAPI**.

## Setup

```bash
cd zevoir_hr_rag
pip install -r requirements.txt

# Make sure Ollama is running and models are pulled:
ollama pull nomic-embed-text
ollama pull llama3

# Build the vector index from the knowledge base
python ingest.py

# Run the backend
uvicorn app:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Project Structure

```
zevoir_hr_rag/
├── knowledge_base/
│   ├── leave_policy.md
│   ├── wfh_policy.md
│   ├── holiday_calendar.md
│   ├── hr_guidelines.md
│   └── employee_handbook.md
├── ingest.py              # Milestone 1: chunk + embed + FAISS index
├── app.py                 # Milestone 2 & 3: FastAPI backend, modes, guardrails
├── static/index.html      # Chat UI
├── requirements.txt
└── test_questions.txt     # Milestone 3: test cases
```

## Milestones

**Milestone 1 — RAG Core (`ingest.py`)**
Reads the 5 knowledge base docs, splits them into ~400-character chunks, embeds each chunk with `nomic-embed-text` via Ollama, stores vectors in a FAISS `IndexFlatIP` index (inner product on unit-length vectors, which equals cosine similarity) alongside chunk text + source metadata (`vector_store.index`, `vector_store_meta.json`).

**Milestone 2 — Backend + Response Modes (`app.py`)**
A single `/query` endpoint:
1. Embeds the user's message and retrieves the top-6 most similar chunks.
2. Runs `detect_mode()` — a lightweight keyword classifier — to decide whether the user wants a **Q&A answer**, an **email**, or a **WhatsApp message**.
3. Builds a mode-specific system prompt (different formatting rules per mode) and calls `llama3` with the retrieved chunks as the only allowed context.

**Milestone 3 — Guardrails & Testing**
- **Similarity threshold** (`SIM_THRESHOLD = 0.46` in `app.py`, cosine similarity — higher means more similar): if the closest retrieved chunk scores below this, the bot refuses before ever calling the LLM. This is the main hallucination-prevention layer, same approach as Zevoir Genie's `SIM_THRESHOLD`.
- **System prompt hard rule**: the LLM is instructed to output the exact not-found sentence if the context doesn't cover the request, rather than guessing.
- **Post-check**: if the LLM's own reply contains the not-found phrase, sources are stripped from the response (avoids citing a source for an answer it didn't actually give).
- `test_questions.txt` has 22 cases spanning in-scope Q&A, out-of-scope Q&A, email generation, and WhatsApp generation — including one deliberately "close but not covered" case (workation from Bali) to test that the bot doesn't over-generalize the WFH policy.

## Running the test questions

With Ollama running and the server started (see Setup), in a second terminal:

```bash
python run_tests.py
```

It sends every question in `test_questions.txt` to `/query` and prints the reply. For the out-of-scope questions it reports PASS when the bot refuses and FAIL when it answers. The in-scope, email and WhatsApp replies need reading by a human. Model answers vary, so treat this as a smoke test and not a guarantee.

## Known limitations
- Retrieval uses 400-character chunks and one similarity threshold, so edge cases can be refused or answered wrongly. Check results after any change to the documents or threshold.
- Mode detection is keyword-based, so unusual wording can pick the wrong mode.
- There is no login, no conversation memory and no rate limiting. It is built for local use, not for exposure to the internet.
- Answer quality depends on the local `llama3` model.

## Tuning notes
`SIM_THRESHOLD` (in `app.py`) is a cosine similarity cutoff — higher means the bot demands a closer match before answering, lower means it's more lenient. It's currently set to `0.46`, matching the Zevoir Genie project. If you get false refusals on clearly in-scope questions, try lowering it slightly (e.g. `0.35`); if it starts answering off-topic questions, raise it. Run a mix of in-scope and out-of-scope questions from `test_questions.txt` after any change.

## Extending the knowledge base
Drop any additional `.md`/`.txt` policy docs into `knowledge_base/` and re-run `python ingest.py` — no code changes needed.
