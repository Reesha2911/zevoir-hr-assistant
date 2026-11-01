"""
Milestone 2 & 3 — FastAPI Backend, Response Modes, and Guardrails
-------------------------------------------------------------------
Serves a single /query endpoint that:
  1. Embeds the incoming user message and retrieves the top-k most
     similar chunks from the FAISS index built by ingest.py.
  2. Detects which RESPONSE MODE is being requested: plain Q&A,
     EMAIL generation, or WHATSAPP message generation.
  3. Calls llama3 (via Ollama) with a strict system prompt that
     forces answers to be grounded ONLY in the retrieved chunks.
  4. Applies a similarity threshold guardrail — if nothing relevant
     enough is retrieved, refuses instead of guessing
     ("I couldn't find this information in the uploaded documents.")

Run with:
    uvicorn app:app --reload --port 8000
"""

import json
import re
import requests
import numpy as np
import faiss
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

INDEX_PATH = "vector_store.index"
META_PATH = "vector_store_meta.json"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"

TOP_K = 6
# Cosine similarity threshold (higher = more similar, range roughly -1 to 1).
# Chunks below this are considered "not relevant enough" -> refuse.
SIM_THRESHOLD = 0.46

NOT_FOUND_MSG = "I couldn't find this information in the uploaded documents."

app = FastAPI(title="Zevoir HR Assistant (RAG)")

# ---- Load index & metadata once at startup ----
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    store = json.load(f)
CHUNKS = store["chunks"]
META = store["meta"]


class QueryRequest(BaseModel):
    message: str


def embed_text(text: str):
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def retrieve(query: str, k: int = TOP_K):
    """Return top-k (chunk_text, source, cosine_similarity) tuples for a query."""
    vec = np.array([embed_text(query)], dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    scores, indices = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((CHUNKS[idx], META[idx]["source"], float(score)))
    return results


def detect_mode(message: str) -> str:
    """Classify the request as 'email', 'whatsapp', or 'qa'."""
    lower = message.lower()
    whatsapp_signals = ["whatsapp", "wa message", "text message", "send a message to"]
    email_signals = ["email", "e-mail", "mail to", "write to my manager", "draft a letter"]

    if any(sig in lower for sig in whatsapp_signals):
        return "whatsapp"
    if any(sig in lower for sig in email_signals):
        return "email"
    return "qa"


def build_system_prompt(mode: str) -> str:
    base_rules = (
        "You are Zevoir Technologies' internal HR assistant. "
        "You must answer ONLY using the CONTEXT provided below, which comes from "
        "the company's official policy documents. "
        "Never invent, assume, or infer any policy detail, number, or rule that is not "
        "explicitly present in the CONTEXT. "
        "Before answering or drafting anything, check whether the CONTEXT contains any "
        "exclusion, restriction, or exception that applies to the specific scenario the user "
        "described (for example: location limits, eligibility limits, caps, or requirements for "
        "separate approval). If such an exclusion applies, you must not treat the request as "
        "routine or pre-approved — call out the restriction explicitly instead. "
        "If the CONTEXT does not contain the information needed to answer or complete "
        "the request, respond with EXACTLY this sentence and nothing else: "
        f"\"{NOT_FOUND_MSG}\"\n\n"
    )

    if mode == "email":
        return base_rules + (
            "TASK: Write a professional email based on the user's request and the CONTEXT. "
            "Structure it with a short subject line (prefixed 'Subject:'), a greeting, "
            "2-4 concise body sentences that reference the correct policy details "
            "(e.g. leave type, entitlement, notice period) from the CONTEXT, and a polite sign-off. "
            "Do not state a specific number of leave days or any policy figure unless it appears "
            "in the CONTEXT.\n\n"
            "IMPORTANT — keep the email natural and in the employee's own voice:\n"
            "- Write the email the way a person would actually write it. Never quote policy text "
            "verbatim or in quotation marks inside the email, and never write phrases like 'as per "
            "the WFH policy' or 'according to the policy' inside the email body — just state the "
            "request plainly and naturally, the way a person would.\n"
            "- The email is addressed TO the manager, so never refer to 'manager approval' as if the "
            "manager were a third party — just ask for their approval directly ('please let me know "
            "if this works for you' / 'could you approve this').\n"
            "- If the CONTEXT indicates the request falls outside standard policy (needs separate HR "
            "approval, isn't covered by the standard allowance, is excluded, etc.), do NOT put that "
            "explanation inside the email itself. Instead, write a short, natural, human email that "
            "simply makes the request, and then AFTER the full email, add a line that starts exactly "
            "with 'Note (not part of the email):' followed by a one-sentence heads-up to the person "
            "asking me to draft this, explaining the restriction in plain language.\n"
            "- If a required policy detail is missing from the CONTEXT entirely, say so with the "
            "standard not-found sentence instead of guessing."
        )
    elif mode == "whatsapp":
        return base_rules + (
            "TASK: Write a short, casual but professional WhatsApp message (3-4 sentences max) "
            "based on the user's request and the CONTEXT. Keep it brief, friendly, and to the point.\n\n"
            "IMPORTANT — keep the message natural:\n"
            "- Write it the way a person actually texts their manager. Never quote policy text "
            "verbatim or say 'as per the policy' inside the message itself.\n"
            "- If the CONTEXT indicates the request is excluded from standard policy or needs separate "
            "approval, do NOT put that explanation inside the WhatsApp message. Instead, write the "
            "natural message, then AFTER it, add a line starting exactly with 'Note (not part of the "
            "message):' followed by a one-sentence heads-up to the person asking me to draft this."
        )
    else:
        return base_rules + (
            "TASK: Answer the user's question directly and concisely (1-3 sentences) using only "
            "the CONTEXT. Do not add caveats, disclaimers, or mention 'the documents' unless refusing."
        )


def call_llm(system_prompt: str, context: str, user_message: str) -> str:
    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER REQUEST:\n{user_message}\n\n"
        f"RESPONSE:"
    )
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _friendly_source_name(filename: str) -> str:
    """Turn 'wfh_policy.md' into 'WFH Policy' for display in the widget UI."""
    name = filename.rsplit(".", 1)[0].replace("_", " ")
    return name.title().replace("Wfh", "WFH").replace("Hr", "HR")


@app.post("/query")
def query(req: QueryRequest):
    message = (req.message or "").strip()
    if not message:
        return {"reply": "Looks like you forgot to type something! Please ask me about leave, WFH, or holidays.", "mode": "qa", "sources": []}

    mode = detect_mode(message)
    results = retrieve(message, k=TOP_K)

    # Guardrail: nothing relevant enough retrieved -> refuse immediately
    if not results or results[0][2] < SIM_THRESHOLD:
        return {"reply": NOT_FOUND_MSG, "mode": mode, "sources": []}

    # Keep only chunks that pass the threshold (avoid diluting context with junk)
    relevant = [r for r in results if r[2] >= SIM_THRESHOLD]
    context = "\n---\n".join(f"[{src}] {chunk}" for chunk, src, _ in relevant)
    sources = sorted(set(_friendly_source_name(src) for _, src, _ in relevant))

    system_prompt = build_system_prompt(mode)
    reply = call_llm(system_prompt, context, message)

    # Guardrail: if the model claims not-found, strip source citations
    if NOT_FOUND_MSG.lower() in reply.lower():
        return {"reply": NOT_FOUND_MSG, "mode": mode, "sources": []}

    return {"reply": reply, "mode": mode, "sources": sources}


@app.post("/chat")
def chat_alias(req: QueryRequest):
    return query(req)


@app.get("/health")
def health():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return {"status": "ok" if ollama_ok else "degraded", "ollama": ollama_ok, "n_chunks": index.ntotal}


# ---- Serve the simple chat frontend ----
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")