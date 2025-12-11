"""
app.py

完整 RAG 服务：
- 读取 ./docs/* (txt, md, pdf)
- 切分成 chunk
- 用 sentence-transformers 生成向量并写入 chromadb（持久化目录 ./chroma_db）
- 查询时：向量检索 -> LLM-based rerank (OpenAI) -> 用 top_n chunk 作为上下文调用 OpenAI 生成答案（并返回 Sources）
- 提供 FastAPI endpoints: /query, /reset_session, /health
- 会话记忆：内存 + 持久化到 sessions.json
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import aiofiles
import time

# ========== CONFIG ==========
DOCS_DIR = "./docs"
CHROMA_DB_DIR = "./chroma_db"
SESSIONS_FILE = "./sessions.json"   # persist minimal session memory across restarts
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # for the model above
TOP_K = 10        # initial vector retrieval k
RERANK_TOP_N = 3  # LLM-based rerank pick top n to feed final LLM
MEMORY_MAX_TURNS = 6  # number of recent user/assistant turns to include
OPENAI_MODEL_FOR_RERANK = "gpt-4o-mini"  # used for scoring and for final answer — change as needed
OPENAI_TEMPERATURE = 0.0
CHUNK_SIZE = 500   # approx characters per chunk (adjust)
CHUNK_OVERLAP = 50

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in .env or environment")

openai.api_key = OPENAI_API_KEY

# ========== UTIL ==========
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text_parts = []
    reader = PdfReader(str(path))
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        text_parts.append(t)
    return "\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c.strip()]

# ========== DOCUMENT LOADER & INDEXER ==========
class DocChunk:
    def __init__(self, id: str, text: str, source: str, chunk_idx: int):
        self.id = id
        self.text = text
        self.source = source
        self.chunk_idx = chunk_idx

def load_and_index_docs(force_reindex: bool = False) -> None:
    """
    Walk docs dir, chunk files, embed chunks, upsert into chroma collection.
    If collection already exists and not force_reindex, skip reindexing.
    """
    print("Loading embedder and chroma client...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection_name = "rag_docs"
    try:
        collection = client.get_collection(name=collection_name)
        exists = True
    except Exception:
        collection = client.create_collection(name=collection_name)
        exists = False

    # If collection exists and not forced, skip to avoid reindexing each restart
    if exists and not force_reindex:
        print("Chroma collection already exists. Skipping indexing (set force_reindex=True to rebuild).")
        return

    print("Building chunks from files in", DOCS_DIR)
    chunks: List[DocChunk] = []
    p = Path(DOCS_DIR)
    if not p.exists():
        raise RuntimeError(f"{DOCS_DIR} not found — create it and add documents (txt/md/pdf).")

    files = sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".txt", ".md", ".pdf"}])
    if not files:
        print("No files found in docs dir.")
    for f in files:
        try:
            if f.suffix.lower() == ".pdf":
                raw = read_pdf(f)
            else:
                raw = read_txt_md(f)
        except Exception as e:
            print("Failed read", f, e)
            continue
        if not raw.strip():
            continue
        cs = chunk_text(raw, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, c in enumerate(cs):
            chunk_id = sha1(str(f) + "_" + str(i))
            chunks.append(DocChunk(id=chunk_id, text=c, source=str(f), chunk_idx=i))

    # prepare embeddings (batch)
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL_NAME} ...")
    texts = [c.text for c in chunks]
    # sentence-transformers returns numpy arrays
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # create or reset collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    # upsert into chroma
    metadatas = [{"source": c.source, "chunk_idx": c.chunk_idx, "text_snippet": c.text[:400]} for c in chunks]
    ids = [c.id for c in chunks]
    print("Upserting to chroma ...")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    print("Indexing complete. persisted to", CHROMA_DB_DIR)

# ========== QUERY / RERANK / ANSWER LOGIC ==========
def vector_retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Returns list of candidate dicts:
    { 'id', 'text', 'source', 'chunk_idx', 'distance' }
    """
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    coll = client.get_collection("rag_docs")
    results = coll.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "distances", "documents", "ids"]
    )
    # result structure: dict with keys documents, ids, metadatas, distances (each is list of lists)
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source"),
            "chunk_idx": results["metadatas"][0][i].get("chunk_idx"),
            "distance": results["distances"][0][i]
        })
    return docs

def llm_rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = RERANK_TOP_N) -> List[Dict[str, Any]]:
    """
    Uses OpenAI LLM to score each candidate's relevance to the query.
    Returns top_n candidates sorted by score desc, with added 'score' field (0-100).
    """
    # Build a compact prompt that asks the model to return JSON list with numeric scores.
    system = (
        "You are a helpful relevance evaluator. "
        "Given a user query and multiple short document passages, assign each passage an integer relevance score from 0 to 100 "
        "where 100 means highly relevant and 0 means not relevant. "
        "Return a JSON array of objects with keys: id and score (integer). "
        "Do NOT add extra commentary."
    )

    # Compose the user content with query and candidate snippets (limit length!)
    pieces = []
    for c in candidates:
        # keep snippet short to avoid very long prompts
        snippet = c["text"]
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        pieces.append(f"ID: {c['id']}\nSource: {c['source']}\nSnippet: {snippet}\n---")

    user_content = (
        f"Query: {query}\n\nCandidates:\n" + "\n".join(pieces) +
        "\n\nReturn JSON like: [{'{'}\"id\":\"...\",\"score\":int{'}'} , ...]"
    )

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL_FOR_RERANK,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=512
        )
        txt = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Rerank call failed:", e)
        # fallback: simple heuristic by chroma distance (if present)
        for i, c in enumerate(candidates):
            c["score"] = int(100 - i * 10)
        return candidates[:top_n]

    # try to extract JSON from model output
    json_text = None
    # naive JSON extraction: find first '[' and last ']'
    try:
        start = txt.index('[')
        end = txt.rindex(']') + 1
        json_text = txt[start:end]
        parsed = json.loads(json_text)
        # map scores
        id2score = {item["id"]: int(item["score"]) for item in parsed if "id" in item and "score" in item}
        for c in candidates:
            c["score"] = id2score.get(c["id"], 0)
    except Exception as e:
        print("Failed parse rerank JSON, fallback. err:", e)
        # If parsing fails, fallback to using simple heuristic
        for i, c in enumerate(candidates):
            c["score"] = int(100 - i * 5)

    sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_cands[:top_n]

def build_final_prompt(query: str, selected_chunks: List[Dict[str, Any]], conversation_history: List[Dict[str,str]] = None) -> str:
    """
    Build the final prompt for the answer-generation LLM.
    We include:
    - Instruction to only use provided chunks
    - Conversation history (if any) as context
    - The selected chunks with explicit source markers
    - The user query
    """
    system_instr = (
        "You are an assistant answering the user's question using ONLY the provided document excerpts. "
        "Cite sources at the end using the format: [source: path/to/file, chunk: i]. "
        "If the answer is not contained in the provided excerpts, say 'I don't know based on provided documents.'"
    )

    history_text = ""
    if conversation_history:
        # conversation_history is list of {"user": "...", "assistant": "..."}
        history_text = "Conversation history (most recent first):\n"
        hparts = []
        # include up to MEMORY_MAX_TURNS turns (already limited when saved)
        for turn in conversation_history[-MEMORY_MAX_TURNS:]:
            u = turn.get("user", "").strip()
            a = turn.get("assistant", "").strip()
            if u:
                hparts.append(f"User: {u}")
            if a:
                hparts.append(f"Assistant: {a}")
        history_text += "\n".join(hparts) + "\n\n"

    chunks_text = []
    for c in selected_chunks:
        chunks_text.append(f"[SOURCE: {c['source']} | CHUNK: {c['chunk_idx']}]\n{c['text']}\n---")
    chunks_block = "\n".join(chunks_text)

    prompt = (
        system_instr + "\n\n" +
        history_text +
        "Document excerpts:\n" + chunks_block + "\n\n" +
        "User question: " + query + "\n\n" +
        "Answer now, and include Sources at the end."
    )
    return prompt

def generate_answer_with_llm(prompt: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL_FOR_RERANK,
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=800
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenAI generation failed:", e)
        raise

# ========== SESSIONS MEMORY (simple file-backed store) ==========
_sessions: Dict[str, List[Dict[str,str]]] = {}  # session_id -> list of turns {"user":..., "assistant":...}

def load_sessions():
    global _sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                _sessions = json.load(f)
        except:
            _sessions = {}
    else:
        _sessions = {}

def persist_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(_sessions, f, ensure_ascii=False, indent=2)

def add_turn(session_id: str, user: str, assistant: str):
    s = _sessions.get(session_id, [])
    s.append({"user": user, "assistant": assistant})
    # keep only recent N turns
    if len(s) > MEMORY_MAX_TURNS * 2:
        s = s[-MEMORY_MAX_TURNS*2:]
    _sessions[session_id] = s
    persist_sessions()

def reset_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        persist_sessions()

# ========== FASTAPI APP ==========
app = FastAPI(title="RAG Service with LLM Rerank & Memory")

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    rerank_scores: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    # ensure docs indexed
    load_sessions()
    # Only index if chroma DB not exist or empty
    if not os.path.exists(CHROMA_DB_DIR) or not any(Path(CHROMA_DB_DIR).iterdir()):
        print("Indexing docs on startup...")
        load_and_index_docs(force_reindex=True)
    else:
        print("Chroma DB exists — skipping reindex (use force_reindex True to rebuild).")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    session_id = req.session_id
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # 1) vector retrieve
    candidates = vector_retrieve(question, top_k=TOP_K)

    # 2) LLM-based rerank (scores)
    reranked = llm_rerank(question, candidates, top_n=RERANK_TOP_N)

    # 3) Build final prompt (include recent memory)
    history = _sessions.get(session_id, [])
    prompt = build_final_prompt(question, reranked, conversation_history=history)

    # 4) Generate answer
    answer = generate_answer_with_llm(prompt)

    # 5) Save memory (append turn)
    add_turn(session_id, question, answer)

    # 6) return answer + sources
    sources = [{"source": c["source"], "chunk_idx": c["chunk_idx"], "id": c["id"], "score": c.get("score", None)} for c in reranked]
    rerank_scores = [{"id": c["id"], "score": c.get("score", None)} for c in reranked]
    return QueryResponse(answer=answer, sources=sources, rerank_scores=rerank_scores)

@app.post("/reset_session")
async def reset_session_endpoint(payload: Dict[str,str]):
    sid = payload.get("session_id")
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    reset_session(sid)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    # for local dev
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
