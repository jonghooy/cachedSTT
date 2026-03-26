# Knowledge Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent Knowledge service (FastAPI + Vue + ChromaDB) that manages callbot knowledge (documents, FAQ, prompts) and provides RAG search to the Brain S2S engine.

**Architecture:** Separate project at `/home/jonghooy/work/knowledge-service/`. Backend (FastAPI port 8100) serves REST API for document ingestion, FAQ/prompt CRUD, and hybrid RAG search. Frontend (Vue 3 + Vite) provides admin UI. Brain connects via HTTP API at startup for config pre-loading and optional real-time search.

**Tech Stack:** Python 3.11, FastAPI, ChromaDB, SQLite (aiosqlite), BGE-M3 (BAAI/bge-m3), BGE-Reranker (BAAI/bge-reranker-v2-m3), Kiwi (korean morpheme), pypdf, python-docx, Vue 3, Vite, Pinia

---

## File Structure

```
/home/jonghooy/work/knowledge-service/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, lifespan
│   ├── config.py                  # Settings (paths, model names, ports)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── sqlite.py              # SQLite schema + CRUD (documents, faq, prompts)
│   │   └── chroma.py              # ChromaDB wrapper (add, search, delete collections)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py              # PDF/Word → plain text with structure markers
│   │   ├── chunker.py             # Semantic chunking + parent-child splitting
│   │   ├── contextual.py          # Prepend document context summary to each chunk
│   │   └── embedder.py            # BGE-M3 dense + Kiwi BM25 sparse embedding
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py       # Dense + sparse search with RRF fusion
│   │   ├── reranker.py            # BGE cross-encoder reranking
│   │   └── query_transform.py     # Colloquial → search query, multi-query expansion
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py           # POST upload, GET list, GET detail, DELETE
│   │   ├── faq.py                 # FAQ CRUD endpoints
│   │   ├── prompts.py             # Prompt CRUD + version history
│   │   ├── search.py              # RAG search endpoint (for testing UI)
│   │   └── brain.py               # Brain-facing API (config, search, webhook trigger)
│   └── tests/
│       ├── __init__.py
│       ├── test_parser.py
│       ├── test_chunker.py
│       ├── test_embedder.py
│       ├── test_hybrid_search.py
│       ├── test_reranker.py
│       ├── test_api_documents.py
│       ├── test_api_faq.py
│       └── test_api_brain.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js
│       ├── App.vue
│       ├── router.js
│       ├── api/client.js           # Axios wrapper for backend API
│       ├── stores/knowledge.js     # Pinia store
│       ├── views/
│       │   ├── DocumentsView.vue   # Upload, list, chunk preview
│       │   ├── FaqView.vue         # FAQ CRUD
│       │   ├── PromptsView.vue     # Prompt editor + version list
│       │   └── SearchView.vue      # RAG test: query → results + scores
│       └── components/
│           ├── FileUpload.vue
│           ├── ChunkPreview.vue
│           └── SearchResult.vue
├── storage/                        # Runtime data (gitignored)
│   ├── documents/
│   ├── chroma/
│   └── knowledge.db
├── requirements.txt
├── .gitignore
└── README.md
```

---

### Task 1: Project Scaffolding + Database Layer

**Files:**
- Create: `backend/main.py`, `backend/config.py`, `backend/db/__init__.py`, `backend/db/sqlite.py`, `backend/db/chroma.py`
- Create: `requirements.txt`, `.gitignore`, `README.md`
- Test: `backend/tests/__init__.py`, `backend/tests/test_db.py`

- [ ] **Step 1: Initialize project directory and git**

```bash
mkdir -p /home/jonghooy/work/knowledge-service
cd /home/jonghooy/work/knowledge-service
git init
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
node_modules/
dist/
storage/
.env
*.log
```

- [ ] **Step 3: Create requirements.txt**

```
fastapi==0.115.*
uvicorn[standard]
aiosqlite
python-multipart
chromadb
sentence-transformers
FlagEmbedding
pypdf
python-docx
kiwipiepy
httpx
pytest
pytest-asyncio
```

- [ ] **Step 4: Create config.py**

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
CHROMA_DIR = STORAGE_DIR / "chroma"
SQLITE_PATH = STORAGE_DIR / "knowledge.db"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Ensure dirs exist
for d in [STORAGE_DIR, DOCUMENTS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Write SQLite schema + CRUD (backend/db/sqlite.py)**

Tables: `documents` (id, filename, title, status, chunk_count, created_at), `faq` (id, question, answer, category, created_at), `prompts` (id, name, content, version, is_active, created_at).

Use `aiosqlite` for async access. Functions: `init_db()`, `add_document()`, `list_documents()`, `delete_document()`, `add_faq()`, `list_faq()`, `update_faq()`, `delete_faq()`, `add_prompt()`, `list_prompts()`, `get_active_prompt()`, `activate_prompt()`.

- [ ] **Step 6: Write ChromaDB wrapper (backend/db/chroma.py)**

Functions: `init_chroma()`, `add_chunks(doc_id, chunks, embeddings, metadatas)`, `search(query_embedding, top_k)`, `delete_document_chunks(doc_id)`, `get_collection_stats()`.

- [ ] **Step 7: Write test for DB layer**

```python
# backend/tests/test_db.py
import pytest, asyncio
from db.sqlite import init_db, add_faq, list_faq, delete_faq

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

@pytest.mark.asyncio
async def test_faq_crud(db_path):
    await init_db(db_path)
    faq_id = await add_faq(db_path, "카드 분실", "즉시 정지", "카드")
    faqs = await list_faq(db_path)
    assert len(faqs) == 1
    assert faqs[0]["question"] == "카드 분실"
    await delete_faq(db_path, faq_id)
    assert len(await list_faq(db_path)) == 0
```

- [ ] **Step 8: Run test**

```bash
cd /home/jonghooy/work/knowledge-service
pip install -r requirements.txt
PYTHONPATH=backend pytest backend/tests/test_db.py -v
```

- [ ] **Step 9: Create FastAPI app skeleton (backend/main.py)**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db.sqlite import init_db
from config import SQLITE_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(SQLITE_PATH)
    yield

app = FastAPI(title="Knowledge Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding with SQLite + ChromaDB layer"
```

---

### Task 2: Document Ingestion Pipeline — Parser + Chunker

**Files:**
- Create: `backend/ingestion/__init__.py`, `backend/ingestion/parser.py`, `backend/ingestion/chunker.py`
- Test: `backend/tests/test_parser.py`, `backend/tests/test_chunker.py`

- [ ] **Step 1: Write parser test**

```python
# backend/tests/test_parser.py
from ingestion.parser import parse_pdf, parse_docx

def test_parse_pdf(tmp_path):
    # Create a minimal test PDF with reportlab or use a fixture
    result = parse_pdf("tests/fixtures/sample.pdf")
    assert result["text"]
    assert result["sections"]  # list of {title, content, level}

def test_parse_docx(tmp_path):
    result = parse_docx("tests/fixtures/sample.docx")
    assert result["text"]
```

- [ ] **Step 2: Implement parser.py**

`parse_pdf(path)` — uses pypdf to extract text, detect headings by font size/bold.
`parse_docx(path)` — uses python-docx to extract paragraphs, detect headings by style.
Both return `{"text": str, "sections": [{"title": str, "content": str, "level": int}]}`.

- [ ] **Step 3: Write chunker test**

```python
# backend/tests/test_chunker.py
from ingestion.chunker import semantic_chunk

def test_parent_child_chunking():
    sections = [
        {"title": "1. 카드 분실", "content": "카드를 분실한 경우...(300자)", "level": 1},
        {"title": "1.1 긴급정지", "content": "긴급정지 절차는...(200자)", "level": 2},
    ]
    chunks = semantic_chunk(sections, child_max=200, parent_overlap=True)
    parents = [c for c in chunks if c["type"] == "parent"]
    children = [c for c in chunks if c["type"] == "child"]
    assert len(parents) > 0
    assert len(children) >= len(parents)
    assert all(c["parent_id"] for c in children)
```

- [ ] **Step 4: Implement chunker.py**

`semantic_chunk(sections, child_max=200, parent_overlap=True)` — splits sections into parent chunks (full section) and child chunks (sub-segments of ~200 chars). Each child has `parent_id` reference. Returns `[{"id", "type", "text", "parent_id", "title", "level"}]`.

- [ ] **Step 5: Run tests, commit**

```bash
PYTHONPATH=backend pytest backend/tests/test_parser.py backend/tests/test_chunker.py -v
git add -A && git commit -m "feat: document parser (PDF/Word) + semantic parent-child chunker"
```

---

### Task 3: Contextual Retrieval + Hybrid Embedding

**Files:**
- Create: `backend/ingestion/contextual.py`, `backend/ingestion/embedder.py`
- Test: `backend/tests/test_embedder.py`

- [ ] **Step 1: Implement contextual.py**

`add_context(chunks, doc_title)` — prepends each chunk with a context line:
`"[문서: {doc_title} | 섹션: {chunk.title}] "` + original chunk text.
This is the simplified version of Anthropic's contextual retrieval (full LLM summarization can be added later).

- [ ] **Step 2: Write embedder test**

```python
# backend/tests/test_embedder.py
from ingestion.embedder import Embedder

def test_dense_embedding():
    embedder = Embedder()
    vectors = embedder.embed_dense(["카드 분실 신고 절차"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024  # BGE-M3 dim

def test_sparse_tokens():
    embedder = Embedder()
    tokens = embedder.tokenize_sparse("카드를 분실했습니다")
    assert "카드" in tokens
    assert "분실" in tokens
```

- [ ] **Step 3: Implement embedder.py**

```python
class Embedder:
    def __init__(self):
        self.dense_model = None  # lazy load BGE-M3
        self.kiwi = None         # lazy load Kiwi

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """BGE-M3 dense embedding."""
        ...

    def tokenize_sparse(self, text: str) -> list[str]:
        """Kiwi morpheme tokenization for BM25."""
        ...
```

- [ ] **Step 4: Run tests, commit**

```bash
PYTHONPATH=backend pytest backend/tests/test_embedder.py -v
git add -A && git commit -m "feat: contextual retrieval + BGE-M3 dense + Kiwi BM25 sparse embedder"
```

---

### Task 4: Retrieval Pipeline — Hybrid Search + Reranker

**Files:**
- Create: `backend/retrieval/__init__.py`, `backend/retrieval/hybrid_search.py`, `backend/retrieval/reranker.py`, `backend/retrieval/query_transform.py`
- Test: `backend/tests/test_hybrid_search.py`, `backend/tests/test_reranker.py`

- [ ] **Step 1: Write hybrid search test**

```python
# backend/tests/test_hybrid_search.py
def test_rrf_fusion():
    from retrieval.hybrid_search import rrf_fusion
    dense_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    sparse_results = [("doc2", 3.5), ("doc1", 2.1), ("doc4", 1.0)]
    fused = rrf_fusion(dense_results, sparse_results, k=60)
    # doc2 should rank highest (appears in both)
    assert fused[0][0] == "doc2"
```

- [ ] **Step 2: Implement hybrid_search.py**

`HybridSearcher` class:
- `search(query, top_k=20)` — runs dense search on ChromaDB + sparse BM25 search, fuses with RRF.
- `rrf_fusion(dense_results, sparse_results, k=60)` — Reciprocal Rank Fusion.

- [ ] **Step 3: Implement reranker.py**

`Reranker` class:
- Lazy-loads `BAAI/bge-reranker-v2-m3`.
- `rerank(query, candidates, top_k=3)` — scores query-chunk pairs, returns top-k.

- [ ] **Step 4: Implement query_transform.py**

`transform_query(query)` — basic colloquial→formal mapping rules.
`multi_query(query)` — generates 2-3 query variants (original + transformed + keyword-only).

- [ ] **Step 5: Run tests, commit**

```bash
PYTHONPATH=backend pytest backend/tests/test_hybrid_search.py backend/tests/test_reranker.py -v
git add -A && git commit -m "feat: hybrid search (dense+sparse+RRF) + cross-encoder reranker"
```

---

### Task 5: REST API — Documents + FAQ + Prompts

**Files:**
- Create: `backend/api/__init__.py`, `backend/api/documents.py`, `backend/api/faq.py`, `backend/api/prompts.py`, `backend/api/search.py`
- Test: `backend/tests/test_api_documents.py`, `backend/tests/test_api_faq.py`

- [ ] **Step 1: Implement documents API**

```
POST /api/documents/upload    — multipart file upload → parse → chunk → embed → store
GET  /api/documents           — list all documents + status
GET  /api/documents/{id}      — document detail + chunk preview
DELETE /api/documents/{id}    — delete document + chunks from ChromaDB
```

Background task for ingestion (upload returns immediately, status updates via polling).

- [ ] **Step 2: Implement FAQ API**

```
GET    /api/faq               — list all FAQ
POST   /api/faq               — create FAQ {question, answer, category}
PUT    /api/faq/{id}          — update FAQ
DELETE /api/faq/{id}          — delete FAQ
```

- [ ] **Step 3: Implement prompts API**

```
GET    /api/prompts           — list all prompt versions
POST   /api/prompts           — create new version {name, content}
PUT    /api/prompts/{id}/activate — set as active prompt
GET    /api/prompts/active    — get current active prompt
```

- [ ] **Step 4: Implement search API (for testing UI)**

```
POST /api/search  {"query": "카드 분실", "top_k": 3}
→ {"results": [{"text", "source", "score", "chunk_type"}]}
```

Calls hybrid search → reranker → parent expansion → return.

- [ ] **Step 5: Register routers in main.py**

```python
from api.documents import router as docs_router
from api.faq import router as faq_router
from api.prompts import router as prompts_router
from api.search import router as search_router
app.include_router(docs_router, prefix="/api/documents", tags=["documents"])
app.include_router(faq_router, prefix="/api/faq", tags=["faq"])
app.include_router(prompts_router, prefix="/api/prompts", tags=["prompts"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
```

- [ ] **Step 6: Write API tests with TestClient**

```python
# backend/tests/test_api_faq.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_faq_crud():
    res = client.post("/api/faq", json={"question": "카드 분실", "answer": "정지 처리", "category": "카드"})
    assert res.status_code == 200
    faq_id = res.json()["id"]
    res = client.get("/api/faq")
    assert len(res.json()["faqs"]) == 1
    client.delete(f"/api/faq/{faq_id}")
    assert len(client.get("/api/faq").json()["faqs"]) == 0
```

- [ ] **Step 7: Run tests, commit**

```bash
PYTHONPATH=backend pytest backend/tests/test_api_faq.py backend/tests/test_api_documents.py -v
git add -A && git commit -m "feat: REST API for documents, FAQ, prompts, and RAG search"
```

---

### Task 6: Brain API — Config + Search + Webhook

**Files:**
- Create: `backend/api/brain.py`
- Test: `backend/tests/test_api_brain.py`

- [ ] **Step 1: Implement Brain API**

```python
# backend/api/brain.py
@router.get("/config")
async def brain_config(brain_id: str = "default"):
    """Brain 시작 시 호출: 시스템 프롬프트 + FAQ + 메타데이터"""
    prompt = await get_active_prompt(SQLITE_PATH)
    faqs = await list_faq(SQLITE_PATH)
    return {
        "system_prompt": prompt["content"] if prompt else "",
        "faq": [{"q": f["question"], "a": f["answer"]} for f in faqs],
        "metadata": {"updated_at": prompt["created_at"] if prompt else None}
    }

@router.post("/search")
async def brain_search(body: dict):
    """Brain에서 실시간 RAG 검색"""
    query = body["query"]
    top_k = body.get("top_k", 3)
    results = await rag_pipeline.search(query, top_k)
    return {"results": results}
```

- [ ] **Step 2: Write test**

```python
def test_brain_config():
    # Add a prompt and FAQ, then call /api/brain/config
    client.post("/api/prompts", json={"name": "v1", "content": "테스트 프롬프트"})
    client.put("/api/prompts/1/activate")
    client.post("/api/faq", json={"question": "Q", "answer": "A", "category": "C"})
    res = client.get("/api/brain/config?brain_id=test")
    assert res.json()["system_prompt"] == "테스트 프롬프트"
    assert len(res.json()["faq"]) == 1
```

- [ ] **Step 3: Register brain router in main.py**

```python
from api.brain import router as brain_router
app.include_router(brain_router, prefix="/api/brain", tags=["brain"])
```

- [ ] **Step 4: Run tests, commit**

```bash
PYTHONPATH=backend pytest backend/tests/test_api_brain.py -v
git add -A && git commit -m "feat: Brain API (config pre-loading + RAG search)"
```

---

### Task 7: Vue Frontend — Admin UI

**Files:**
- Create: all files under `frontend/`

- [ ] **Step 1: Scaffold Vue project**

```bash
cd /home/jonghooy/work/knowledge-service
npm create vite@latest frontend -- --template vue
cd frontend && npm install && npm install vue-router@4 pinia axios
```

- [ ] **Step 2: Create API client (frontend/src/api/client.js)**

```javascript
import axios from 'axios'
const api = axios.create({ baseURL: 'http://localhost:8100/api' })
export default api
```

- [ ] **Step 3: Create router (frontend/src/router.js)**

4 routes: `/documents`, `/faq`, `/prompts`, `/search`

- [ ] **Step 4: Create DocumentsView.vue**

- File upload form (drag & drop)
- Documents list table (filename, status, chunk count, actions)
- Click row → chunk preview panel

- [ ] **Step 5: Create FaqView.vue**

- FAQ list table (question, answer, category)
- Add/edit form (inline or modal)
- Delete button with confirmation

- [ ] **Step 6: Create PromptsView.vue**

- Prompt editor textarea (monospace, large)
- Version history list
- "Activate" button per version
- Active version highlighted

- [ ] **Step 7: Create SearchView.vue**

- Query input + search button
- Results list: text preview, source file, relevance score, chunk type (parent/child)
- Useful for testing RAG quality

- [ ] **Step 8: Configure vite proxy for dev**

```javascript
// vite.config.js
export default {
  server: {
    proxy: { '/api': 'http://localhost:8100' }
  }
}
```

- [ ] **Step 9: Test frontend dev server**

```bash
cd frontend && npm run dev
# Open http://localhost:5173 and verify pages load
```

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat: Vue admin UI (documents, FAQ, prompts, search test)"
```

---

### Task 8: Brain-side Knowledge Client Integration

**Files:**
- Modify: `/home/jonghooy/work/cachedSTT/realtime_demo/s2s_pipeline.py`
- Create: `/home/jonghooy/work/cachedSTT/realtime_demo/knowledge_client.py`

- [ ] **Step 1: Create knowledge_client.py in Brain project**

```python
class KnowledgeClient:
    """Knowledge 서비스에서 설정을 로딩하고 로컬 캐시."""
    def __init__(self, base_url="http://localhost:8100/api/brain", brain_id="default"):
        ...
    async def load_config(self) -> dict:
        """시작 시 1회: 시스템 프롬프트 + FAQ 로딩."""
        ...
    async def search(self, query, top_k=3) -> list:
        """실시간 RAG 검색 (캐시 미스 시 fallback)."""
        ...
    def get_system_prompt(self) -> str:
        """캐시된 시스템 프롬프트 반환."""
        ...
    def get_faq_context(self, query) -> str:
        """캐시된 FAQ에서 관련 항목 검색."""
        ...
```

- [ ] **Step 2: Integrate into s2s_pipeline.py**

Modify `S2SPipeline.__init__()` to accept optional `knowledge_client`. If provided, use knowledge client's system prompt instead of hardcoded `SYSTEM_PROMPT`. Append FAQ matches to LLM context.

- [ ] **Step 3: Add webhook endpoint to Brain server.py**

```python
@app.post("/api/knowledge/refresh")
async def knowledge_refresh(body: dict):
    """Knowledge 변경 시 캐시 갱신."""
    if knowledge_client:
        await knowledge_client.load_config()
    return {"status": "refreshed"}
```

- [ ] **Step 4: Test integration**

```bash
# Terminal 1: Start Knowledge service
cd /home/jonghooy/work/knowledge-service
PYTHONPATH=backend uvicorn backend.main:app --port 8100

# Terminal 2: Start Brain with --s2s
cd /home/jonghooy/work/cachedSTT
python realtime_demo/server.py --s2s --port 3000

# Terminal 3: Verify
curl http://localhost:8100/api/brain/config
curl http://localhost:3000/api/s2s/status
```

- [ ] **Step 5: Commit in both projects**

```bash
# Knowledge project
cd /home/jonghooy/work/knowledge-service
git add -A && git commit -m "feat: complete Knowledge service v1"

# Brain project
cd /home/jonghooy/work/cachedSTT
git add realtime_demo/knowledge_client.py realtime_demo/s2s_pipeline.py realtime_demo/server.py
git commit -m "feat: Knowledge client integration for dynamic prompts and FAQ"
```

---

### Task 9: End-to-End Verification

- [ ] **Step 1: Upload a test document via UI**

Open `http://localhost:5173/documents`, upload a PDF manual, verify chunks appear.

- [ ] **Step 2: Add FAQ entries via UI**

Open `/faq`, add 3-5 callbot FAQ entries.

- [ ] **Step 3: Create and activate a system prompt**

Open `/prompts`, write a callbot system prompt, activate it.

- [ ] **Step 4: Test RAG search**

Open `/search`, enter "카드 분실 신고", verify relevant chunks returned with scores.

- [ ] **Step 5: Test Brain integration**

Speak into Brain (`http://localhost:3000`), verify LLM uses Knowledge-provided prompt and FAQ context.

- [ ] **Step 6: Final commit and push**

```bash
cd /home/jonghooy/work/knowledge-service
git add -A && git commit -m "docs: end-to-end verification complete"
git remote add origin <repo-url>
git push -u origin master
```
