from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_chain import rag_answer

# ── App Setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG API",
    description="Hybrid search RAG system with BM25 + Vector search + Reranking",
    version="1.0.0"
)

# ── Request/Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[str]


# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Enterprise RAG API is running!",
        "endpoints": {
            "query":  "POST /query",
            "health": "GET /health",
            "docs":   "GET /docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_answer(request.question)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))