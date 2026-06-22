import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ── Paths & Models ─────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db/"
BM25_INDEX_PATH = "bm25_index.pkl"
EMBED_MODEL     = "nomic-embed-text"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Load indexes once at import time ───────────────────────────────────
print("Loading vector store...")
embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

print("Loading BM25 index...")
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25      = bm25_data["bm25"]
texts     = bm25_data["texts"]
metadatas = bm25_data["metadatas"]

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("All indexes loaded successfully!")


# ── Step 1: Vector Search ──────────────────────────────────────────────
def vector_search(query, top_k=10):
    results = vectorstore.similarity_search(query, k=top_k)
    return results   # list of Document objects


# ── Step 2: BM25 Keyword Search ────────────────────────────────────────
def bm25_search(query, top_k=10):
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)

    # Get top_k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text":     texts[idx],
            "metadata": metadatas[idx],
            "score":    scores[idx]
        })
    return results


# ── Step 3: Reciprocal Rank Fusion ─────────────────────────────────────
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}

    # Score from vector results
    for rank, doc in enumerate(vector_results):
        key         = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Score from BM25 results
    for rank, doc in enumerate(bm25_results):
        key         = doc["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return top 20 texts for reranking
    return [key for key, _ in sorted_keys[:20]]


# ── Step 4: Rerank with Cross-Encoder ─────────────────────────────────
def rerank(query, candidates, top_k=5):
    if not candidates:
        return []

    pairs  = [(query, text) for text in candidates]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [text for text, _ in ranked[:top_k]]


# ── Main Hybrid Retriever ──────────────────────────────────────────────
def hybrid_retrieve(query, top_k=5):
    print(f"\n🔍 Query: {query}")

    # Step 1: Both searches
    vector_results = vector_search(query, top_k=10)
    bm25_results   = bm25_search(query, top_k=10)
    print(f"   Vector results: {len(vector_results)} | BM25 results: {len(bm25_results)}")

    # Step 2: Fuse with RRF
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    print(f"   After RRF fusion: {len(fused)} candidates")

    # Step 3: Rerank
    final = rerank(query, fused, top_k=top_k)
    print(f"   After reranking: {len(final)} final chunks")

    return final


# ── Quick Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    query   = "What is this document about?"
    results = hybrid_retrieve(query)

    print("\n── Top Results ──────────────────────────────────────────")
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] {chunk[:300]}...")