from langchain_ollama import OllamaLLM
from app.retriever import hybrid_retrieve

# ── LLM Setup ──────────────────────────────────────────────────────────
LLM_MODEL = "llama3.2:3b"
llm       = OllamaLLM(model=LLM_MODEL)


# ── Prompt Template ────────────────────────────────────────────────────
def build_prompt(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt  = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt


# ── RAG Chain ──────────────────────────────────────────────────────────
def rag_answer(query):
    # Step 1: Retrieve relevant chunks
    chunks = hybrid_retrieve(query)

    if not chunks:
        return {
            "answer":  "No relevant documents found.",
            "sources": []
        }

    # Step 2: Build prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate answer
    print("Generating answer...")
    answer = llm.invoke(prompt)

    return {
        "answer":  answer,
        "sources": chunks
    }


# ── Quick Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    query  = "What security tools or technologies are mentioned?"
    result = rag_answer(query)

    print("\n── Answer ───────────────────────────────────────────────")
    print(result["answer"])
    print("\n── Sources Used ─────────────────────────────────────────")
    for i, chunk in enumerate(result["sources"], 1):
        print(f"\n[{i}] {chunk[:200]}...")