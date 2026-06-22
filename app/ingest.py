import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR        = "data/"
CHROMA_DIR      = "chroma_db/"
BM25_INDEX_PATH = "bm25_index.pkl"
EMBED_MODEL     = "nomic-embed-text"

def load_and_chunk_pdfs():
    print("📄 Loading PDFs...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages from PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    print("🔢 Building ChromaDB vector store...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"   Vector store saved to {CHROMA_DIR}")
    return vectorstore

def build_bm25_index(chunks):
    print("🔍 Building BM25 keyword index...")
    texts = [chunk.page_content for chunk in chunks]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    # Save BM25 index + raw texts to disk
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": [c.metadata for c in chunks]}, f)

    print(f"   BM25 index saved to {BM25_INDEX_PATH}")
    return bm25

if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    build_vector_store(chunks)
    build_bm25_index(chunks)
    print("\n✅ Ingestion complete! Ready for hybrid search.")