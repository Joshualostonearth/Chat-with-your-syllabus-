"""
ingest.py — Phase 1: Document Ingestion Pipeline
==================================================
Loads a PDF or DOCX, splits it into chunks, generates embeddings,
and stores everything in ChromaDB.

Usage (standalone):
    python ingest.py path/to/syllabus.pdf

Usage (from app.py):
    from ingest import run_ingestion_pipeline
    stats = run_ingestion_pipeline("path/to/file.pdf", source_name="syllabus.pdf")
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── Constants (must match rag_chain.py) ──────────────────────────────────────

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "syllabus_chunks"
EMBED_MODEL     = "models/text-embedding-004"   # Must match rag_chain.py

CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 80    # overlap between consecutive chunks


# ── Step 1: Load document ─────────────────────────────────────────────────────

def load_document(file_path: str) -> list:
    """Load a PDF or DOCX and return a list of LangChain Document objects."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .docx")

    docs = loader.load()
    print(f"  [load]  Loaded {len(docs)} page(s) from '{Path(file_path).name}'")
    return docs


# ── Step 2: Chunk pages ───────────────────────────────────────────────────────

def chunk_pages(docs: list, source_name: str = "") -> list:
    """
    Split documents into overlapping chunks.
    Preserves page number and source in metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Normalise metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if source_name:
            chunk.metadata["source"] = source_name
        # PyPDFLoader stores page as 0-indexed integer; convert to 1-indexed
        if "page" in chunk.metadata:
            chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

    print(f"  [chunk] Split into {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Step 3: Generate embeddings & store ──────────────────────────────────────

def store_in_chroma(chunks: list) -> Chroma:
    """
    Embed all chunks and upsert them into ChromaDB.
    Re-uses an existing collection so re-ingestion replaces old data.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    # Delete existing collection so we start fresh on each ingest
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  [chroma] Cleared previous collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    count = vectorstore._collection.count()
    print(f"  [chroma] Stored {count} vectors in '{CHROMA_DIR}'")
    return vectorstore


# ── Step 4: Derive topic count (heuristic) ────────────────────────────────────

def estimate_topics(chunks: list) -> int:
    """
    Rough heuristic: count chunks that start with heading-like text
    (ALL CAPS word, or numbered section like '1.' / 'Unit 3').
    """
    import re
    pattern = re.compile(
        r"^(Unit\s+\d|Module\s+\d|\d+\.\s+[A-Z]|[A-Z][A-Z ]{4,})",
        re.MULTILINE
    )
    hits = set()
    for chunk in chunks:
        if pattern.search(chunk.page_content):
            hits.add(chunk.page_content[:40])
    return max(len(hits), 1)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_ingestion_pipeline(file_path: str, source_name: str = "") -> dict:
    """
    Run the complete ingestion pipeline and return processing stats.

    Returns:
        { "pages": int, "chunks": int, "topics": int }
    """
    print(f"\n{'='*50}")
    print(f"  SyllabusAI — Ingestion Pipeline")
    print(f"  File: {file_path}")
    print(f"{'='*50}")

    name   = source_name or Path(file_path).name
    docs   = load_document(file_path)
    chunks = chunk_pages(docs, source_name=name)
    store_in_chroma(chunks)

    stats = {
        "pages":  len(docs),
        "chunks": len(chunks),
        "topics": estimate_topics(chunks),
    }

    print(f"\n  ✅  Done — {stats['pages']} pages, "
          f"{stats['chunks']} chunks, ~{stats['topics']} topics\n")
    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_or_docx>")
        sys.exit(1)

    run_ingestion_pipeline(sys.argv[1])