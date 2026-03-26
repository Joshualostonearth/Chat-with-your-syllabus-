"""
app.py — SyllabusAI FastAPI Backend
=====================================
Exposes the RAG chain and document ingestion pipeline as HTTP endpoints.
The frontend connects to these via fetch().

Run with:
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /ingest          Upload and process a PDF/DOCX syllabus
    POST /chat            Ask a question, get answer + citations
    GET  /status          Get vectorstore stats (chunk count, etc.)
    POST /quiz            Generate quiz questions from the syllabus
    POST /study-plan      Generate a study plan from the syllabus
    POST /topics          Extract and summarise key topics
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_chain import answer_question, get_vectorstore, CHROMA_DIR
from ingest import run_ingestion_pipeline

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="SyllabusAI API", version="1.0.0")

# Allow the HTML frontend (opened via file:// or a dev server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from the same directory
FRONTEND_DIR = Path(__file__).parent
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    task: str  # "quiz" | "study_plan" | "topics"
    extra: str = ""


# ── Helper: check vectorstore is populated ───────────────────────────────────

def require_documents():
    """Raise 400 if no documents have been ingested yet."""
    try:
        vs    = get_vectorstore()
        count = vs._collection.count()
        if count == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents ingested yet. Upload a syllabus first."
            )
        return count
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Vectorstore error: {e}")


@app.get("/status")
def get_status():
    """Return current vectorstore stats."""
    try:
        vs    = get_vectorstore()
        count = vs._collection.count()
        # Peek at metadata to derive page/topic counts
        sample = vs._collection.get(limit=count, include=["metadatas"])
        pages  = set()
        for m in sample.get("metadatas", []):
            if m and "page" in m:
                pages.add(m["page"])

        return {
            "status":      "ready" if count > 0 else "empty",
            "chunk_count": count,
            "page_count":  len(pages),
        }
    except Exception:
        return {"status": "empty", "chunk_count": 0, "page_count": 0}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Accept a PDF or DOCX upload, run the full Phase 1 ingestion pipeline,
    and return stats about the processed document.
    """
    allowed = {".pdf", ".docx"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        stats = run_ingestion_pipeline(tmp_path, source_name=file.filename)
        return {
            "message":     "Document ingested successfully.",
            "filename":    file.filename,
            "pages":       stats.get("pages",  0),
            "chunks":      stats.get("chunks", 0),
            "topics":      stats.get("topics", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Answer a question about the uploaded syllabus.
    Returns the answer text and a list of source citations.
    """
    require_documents()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(req.question)
    return {
        "answer":    result["answer"],
        "citations": result["citations"],
        "in_scope":  result["in_scope"],
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Use the RAG chain to generate structured content:
    - quiz:       5 Q&A pairs from the syllabus
    - study_plan: Week-by-week study schedule
    - topics:     List of key topics with brief descriptions
    """
    require_documents()

    prompts = {
        "quiz": (
            "Generate exactly 5 quiz questions with answers based strictly on the "
            "syllabus content. Format each as:\n"
            "Q1: <question>\nA1: <answer>\n\n"
            "Q2: <question>\nA2: <answer>\n...\n"
            "Only include topics explicitly mentioned in the syllabus."
        ),
        "study_plan": (
            "Based on the topics, assignments, and deadlines in this syllabus, "
            "create a practical week-by-week study plan for the entire course. "
            "Format: Week N: <focus topic> — <specific tasks>. "
            "Include exam prep weeks."
        ),
        "topics": (
            "List all major topics and units covered in this course. "
            "For each topic write one sentence describing what it covers. "
            "Format: • <Topic Name>: <one-sentence description>. "
            "Include at least 6 topics if they exist."
        ),
    }

    task = req.task
    if task not in prompts:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    result = answer_question(prompts[task])
    return {
        "content":  result["answer"],
        "in_scope": result["in_scope"],
    }