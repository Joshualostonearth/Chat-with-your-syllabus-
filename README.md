# SyllabusAI — Chat with your Course Syllabus
====================

A **fully functional** RAG (Retrieval-Augmented Generation) application that lets you upload a course syllabus (PDF/DOCX) and ask natural questions about grading policies, deadlines, topics, attendance rules, etc.

**Uses Gemini 1.5 Flash** (free tier OK), **ChromaDB** vector store, **FastAPI** backend, modern React-free frontend.

## 🎯 Quick Start (3 minutes)
```
1. Get FREE Gemini API key: https://aistudio.google.com/app/apikey
2. pip install -r requirements.txt
3. cp .env.example .env && edit .env (add your GEMINI_API_KEY)
4. Download sample syllabus PDF (or use your own)
5. python ingest.py your_syllabus.pdf
6. uvicorn app:app --reload --port 8000
7. Open http://localhost:8000
```

**Upload → Chat → Grounded answers with page citations!** ✨

## 📁 Project Structure
```
Chat-with-your-syllabus-/
├── ingest.py      # Phase 1: Load → Chunk → Embed → ChromaDB
├── rag_chain.py   # Phase 2: Retrieve → Gemini → Answer + Citations
├── app.py         # Phase 3: FastAPI (/ingest, /chat, /generate)
├── index.html     # Frontend (served at /)
├── requirements.txt
├── .env.example   # Copy to .env, add GEMINI_API_KEY
├── chroma_db/     # Created after first ingest
└── TODO.md        # Progress tracker
```

## 🚀 Features
| Tab | Endpoint | What it does |
|-----|----------|--------------|
| 💬 **Chat** | `POST /chat` | Ask anything → Answer + page sources |
| 🔍 **Retrieval** | - | See exact chunks retrieved (w/ scores) |
| 📊 **Topics** | `POST /generate?task=topics` | Extract key units/modules |
| 🧠 **Quiz** | `POST /generate?task=quiz` | 5 syllabus-grounded Q&A |
| 📅 **Study Plan** | `POST /generate?task=study_plan` | Week-by-week schedule |

**Status API:** `GET /status` → `{chunk_count, page_count}`

## 🛠️ Troubleshooting
```
❌ "No module named langchain_google_genai"
   → pip install -r requirements.txt

❌ "GEMINI_API_KEY not found"
   → cp .env.example .env && add your key

❌ "No documents ingested"
   → python ingest.py syllabus.pdf FIRST

❌ Frontend says "API offline"
   → uvicorn app:app --reload --port 8000
     (then visit http://localhost:8000)

❌ test_rag.py fails
   → Run Phase 1 ingest first!
```

## 🔬 Testing
```bash
# Phase 1
python ingest.py path/to/syllabus.pdf

# Phase 2 (RAG)
python test_rag.py

# Phase 3 (API)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the final exam worth?"}'
```

## 📈 Architecture
```
PDF/DOCX → ingest.py → ChromaDB (500-char chunks)
       ↓
Query → rag_chain.py → Retrieve TOP-4 → Gemini → Answer + Citations
       ↓
FastAPI → Frontend (file:// or http://localhost:8000)
```

**Production:** Deploy FastAPI to Railway/Vercel, persist `./chroma_db`.

## 🤝 Credits
Built with LangChain, ChromaDB, Google Gemini, FastAPI. UI: Tailwind-inspired, vanilla JS.

---
⭐ **Star if useful!** Questions? Open an issue.

