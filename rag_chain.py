"""
Phase 2: RAG Chain
==================
Connects ChromaDB retriever → Prompt Template → LLM → Answer with citations.
Includes fallback logic for out-of-scope queries.

Usage:
    from rag_chain import answer_question
    result = answer_question("What is the final exam worth?")
    print(result["answer"])
    print(result["citations"])
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ── 1. Constants ────────────────────────────────────────────────────────────

CHROMA_DIR      = "./chroma_db"          # Must match Phase 1
COLLECTION_NAME = "syllabus_chunks"      # Must match Phase 1
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"          # Swap for "gemini-pro" etc.
TOP_K           = 4                      # Chunks to retrieve per query
SCORE_THRESHOLD = 0.35                   # Cosine distance cutoff (lower = more similar)

# ── 2. Initialise shared resources (lazy singletons) ────────────────────────

_vectorstore = None
_llm         = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings   = OpenAIEmbeddings(model=EMBED_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.2,       # Low temp → factual answers
            max_tokens=512,
        )
    return _llm


# ── 3. Retriever ─────────────────────────────────────────────────────────────

def retrieve_chunks(query: str) -> list[dict]:
    """
    Returns up to TOP_K chunks with their metadata and similarity score.
    Filters out chunks that are too dissimilar (score > SCORE_THRESHOLD).
    """
    vs      = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=TOP_K)

    # results: list of (Document, float) where float is cosine *similarity* (0–1)
    filtered = [
        {
            "content":  doc.page_content,
            "page":     doc.metadata.get("page", "?"),
            "source":   doc.metadata.get("source", "syllabus"),
            "score":    round(score, 4),
        }
        for doc, score in results
        if score >= (1 - SCORE_THRESHOLD)   # convert distance to similarity
    ]

    return filtered


# ── 4. Prompt Template ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SyllabusAI, a helpful assistant that answers questions \
strictly based on the provided course syllabus excerpts.

Rules:
- Answer ONLY from the context below. Do NOT use outside knowledge.
- If the answer is not in the context, reply with exactly: OUT_OF_SCOPE
- Be concise (2–4 sentences max).
- Always end your answer with a "Source:" line listing the page number(s).

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  "{question}"),
])


# ── 5. Fallback logic ────────────────────────────────────────────────────────

FALLBACK_MESSAGE = (
    "I couldn't find information about that in the uploaded syllabus. "
    "Try rephrasing, or ask about topics like grading, deadlines, or course topics."
)


def format_context(chunks: list[dict]) -> str:
    """Flatten retrieved chunks into a single context string for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (Page {c['page']}, score={c['score']})\n{c['content']}")
    return "\n\n".join(parts)


def build_citations(chunks: list[dict]) -> list[dict]:
    """Return a clean citation list for the UI."""
    return [
        {
            "rank":    i,
            "page":    c["page"],
            "score":   c["score"],
            "preview": c["content"][:120] + "…" if len(c["content"]) > 120 else c["content"],
        }
        for i, c in enumerate(chunks, 1)
    ]


# ── 6. Main entry point ──────────────────────────────────────────────────────

def answer_question(query: str) -> dict:
    """
    Full RAG pipeline.

    Returns:
        {
            "answer":    str,           # Final answer text shown to user
            "citations": list[dict],    # Source chunks for the UI grounding panel
            "in_scope":  bool,          # False if question was out of scope
            "raw_chunks": list[dict],   # All retrieved chunks (for debugging)
        }
    """
    # 1. Retrieve
    chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "answer":     FALLBACK_MESSAGE,
            "citations":  [],
            "in_scope":   False,
            "raw_chunks": [],
        }

    # 2. Build context string
    context = format_context(chunks)

    # 3. Call LLM
    chain  = prompt | get_llm() | StrOutputParser()
    raw    = chain.invoke({"context": context, "question": query})

    # 4. Detect fallback
    if "OUT_OF_SCOPE" in raw:
        return {
            "answer":     FALLBACK_MESSAGE,
            "citations":  [],
            "in_scope":   False,
            "raw_chunks": chunks,
        }

    return {
        "answer":     raw.strip(),
        "citations":  build_citations(chunks),
        "in_scope":   True,
        "raw_chunks": chunks,
    }


# ── 7. Quick CLI test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What percentage of the grade is the final exam?",
        "When are office hours?",
        "What is the capital of France?",   # Should trigger fallback
    ]

    print("=" * 60)
    print("RAG Chain — Phase 2 Test")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQ: {q}")
        result = answer_question(q)
        print(f"A: {result['answer']}")
        if result["citations"]:
            print("Citations:")
            for c in result["citations"]:
                print(f"  • Page {c['page']} (score {c['score']}): {c['preview']}")
        print("-" * 40)