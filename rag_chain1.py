import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_store"
COLLECTION_NAME = "syllabus"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CHAT_MODEL      = "gemini-1.5-flash"
TOP_K           = 4
SCORE_THRESHOLD = 0.35

FALLBACK_MESSAGE = (
    "I couldn't find information about that in the uploaded syllabus. "
    "Try rephrasing, or ask about topics like grading, deadlines, or course topics."
)

# ── Singletons ────────────────────────────────────────────────────────────────
_collection = None
_embedder   = None
_llm        = None

def get_vectorstore():
    global _collection, _embedder
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
        _embedder   = SentenceTransformer(EMBED_MODEL)
    return _collection, _embedder

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0.2,
            max_output_tokens=512,
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    return _llm

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_chunks(query: str) -> list[dict]:
    collection, embedder = get_vectorstore()
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 4)  # cosine distance → similarity
        if score < SCORE_THRESHOLD:
            continue
        chunks.append({
            "content": doc,
            "page":    meta.get("page", "?"),
            "score":   score,
        })
    return chunks

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are SyllabusAI. Answer ONLY from the context below.
If the answer is not in the context, reply with exactly: OUT_OF_SCOPE
Be concise. Always end with a "Source:" line listing the page number(s).

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  "{question}"),
])

# ── Answer ────────────────────────────────────────────────────────────────────
def answer_question(query: str) -> dict:
    chunks = retrieve_chunks(query)

    if not chunks:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "in_scope": False}

    context = "\n\n".join(
        f"[{i}] (Page {c['page']})\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )

    chain = prompt | get_llm() | StrOutputParser()
    raw   = chain.invoke({"context": context, "question": query})

    if "OUT_OF_SCOPE" in raw:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "in_scope": False}

    citations = [
        {
            "rank":    i,
            "page":    c["page"],
            "score":   c["score"],
            "preview": c["content"][:120] + "…" if len(c["content"]) > 120 else c["content"],
        }
        for i, c in enumerate(chunks, 1)
    ]

    return {"answer": raw.strip(), "citations": citations, "in_scope": True}
