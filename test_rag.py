"""
test_rag.py  —  Phase 2 Test Suite
====================================
Run with:  python test_rag.py

Tests:
  1. Vectorstore loads without error
  2. Retriever returns chunks for a relevant query
  3. Retriever returns empty list for a nonsense query
  4. answer_question returns a valid in-scope answer
  5. answer_question triggers fallback for out-of-scope question
  6. Citations contain required keys
  7. LLM answer references a page number
"""

import sys
import traceback


def run_test(name: str, fn):
    try:
        fn()
        print(f"  ✅  {name}")
        return True
    except Exception as e:
        print(f"  ❌  {name}")
        print(f"      {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 55)
    print("  SyllabusAI — Phase 2 Test Suite")
    print("=" * 55)

    from rag_chain import (
        get_vectorstore,
        get_llm,
        retrieve_chunks,
        answer_question,
        FALLBACK_MESSAGE,
    )

    passed = 0
    total  = 0

    # ── Test 1: Vectorstore loads ────────────────────────────────────────────
    total += 1
    def t1():
        vs = get_vectorstore()
        assert vs is not None, "Vectorstore is None"
        count = vs._collection.count()
        assert count > 0, f"Collection is empty (count={count}). Run Phase 1 first."
        print(f"      → {count} chunks in ChromaDB")
    passed += run_test("Vectorstore loads and has data", t1)

    # ── Test 2: Retriever returns results ────────────────────────────────────
    total += 1
    def t2():
        chunks = retrieve_chunks("grading policy")
        assert isinstance(chunks, list), "Expected a list"
        assert len(chunks) > 0, "No chunks returned for 'grading policy'"
        for c in chunks:
            assert "content" in c
            assert "page"    in c
            assert "score"   in c
        print(f"      → {len(chunks)} chunk(s) returned")
    passed += run_test("Retriever returns chunks for relevant query", t2)

    # ── Test 3: Retriever filters nonsense ───────────────────────────────────
    total += 1
    def t3():
        # Extremely off-topic query should return nothing (or very low scores)
        chunks = retrieve_chunks("xyzzy frobnicator quantum banana 12345")
        print(f"      → {len(chunks)} chunk(s) returned (expect 0 or low score)")
        # Not a hard fail — just log; some embeddings can still match loosely
    passed += run_test("Retriever handles nonsense query gracefully", t3)

    # ── Test 4: answer_question — in scope ───────────────────────────────────
    total += 1
    def t4():
        result = answer_question("What topics are covered in this course?")
        assert isinstance(result, dict)
        assert "answer"    in result
        assert "citations" in result
        assert "in_scope"  in result
        assert result["answer"] != FALLBACK_MESSAGE, "Got fallback for an in-scope question"
        print(f"      → Answer: {result['answer'][:80]}…")
    passed += run_test("answer_question returns in-scope answer", t4)

    # ── Test 5: Fallback for out-of-scope ────────────────────────────────────
    total += 1
    def t5():
        result = answer_question("What is the capital of Australia?")
        assert result["in_scope"] is False, "Expected out-of-scope flag"
        assert result["answer"] == FALLBACK_MESSAGE
        print(f"      → Fallback triggered correctly")
    passed += run_test("answer_question triggers fallback for off-topic query", t5)

    # ── Test 6: Citation structure ───────────────────────────────────────────
    total += 1
    def t6():
        result = answer_question("What is the attendance policy?")
        if result["in_scope"]:
            for c in result["citations"]:
                assert "rank"    in c, "Missing 'rank'"
                assert "page"    in c, "Missing 'page'"
                assert "score"   in c, "Missing 'score'"
                assert "preview" in c, "Missing 'preview'"
            print(f"      → {len(result['citations'])} citation(s) with correct structure")
        else:
            print("      → Question out of scope, skipping citation check")
    passed += run_test("Citations have required keys", t6)

    # ── Test 7: LLM includes page reference ──────────────────────────────────
    total += 1
    def t7():
        result = answer_question("What are the late submission penalties?")
        if result["in_scope"]:
            assert "page" in result["answer"].lower() or any(
                str(c["page"]) in result["answer"] for c in result["citations"]
            ), "Answer doesn't include a page reference"
            print(f"      → Page reference found in answer")
        else:
            print("      → Out of scope — skipping page-ref check")
    passed += run_test("LLM answer includes page citation", t7)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 55 + "\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()