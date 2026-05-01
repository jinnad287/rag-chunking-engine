"""
Test suite for the Smart Chunking & Retrieval System.
Run with:  python test_chunking.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from chunking_system import (
    preprocess, tokenise, tf, tfidf_vector, cosine_similarity, build_idf,
    split_sentences, semantic_chunk, structure_aware_chunk, overlap_chunk,
    score_chunks, analyse, load_document, Chunk,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    _results.append(condition)


# ──────────────────────────────────────────────────────────────────────────────
# Sample texts
# ──────────────────────────────────────────────────────────────────────────────

SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Dogs are loyal animals. "
    "Foxes are clever creatures that adapt well to many environments."
)

STRUCTURED_TEXT = """
# Chapter 1: Climate Change Basics

Climate change refers to long-term shifts in global temperatures and weather patterns.
These shifts may be natural, but since the 19th century, human activities have been
the main driver of climate change.

## 1.1 Greenhouse Gases

The main greenhouse gases are carbon dioxide, methane, and nitrous oxide. Carbon dioxide
is released by burning fossil fuels such as coal, oil, and natural gas. Methane is
emitted during the production and transport of coal, oil, and natural gas.

## 1.2 Effects

Rising temperatures lead to melting ice caps, rising sea levels, and more extreme
weather events. Coral reefs are dying due to ocean acidification. Species are going
extinct at unprecedented rates.

# Chapter 2: Solutions

Renewable energy sources such as solar, wind, and hydropower can replace fossil fuels.
Energy efficiency improvements in buildings, transport, and industry reduce emissions.
Carbon capture and storage technology removes CO2 directly from the atmosphere.

## 2.1 Policy Approaches

Carbon taxes and cap-and-trade systems create economic incentives to reduce emissions.
International agreements like the Paris Accord set national targets. Green New Deal
proposals combine climate action with economic development goals.
"""

LONG_TEXT = (STRUCTURED_TEXT + "\n\n") * 3  # replicate for overlap testing


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing():
    print("\n[Preprocessing]")
    t = preprocess("  Hello\r\nWorld\n\n\n  End  ")
    check("strips leading/trailing whitespace", not t.startswith(' ') and not t.endswith(' '))
    check("normalises line endings", '\r' not in t)
    check("collapses triple newlines", '\n\n\n' not in t)


def test_tokeniser():
    print("\n[Tokeniser]")
    tokens = tokenise("Machine Learning is FUN! 123 go.")
    check("lowercases tokens", all(t == t.lower() for t in tokens))
    check("ignores numbers", '123' not in tokens)
    check("ignores single-char", all(len(t) >= 2 for t in tokens))
    check("returns list", isinstance(tokens, list))


def test_cosine_similarity():
    print("\n[Cosine Similarity]")
    a = {'dog': 0.5, 'cat': 0.5}
    b = {'dog': 0.5, 'cat': 0.5}
    c = {'fish': 1.0}
    check("identical vectors → 1.0", abs(cosine_similarity(a, b) - 1.0) < 1e-6)
    check("orthogonal vectors → 0.0", cosine_similarity(a, c) == 0.0)
    check("empty vectors → 0.0", cosine_similarity({}, b) == 0.0)
    partial = {'dog': 0.5, 'fish': 0.5}
    sim = cosine_similarity(a, partial)
    check("partial overlap in (0, 1)", 0 < sim < 1, f"{sim:.4f}")


def test_tfidf_pipeline():
    print("\n[TF-IDF Pipeline]")
    corpus = [['cat', 'sat', 'mat'], ['dog', 'ran', 'fast'], ['cat', 'dog']]
    idf = build_idf(corpus)
    check("idf is a dict", isinstance(idf, dict))
    check("rare term has higher idf than common term",
          idf.get('sat', 0) > idf.get('cat', 0),
          f"sat={idf.get('sat',0):.3f} > cat={idf.get('cat',0):.3f}")
    vec = tfidf_vector(['cat', 'sat', 'mat'], idf)
    check("tfidf vector is a dict", isinstance(vec, dict))
    check("all values > 0", all(v > 0 for v in vec.values()))


def test_semantic_chunking():
    print("\n[Semantic Chunking]")
    chunks = semantic_chunk(SHORT_TEXT)
    check("returns list of Chunk", all(isinstance(c, Chunk) for c in chunks))
    check("at least one chunk", len(chunks) >= 1)
    check("all chunks non-empty", all(c.text.strip() for c in chunks))
    check("strategy label correct", all(c.strategy == 'semantic' for c in chunks))

    chunks_struct = semantic_chunk(STRUCTURED_TEXT)
    check("more complex text → multiple chunks", len(chunks_struct) >= 2,
          f"{len(chunks_struct)} chunks")


def test_structure_aware_chunking():
    print("\n[Structure-Aware Chunking]")
    chunks = structure_aware_chunk(STRUCTURED_TEXT)
    check("returns list of Chunk", all(isinstance(c, Chunk) for c in chunks))
    check("detects multiple sections", len(chunks) >= 3, f"{len(chunks)} chunks")
    check("strategy label correct", all(c.strategy == 'structure_aware' for c in chunks))
    check("all chunks non-empty", all(c.text.strip() for c in chunks))

    # Fallback: plain text without headings
    plain = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph here."
    plain_chunks = structure_aware_chunk(plain)
    check("fallback to paragraphs on plain text", len(plain_chunks) >= 1)


def test_overlap_chunking():
    print("\n[Overlap-Based Chunking]")
    chunks = overlap_chunk(LONG_TEXT, window=4, overlap=2)
    check("returns list of Chunk", all(isinstance(c, Chunk) for c in chunks))
    check("produces multiple overlapping chunks", len(chunks) >= 3, f"{len(chunks)} chunks")
    check("strategy label correct", all(c.strategy == 'overlap' for c in chunks))

    # Verify overlap: consecutive chunks share sentences
    if len(chunks) >= 2:
        sents_a = set(split_sentences(chunks[0].text))
        sents_b = set(split_sentences(chunks[1].text))
        check("consecutive chunks share some sentences", len(sents_a & sents_b) > 0)

    # Edge case: very short text
    short_chunks = overlap_chunk("Only one sentence here.", window=4, overlap=2)
    check("handles very short text gracefully", len(short_chunks) >= 1)


def test_score_chunks():
    print("\n[Scoring]")
    chunks = [
        Chunk(text="Renewable energy includes solar and wind power.", index=0, strategy='test'),
        Chunk(text="The quick brown fox jumps over the lazy dog.", index=1, strategy='test'),
        Chunk(text="Solar panels convert sunlight into electricity efficiently.", index=2, strategy='test'),
        Chunk(text="Wind turbines generate electricity from kinetic energy of moving air.", index=3, strategy='test'),
    ]
    query = "solar energy and renewable power"
    result = score_chunks(chunks, query, top_k=2)

    check("returns ChunkResult", hasattr(result, 'top_chunks'))
    check("top_k chunks returned", len(result.top_chunks) == 2)
    check("scores list matches top_k length", len(result.similarity_scores) == 2)
    check("scores descending", result.similarity_scores[0] >= result.similarity_scores[1])
    check("avg_score correct",
          abs(result.avg_score - sum(result.similarity_scores) / 2) < 1e-6)
    check("relevant chunk ranked higher than fox chunk",
          "fox" not in result.top_chunks[0].lower())


def test_full_pipeline():
    print("\n[Full Pipeline — analyse()]")
    result = analyse(STRUCTURED_TEXT, "What are the effects of climate change?", verbose=False)

    check("result is dict", isinstance(result, dict))
    check("has 'results' key", 'results' in result)
    check("has 'best_strategy' key", 'best_strategy' in result)
    check("has 'best_chunks' key", 'best_chunks' in result)
    check("has 'final_score' key", 'final_score' in result)

    res = result['results']
    check("three strategies present", set(res.keys()) == {'semantic', 'structure_aware', 'overlap'})

    for strat in ['semantic', 'structure_aware', 'overlap']:
        d = res[strat]
        check(f"  {strat}: has top_chunks",        'top_chunks' in d)
        check(f"  {strat}: has similarity_scores",  'similarity_scores' in d)
        check(f"  {strat}: has avg_score",          'avg_score' in d)
        check(f"  {strat}: avg_score >= 0",         d['avg_score'] >= 0)

    check("best_strategy is valid",
          result['best_strategy'] in {'semantic', 'structure_aware', 'overlap'})
    check("best_chunks non-empty", len(result['best_chunks']) > 0)
    check("final_score >= 0", result['final_score'] >= 0)
    check("best strategy has highest avg",
          result['final_score'] == max(res[s]['avg_score'] for s in res))


def test_load_document():
    print("\n[Document Loading]")
    # Raw string passthrough
    text = "Hello, this is a test document."
    loaded = load_document(text)
    check("raw string passthrough", loaded == text)

    # .txt file
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("This is a text file.\nWith two lines.")
        fname = f.name
    try:
        loaded_txt = load_document(fname)
        check(".txt file loads correctly", "text file" in loaded_txt)
    finally:
        os.unlink(fname)

    # .json file — dict
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump({"title": "Test", "body": "JSON body text."}, f)
        jname = f.name
    try:
        loaded_json = load_document(jname)
        check(".json dict loads as JSON string", "title" in loaded_json)
    finally:
        os.unlink(jname)


def test_edge_cases():
    print("\n[Edge Cases]")
    # Empty-ish input
    check("semantic_chunk: single sentence",
          len(semantic_chunk("Just one sentence.")) >= 1)
    check("overlap_chunk: single sentence",
          len(overlap_chunk("Just one sentence.")) >= 1)
    check("structure_aware_chunk: single paragraph",
          len(structure_aware_chunk("Short paragraph with no headings at all.")) >= 1)

    # Query with no overlap
    chunks = [Chunk(text="Cats and dogs are pets.", index=0, strategy='t')]
    result = score_chunks(chunks, "quantum physics nuclear fission", top_k=1)
    check("zero-relevance query returns score >= 0", result.avg_score >= 0)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 60)
    print("  SMART CHUNKING SYSTEM — TEST SUITE")
    print("=" * 60)

    test_preprocessing()
    test_tokeniser()
    test_cosine_similarity()
    test_tfidf_pipeline()
    test_semantic_chunking()
    test_structure_aware_chunking()
    test_overlap_chunking()
    test_score_chunks()
    test_full_pipeline()
    test_load_document()
    test_edge_cases()

    passed = sum(_results)
    total  = len(_results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} tests passed", end="  ")
    if passed == total:
        print("\033[92m— ALL PASSED ✓\033[0m")
    else:
        print(f"\033[91m— {total - passed} FAILED ✗\033[0m")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
