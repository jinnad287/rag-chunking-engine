"""
Smart Chunking & Retrieval System
Implements semantic, structure-aware, and overlap-based chunking strategies
and selects the best strategy for a given query.
"""

import re
import json
import math
import sys
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    top_chunks: List[str]
    similarity_scores: List[float]
    avg_score: float


# ──────────────────────────────────────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Normalize whitespace and clean up text."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Similarity Metric: TF-IDF Cosine Similarity
#
# Why TF-IDF cosine over raw word overlap (Jaccard)?
#   • Jaccard treats all shared words equally — "the", "a", "is" inflate scores.
#   • TF-IDF down-weights common words and up-weights rare, query-specific terms,
#     so a chunk that shares the query's *distinctive* vocabulary ranks higher.
#   • Cosine normalises for chunk length, preventing long chunks from
#     dominating simply because they contain more words overall.
# ──────────────────────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'if', 'so', 'not', 'no', 'than',
    'then', 'when', 'where', 'which', 'who', 'whom', 'what', 'how', 'all',
    'each', 'both', 'any', 'some', 'such', 'into', 'through', 'during',
    'about', 'also',
}


def tf(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf_scores = tf(tokens)
    return {t: tf_scores[t] * idf.get(t, 1.0) for t in tf_scores}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_idf(corpus_tokens: List[List[str]]) -> Dict[str, float]:
    """Compute IDF across a list of token-lists (one per chunk)."""
    n = len(corpus_tokens)
    doc_freq: Counter = Counter()
    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))
    return {
        term: math.log((n + 1) / (df + 1)) + 1  # smoothed IDF
        for term, df in doc_freq.items()
    }


def score_chunks(
    chunks: List[Chunk],
    query: str,
    top_k: int = 5,
) -> ChunkResult:
    """Score all chunks against a query and return top-k."""
    corpus_tokens = [
        [t for t in tokenise(c.text) if t not in STOPWORDS]
        for c in chunks
    ]
    query_tokens = [t for t in tokenise(query) if t not in STOPWORDS]

    idf = build_idf(corpus_tokens + [query_tokens])
    query_vec = tfidf_vector(query_tokens, idf)

    scored: List[Tuple[float, Chunk]] = []
    for tokens, chunk in zip(corpus_tokens, chunks):
        chunk_vec = tfidf_vector(tokens, idf)
        sim = cosine_similarity(query_vec, chunk_vec)
        scored.append((sim, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    scores = [s for s, _ in top]
    texts = [c.text for _, c in top]
    avg = sum(scores) / len(scores) if scores else 0.0

    return ChunkResult(top_chunks=texts, similarity_scores=scores, avg_score=avg)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Semantic Chunking
#   Split on sentence boundaries; merge sentences until a coherence drop is
#   detected (cosine similarity between consecutive sentence groups falls below
#   a threshold).  This keeps topically related sentences together.
# ──────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def sentence_vector(sentence: str) -> Dict[str, float]:
    tokens = [t for t in tokenise(sentence) if t not in STOPWORDS]
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def semantic_chunk(text: str, threshold: float = 0.15, max_sentences: int = 8) -> List[Chunk]:
    """
    Group sentences by semantic coherence.
    A new chunk begins when similarity to the running group drops below *threshold*.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [Chunk(text=text, index=0, strategy='semantic')]

    chunks: List[Chunk] = []
    current: List[str] = [sentences[0]]
    current_vec = sentence_vector(sentences[0])

    for sent in sentences[1:]:
        s_vec = sentence_vector(sent)
        sim = cosine_similarity(current_vec, s_vec)

        if sim < threshold or len(current) >= max_sentences:
            chunk_text = ' '.join(current)
            if chunk_text.strip():
                chunks.append(Chunk(text=chunk_text, index=len(chunks), strategy='semantic'))
            current = [sent]
            current_vec = s_vec
        else:
            current.append(sent)
            # Update running centroid (moving average of tf vectors)
            tokens = [t for t in tokenise(sent) if t not in STOPWORDS]
            for t in tokens:
                current_vec[t] = current_vec.get(t, 0) * 0.7 + (1 / max(len(tokens), 1)) * 0.3

    if current:
        chunk_text = ' '.join(current)
        if chunk_text.strip():
            chunks.append(Chunk(text=chunk_text, index=len(chunks), strategy='semantic'))

    return chunks or [Chunk(text=text, index=0, strategy='semantic')]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Structure-Aware Chunking
#   Exploit markdown / document structure: headings (# ## ###), horizontal
#   rules, numbered sections, blank-line-separated paragraphs.
#   Falls back to paragraph splitting when no structure is detected.
# ──────────────────────────────────────────────────────────────────────────────

HEADING_RE = re.compile(
    r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}:|'          # Markdown or ALL-CAPS headings
    r'\d+[\.\)]\s+[A-Z].{3,}|'                     # Numbered sections
    r'(?:Chapter|Section|Part)\s+\w+)',             # Chapter/Section/Part
    re.MULTILINE
)


def structure_aware_chunk(text: str, min_chunk_len: int = 100) -> List[Chunk]:
    """Split on structural markers; merge tiny fragments with the next chunk."""
    lines = text.split('\n')
    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_heading = ''

    def flush(heading: str, body_lines: List[str]):
        body = '\n'.join(body_lines).strip()
        if body:
            full = (f"{heading}\n{body}" if heading else body).strip()
            chunks.append(Chunk(
                text=full,
                index=len(chunks),
                strategy='structure_aware',
                metadata={'heading': heading}
            ))

    for line in lines:
        if HEADING_RE.match(line.strip()):
            flush(current_heading, current_lines)
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush(current_heading, current_lines)

    # Merge tiny chunks into the previous one
    merged: List[Chunk] = []
    for chunk in chunks:
        if merged and len(chunk.text) < min_chunk_len:
            merged[-1] = Chunk(
                text=merged[-1].text + '\n' + chunk.text,
                index=merged[-1].index,
                strategy='structure_aware',
                metadata=merged[-1].metadata,
            )
        else:
            merged.append(chunk)

    # Fallback: paragraph splitting
    if not merged:
        paragraphs = re.split(r'\n{2,}', text)
        merged = [
            Chunk(text=p.strip(), index=i, strategy='structure_aware')
            for i, p in enumerate(paragraphs)
            if len(p.strip()) >= min_chunk_len
        ]

    return merged or [Chunk(text=text, index=0, strategy='structure_aware')]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Overlap-Based Chunking
#   Fixed-size sliding window over sentences with a configurable overlap ratio.
#   Overlap ensures no query-relevant context is cut in half at a boundary.
# ──────────────────────────────────────────────────────────────────────────────

def overlap_chunk(
    text: str,
    window: int = 6,
    overlap: int = 2,
) -> List[Chunk]:
    """
    Slide a window of *window* sentences across the document,
    stepping forward by (window - overlap) sentences each time.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [Chunk(text=text, index=0, strategy='overlap')]

    step = max(1, window - overlap)
    chunks: List[Chunk] = []
    i = 0
    while i < len(sentences):
        window_sents = sentences[i: i + window]
        chunk_text = ' '.join(window_sents).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                strategy='overlap',
                metadata={'start_sent': i, 'end_sent': i + len(window_sents) - 1}
            ))
        i += step

    return chunks or [Chunk(text=text, index=0, strategy='overlap')]


# ──────────────────────────────────────────────────────────────────────────────
# Document Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_document(source: str) -> str:
    """
    Load document from:
      • a file path (.txt, .json, .pdf)
      • a raw string passed directly
    """
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == '.txt':
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.json':
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                return json.dumps(data, indent=2)
            if isinstance(data, list):
                return '\n\n'.join(
                    item if isinstance(item, str) else json.dumps(item)
                    for item in data
                )
        elif ext == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(source) as pdf:
                    return '\n\n'.join(
                        page.extract_text() or ''
                        for page in pdf.pages
                    )
            except ImportError:
                raise ImportError(
                    "pdfplumber is required for PDF loading: pip install pdfplumber"
                )
        else:
            # Try reading as plain text anyway
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
    # Treat as raw document string
    return source


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyse(
    document: str,
    query: str,
    top_k: int = 5,
    semantic_threshold: float = 0.15,
    overlap_window: int = 6,
    overlap_size: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Run all three chunking strategies, score each against the query,
    and return a unified results dictionary plus the best strategy.

    Parameters
    ----------
    document : str
        Path to a .txt / .json / .pdf file, OR the raw document text.
    query : str
        The user's retrieval query.
    top_k : int
        Number of top chunks to retrieve per strategy.
    semantic_threshold : float
        Cosine-similarity drop threshold for semantic chunking.
    overlap_window : int
        Sentence-window size for overlap chunking.
    overlap_size : int
        Number of overlapping sentences between consecutive windows.
    verbose : bool
        Print a summary to stdout.

    Returns
    -------
    dict  with keys:
        "results"         – per-strategy ChunkResult dicts
        "best_strategy"   – name of the winning strategy
        "best_chunks"     – top chunks from the best strategy
        "final_score"     – avg_score of the best strategy
    """
    raw = load_document(document)
    text = preprocess(raw)

    if not text:
        raise ValueError("Document is empty after preprocessing.")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    semantic_chunks      = semantic_chunk(text, threshold=semantic_threshold)
    structure_chunks     = structure_aware_chunk(text)
    overlap_chunks_list  = overlap_chunk(text, window=overlap_window, overlap=overlap_size)

    # ── Score ─────────────────────────────────────────────────────────────────
    sem_result  = score_chunks(semantic_chunks,   query, top_k)
    str_result  = score_chunks(structure_chunks,  query, top_k)
    ovl_result  = score_chunks(overlap_chunks_list, query, top_k)

    results = {
        "semantic": {
            "top_chunks":        sem_result.top_chunks,
            "similarity_scores": [round(s, 4) for s in sem_result.similarity_scores],
            "avg_score":         round(sem_result.avg_score, 4),
            "num_chunks":        len(semantic_chunks),
        },
        "structure_aware": {
            "top_chunks":        str_result.top_chunks,
            "similarity_scores": [round(s, 4) for s in str_result.similarity_scores],
            "avg_score":         round(str_result.avg_score, 4),
            "num_chunks":        len(structure_chunks),
        },
        "overlap": {
            "top_chunks":        ovl_result.top_chunks,
            "similarity_scores": [round(s, 4) for s in ovl_result.similarity_scores],
            "avg_score":         round(ovl_result.avg_score, 4),
            "num_chunks":        len(overlap_chunks_list),
        },
    }

    # ── Select best ───────────────────────────────────────────────────────────
    strategy_scores = {
        "semantic":        sem_result.avg_score,
        "structure_aware": str_result.avg_score,
        "overlap":         ovl_result.avg_score,
    }
    best_strategy = max(strategy_scores, key=strategy_scores.__getitem__)
    best_chunks   = results[best_strategy]["top_chunks"]
    final_score   = results[best_strategy]["avg_score"]

    output = {
        "results":        results,
        "best_strategy":  best_strategy,
        "best_chunks":    best_chunks,
        "final_score":    round(final_score, 4),
    }

    if verbose:
        _print_summary(output, query)

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Pretty Printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(output: Dict, query: str) -> None:
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  SMART CHUNKING & RETRIEVAL SYSTEM")
    print(f"{'═' * 72}")
    print(f"  Query : {query!r}")
    print(sep)

    for strategy, data in output["results"].items():
        label = strategy.replace('_', '-').title()
        print(f"\n  [{label}]  ({data['num_chunks']} chunks produced)")
        print(f"  Avg TF-IDF Cosine Score : {data['avg_score']:.4f}")
        print(f"  Top scores              : {data['similarity_scores']}")
        print(f"  Best chunk preview      :")
        preview = data["top_chunks"][0][:200].replace('\n', ' ') if data["top_chunks"] else "(none)"
        print(f"    \"{preview}{'…' if len(data['top_chunks'][0]) > 200 else ''}\"")

    print(f"\n{sep}")
    print(f"  ✦ BEST STRATEGY  : {output['best_strategy'].replace('_', '-').title()}")
    print(f"  ✦ FINAL SCORE    : {output['final_score']:.4f}")
    print(f"\n  ✦ BEST RETRIEVED CHUNKS:")
    for i, chunk in enumerate(output["best_chunks"], 1):
        preview = chunk[:300].replace('\n', ' ')
        print(f"\n  [{i}] {preview}{'…' if len(chunk) > 300 else ''}")
    print(f"\n{'═' * 72}\n")
