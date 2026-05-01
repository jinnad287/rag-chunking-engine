# 🧠 Smart Chunking & Retrieval Engine

A robust, Python-based document processing pipeline designed to evaluate and dynamically select the optimal text chunking strategy for Retrieval-Augmented Generation (RAG) systems. 

This engine parses raw documents, applies multiple chunking algorithms, and scores the resulting text blocks against user queries using a custom TF-IDF and Cosine Similarity pipeline to find the most contextually relevant information.

## ✨ Features

*   **Multi-Format Document Parsing:** Native support for parsing `.txt`, `.json`, and `.pdf` files.
*   **Dynamic Strategy Evaluation:** Automatically runs and compares three distinct chunking strategies to find the best fit for a given query:
    *   **Semantic Chunking:** Groups sentences based on topical coherence and similarity thresholds.
    *   **Structure-Aware Chunking:** Intelligently splits text based on markdown, headings, and paragraph boundaries.
    *   **Overlap-Based Chunking:** Utilizes a sliding window approach to ensure context is never lost across hard boundaries.
*   **Intelligent Scoring:** Replaces standard word-overlap (Jaccard) with normalized TF-IDF and Cosine Similarity to prioritize distinct, query-specific vocabulary.
*   **Built-in Test Suite:** Comprehensive unit testing for tokenization, vectorization, and algorithmic accuracy.

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-organization/rag-chunking-engine.git
cd rag-chunking-engine
