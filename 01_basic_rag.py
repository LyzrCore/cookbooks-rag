"""
Lyzr ADK Cookbook #1: Basic RAG Pipeline
========================================

Demonstrates the complete RAG (Retrieval-Augmented Generation) workflow using
Lyzr's ADK SDK, including:

- Knowledge base creation with Qdrant vector store
- Document ingestion with all 3 PDF parsers (pymupdf, llmsherpa, lyzr_parse)
- Adding DOCX, TXT, website, and raw text sources
- Querying with all 4 retrieval strategies (basic, mmr, hyde, time_aware)
- Fine-tuning retrieval parameters

Prerequisites:
    pip install lyzr-adk

Usage:
    export LYZR_API_KEY="your-api-key"
    python 01_basic_rag.py
"""

import os
import sys

from lyzr import Studio


# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.environ.get("LYZR_API_KEY")
if not API_KEY:
    print("Error: Set the LYZR_API_KEY environment variable.")
    sys.exit(1)

# Placeholder document paths — replace with your actual files
PDF_PATH = "data/sample_document.pdf"
DOCX_PATH = "data/sample_document.docx"
TXT_PATH = "data/sample_document.txt"
WEBSITE_URL = "https://example.com/documentation"


# =============================================================================
# 1. Initialize Studio
# =============================================================================

def init_studio():
    """Initialize the Lyzr Studio client."""
    studio = Studio(api_key=API_KEY)
    print("Studio initialized successfully.")
    return studio


# =============================================================================
# 2. Create a Knowledge Base
# =============================================================================

def create_knowledge_base(studio):
    """
    Create a knowledge base with Qdrant vector store and OpenAI embeddings.

    Supported vector stores: qdrant, weaviate, pg_vector, milvus, neptune
    """
    kb = studio.create_knowledge_base(
        name="industrial_docs_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Industrial documentation knowledge base for RAG cookbook demo",
    )
    print(f"Knowledge base created: {kb.name}")
    return kb


# =============================================================================
# 3. Add Documents — All Supported Formats
# =============================================================================

def add_documents(kb):
    """
    Add documents to the knowledge base using different parsers and formats.

    PDF Parsers:
        - pymupdf:    Fast, text-only extraction. Best for clean, text-heavy PDFs.
        - llmsherpa:  Layout-aware parsing. Preserves tables, headers, sections.
                      Default parser if none specified.
        - lyzr_parse: Multi-modal (Docling + VLM). Extracts text, tables, images
                      with AI-powered visual understanding. Best for complex docs.
    """

    # --- 3a. PDF with pymupdf (fast, text-only) ---
    print("\n--- Adding PDF with pymupdf parser ---")
    try:
        result = kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="pymupdf",
        )
        print(f"  Added with pymupdf: {result}")
    except Exception as e:
        print(f"  pymupdf upload: {e}")

    # --- 3b. PDF with llmsherpa (layout-aware, default) ---
    print("\n--- Adding PDF with llmsherpa parser ---")
    try:
        result = kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
        )
        print(f"  Added with llmsherpa: {result}")
    except Exception as e:
        print(f"  llmsherpa upload: {e}")

    # --- 3c. PDF with lyzr_parse (multi-modal: Docling + VLM) ---
    print("\n--- Adding PDF with lyzr_parse parser (multi-modal) ---")
    try:
        result = kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="lyzr_parse",
        )
        print(f"  Added with lyzr_parse: {result}")
    except Exception as e:
        print(f"  lyzr_parse upload: {e}")

    # --- 3d. DOCX ---
    print("\n--- Adding DOCX document ---")
    try:
        result = kb.add_docx(
            file_path=DOCX_PATH,
            chunk_size=1024,
            chunk_overlap=128,
        )
        print(f"  Added DOCX: {result}")
    except Exception as e:
        print(f"  DOCX upload: {e}")

    # --- 3e. TXT ---
    print("\n--- Adding TXT document ---")
    try:
        result = kb.add_txt(
            file_path=TXT_PATH,
            chunk_size=512,
            chunk_overlap=64,
        )
        print(f"  Added TXT: {result}")
    except Exception as e:
        print(f"  TXT upload: {e}")

    # --- 3f. Website ---
    print("\n--- Adding website content ---")
    try:
        result = kb.add_website(
            url=WEBSITE_URL,
            max_pages=10,
            max_depth=2,
            chunk_size=1024,
            chunk_overlap=128,
        )
        print(f"  Added website: {result}")
    except Exception as e:
        print(f"  Website upload: {e}")

    # --- 3g. Raw text ---
    print("\n--- Adding raw text ---")
    try:
        result = kb.add_text(
            text=(
                "Predictive maintenance uses sensor data and machine learning "
                "to forecast equipment failures before they occur, reducing "
                "unplanned downtime by up to 50%."
            ),
            source="manual_entry",
            chunk_size=512,
            chunk_overlap=64,
        )
        print(f"  Added raw text: {result}")
    except Exception as e:
        print(f"  Raw text upload: {e}")


# =============================================================================
# 4. Query — All 4 Retrieval Strategies
# =============================================================================

def print_results(results, label):
    """Pretty-print query results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Score: {r.score:.4f}")
        print(f"      Source: {r.source}")
        print(f"      Text:   {r.text[:200]}...")
        if r.metadata:
            print(f"      Meta:   {r.metadata}")


def query_basic(kb):
    """
    Basic retrieval — standard cosine similarity search.
    Returns the top_k most similar chunks by embedding distance.
    """
    results = kb.query(
        query="What are the maintenance procedures for industrial turbines?",
        top_k=5,
        retrieval_type="basic",
        score_threshold=0.0,  # 0.0 = no threshold filtering
    )
    print_results(results, "BASIC Retrieval (Cosine Similarity)")
    return results


def query_mmr(kb):
    """
    MMR (Maximal Marginal Relevance) — balances relevance with diversity.
    Avoids returning near-duplicate chunks.

    lambda_param: 0.0 = max diversity, 1.0 = max relevance (default: 0.5)
    """
    results = kb.query(
        query="What are the maintenance procedures for industrial turbines?",
        top_k=5,
        retrieval_type="mmr",
        score_threshold=0.3,
        lambda_param=0.6,  # Slightly favor relevance over diversity
    )
    print_results(results, "MMR Retrieval (Relevance + Diversity)")
    return results


def query_hyde(kb):
    """
    HyDE (Hypothetical Document Embeddings) — generates a hypothetical
    answer to the query, then searches using that answer's embedding.
    Improves retrieval for complex or abstract questions.
    """
    results = kb.query(
        query="How can AI improve predictive maintenance in manufacturing?",
        top_k=5,
        retrieval_type="hyde",
        score_threshold=0.3,
    )
    print_results(results, "HyDE Retrieval (Hypothetical Document Embeddings)")
    return results


def query_time_aware(kb):
    """
    Time-aware retrieval — boosts recent documents using exponential decay.
    Useful for evolving knowledge bases where newer info is more relevant.

    time_decay_factor: Higher = faster decay of old documents.
        0.1 = gentle decay (old docs still relevant)
        1.0 = aggressive decay (strongly prefer recent docs)
    """
    results = kb.query(
        query="Latest safety standards for industrial equipment",
        top_k=5,
        retrieval_type="time_aware",
        score_threshold=0.2,
        time_decay_factor=0.3,
    )
    print_results(results, "TIME-AWARE Retrieval (Recency-Boosted)")
    return results


# =============================================================================
# 5. Parameter Tuning Examples
# =============================================================================

def parameter_tuning_examples(kb):
    """Demonstrate how different parameter combinations affect results."""

    print("\n" + "=" * 60)
    print("  PARAMETER TUNING EXAMPLES")
    print("=" * 60)

    # High precision: strict threshold, fewer results
    print("\n--- High Precision (strict threshold) ---")
    results = kb.query(
        query="Equipment failure root cause analysis",
        top_k=3,
        retrieval_type="basic",
        score_threshold=0.7,  # Only return highly relevant chunks
    )
    print(f"  Results with score >= 0.7: {len(results)} chunks")

    # High recall: relaxed threshold, more results
    print("\n--- High Recall (relaxed threshold) ---")
    results = kb.query(
        query="Equipment failure root cause analysis",
        top_k=20,
        retrieval_type="basic",
        score_threshold=0.1,  # Return more, less-relevant chunks
    )
    print(f"  Results with score >= 0.1: {len(results)} chunks")

    # MMR with max diversity
    print("\n--- MMR with Maximum Diversity ---")
    results = kb.query(
        query="Equipment failure root cause analysis",
        top_k=5,
        retrieval_type="mmr",
        lambda_param=0.2,  # Strong diversity preference
    )
    print(f"  Diverse results: {len(results)} chunks")

    # MMR with max relevance (behaves like basic)
    print("\n--- MMR with Maximum Relevance ---")
    results = kb.query(
        query="Equipment failure root cause analysis",
        top_k=5,
        retrieval_type="mmr",
        lambda_param=1.0,  # Pure relevance, no diversity
    )
    print(f"  Relevant results: {len(results)} chunks")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #1: Basic RAG Pipeline")
    print("=" * 60)

    # Step 1: Initialize
    studio = init_studio()

    # Step 2: Create knowledge base
    kb = create_knowledge_base(studio)

    # Step 3: Add documents with different parsers
    add_documents(kb)

    # Step 4: Query with all retrieval strategies
    query_basic(kb)
    query_mmr(kb)
    query_hyde(kb)
    query_time_aware(kb)

    # Step 5: Parameter tuning
    parameter_tuning_examples(kb)

    print("\n" + "=" * 60)
    print("  Cookbook #1 complete!")
    print("=" * 60)
