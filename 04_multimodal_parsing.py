"""
Lyzr ADK Cookbook #4: Multi-Modal Document Parsing
===================================================

Demonstrates advanced document parsing with Lyzr's 3 PDF parsers,
focusing on the multi-modal lyzr_parse parser (Docling + VLM):

- Side-by-side parser comparison (pymupdf vs llmsherpa vs lyzr_parse)
- Multi-modal parsing with image description and table extraction
- Chunking strategy comparison (small vs large chunks)
- Optimal parser selection guide for different document types

Prerequisites:
    pip install lyzr-adk

Usage:
    export LYZR_API_KEY="your-api-key"
    python 04_multimodal_parsing.py
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

# Use the same PDF for comparison across all parsers
PDF_PATH = "data/sample_document.pdf"
# A PDF with tables and images for multi-modal testing
COMPLEX_PDF_PATH = "data/complex_report_with_tables_and_images.pdf"
DOCX_PATH = "data/sample_document.docx"


# =============================================================================
# Parser Overview
# =============================================================================
#
# Lyzr supports 3 PDF parsers, each with different strengths:
#
# +------------+------------------+------------------------------------------+
# | Parser     | Method           | Best For                                 |
# +------------+------------------+------------------------------------------+
# | pymupdf    | Text extraction  | Clean, text-heavy PDFs. Fastest parser.  |
# |            |                  | No layout awareness.                     |
# +------------+------------------+------------------------------------------+
# | llmsherpa  | Layout-aware     | Documents with tables, headers, sections.|
# |            |                  | Preserves document structure. Default.   |
# +------------+------------------+------------------------------------------+
# | lyzr_parse | Docling + VLM    | Complex docs with images, charts, tables.|
# |            |                  | AI-powered visual understanding.         |
# |            |                  | Highest quality, slowest.                |
# +------------+------------------+------------------------------------------+
#


# =============================================================================
# 1. Parser Comparison: Same Document, 3 Parsers
# =============================================================================

def compare_parsers(studio):
    """
    Process the same PDF with all 3 parsers and compare the results.
    Each parser creates a separate KB for isolated comparison.
    """
    print("\n" + "=" * 60)
    print("  PARSER COMPARISON")
    print("=" * 60)

    parsers = ["pymupdf", "llmsherpa", "lyzr_parse"]
    results = {}

    for parser in parsers:
        print(f"\n--- Parser: {parser} ---")

        # Create a dedicated KB for each parser
        kb = studio.create_knowledge_base(
            name=f"parser_comparison_{parser}",
            vector_store="qdrant",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            description=f"Parser comparison KB using {parser}",
        )

        # Add the same PDF with the specific parser
        try:
            result = kb.add_pdf(
                file_path=PDF_PATH,
                chunk_size=1024,
                chunk_overlap=128,
                data_parser=parser,
            )
            print(f"  Upload result: {result}")
        except Exception as e:
            print(f"  Upload error: {e}")
            continue

        # Query for a specific topic to compare retrieval quality
        query = "What are the key maintenance procedures described in this document?"
        try:
            search_results = kb.query(
                query=query,
                top_k=3,
                retrieval_type="basic",
            )
            results[parser] = search_results

            print(f"  Query: '{query}'")
            print(f"  Results: {len(search_results)} chunks")
            if search_results:
                print(f"  Top result (score={search_results[0].score:.4f}):")
                print(f"    {search_results[0].text[:200]}...")
        except Exception as e:
            print(f"  Query error: {e}")

    # Summary comparison
    print("\n" + "-" * 60)
    print("  COMPARISON SUMMARY")
    print("-" * 60)
    for parser, res in results.items():
        if res:
            avg_score = sum(r.score for r in res) / len(res)
            print(f"  {parser:12s} | {len(res)} results | avg score: {avg_score:.4f}")
        else:
            print(f"  {parser:12s} | No results")

    return results


# =============================================================================
# 2. Multi-Modal Parsing with lyzr_parse (Docling + VLM)
# =============================================================================

def multimodal_parsing(studio):
    """
    Demonstrate lyzr_parse's multi-modal capabilities:
    - AI-powered table extraction
    - Image/chart description via Vision Language Model (VLM)
    - Layout-aware text segmentation
    """
    print("\n" + "=" * 60)
    print("  MULTI-MODAL PARSING (lyzr_parse)")
    print("=" * 60)

    kb = studio.create_knowledge_base(
        name="multimodal_demo",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Multi-modal parsing demonstration",
    )

    # --- Standard lyzr_parse ---
    print("\n--- Standard lyzr_parse (Docling + VLM) ---")
    try:
        result = kb.add_pdf(
            file_path=COMPLEX_PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="lyzr_parse",
        )
        print(f"  Upload result: {result}")
    except Exception as e:
        print(f"  Upload: {e}")

    # Query for table content
    print("\n--- Querying for table data ---")
    try:
        results = kb.query(
            query="Show me the data from tables in the document",
            top_k=5,
            retrieval_type="basic",
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] Score: {r.score:.4f}")
                print(f"      Text: {r.text[:250]}...")
        else:
            print("  No table data found.")
    except Exception as e:
        print(f"  Query error: {e}")

    # Query for image descriptions
    print("\n--- Querying for image/chart descriptions ---")
    try:
        results = kb.query(
            query="Describe the charts, figures, or images in the document",
            top_k=5,
            retrieval_type="basic",
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] Score: {r.score:.4f}")
                print(f"      Text: {r.text[:250]}...")
        else:
            print("  No image descriptions found.")
    except Exception as e:
        print(f"  Query error: {e}")

    return kb


# =============================================================================
# 3. Chunking Strategy Comparison
# =============================================================================

def chunking_comparison(studio):
    """
    Compare different chunking parameters to understand their impact on
    retrieval quality and granularity.

    chunk_size:    Number of characters per chunk. Smaller = more granular.
    chunk_overlap: Overlap between consecutive chunks. Prevents context loss
                   at chunk boundaries.
    """
    print("\n" + "=" * 60)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    strategies = [
        {
            "name": "small_chunks",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "description": "Small chunks (256 chars) — high granularity, may lose context",
        },
        {
            "name": "medium_chunks",
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "description": "Medium chunks (1024 chars) — balanced (recommended default)",
        },
        {
            "name": "large_chunks",
            "chunk_size": 2048,
            "chunk_overlap": 256,
            "description": "Large chunks (2048 chars) — more context, less granular",
        },
    ]

    query = "What are the safety requirements for high-pressure systems?"
    results_map = {}

    for strategy in strategies:
        print(f"\n--- {strategy['description']} ---")

        kb = studio.create_knowledge_base(
            name=f"chunking_{strategy['name']}",
            vector_store="qdrant",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            description=f"Chunking test: {strategy['name']}",
        )

        try:
            kb.add_pdf(
                file_path=PDF_PATH,
                chunk_size=strategy["chunk_size"],
                chunk_overlap=strategy["chunk_overlap"],
                data_parser="llmsherpa",
            )
        except Exception as e:
            print(f"  Upload: {e}")
            continue

        try:
            results = kb.query(
                query=query,
                top_k=5,
                retrieval_type="basic",
            )
            results_map[strategy["name"]] = results

            print(f"  Query: '{query}'")
            print(f"  Results: {len(results)} chunks")
            if results:
                print(f"  Top score: {results[0].score:.4f}")
                print(f"  Top chunk length: {len(results[0].text)} chars")
                print(f"  Preview: {results[0].text[:150]}...")
        except Exception as e:
            print(f"  Query error: {e}")

    # Summary
    print("\n" + "-" * 60)
    print("  CHUNKING SUMMARY")
    print("-" * 60)
    print(f"  {'Strategy':15s} | {'Chunks':6s} | {'Top Score':9s} | {'Avg Chunk Len':13s}")
    print(f"  {'-'*15} | {'-'*6} | {'-'*9} | {'-'*13}")
    for name, res in results_map.items():
        if res:
            avg_len = sum(len(r.text) for r in res) / len(res)
            print(f"  {name:15s} | {len(res):6d} | {res[0].score:9.4f} | {avg_len:13.0f}")


# =============================================================================
# 4. DOCX Parsing
# =============================================================================

def docx_parsing(studio):
    """
    Parse DOCX files. DOCX parsing preserves:
    - Headings and paragraph structure
    - Tables (as text)
    - Lists and bullet points
    """
    print("\n" + "=" * 60)
    print("  DOCX PARSING")
    print("=" * 60)

    kb = studio.create_knowledge_base(
        name="docx_parsing_demo",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="DOCX parsing demonstration",
    )

    try:
        result = kb.add_docx(
            file_path=DOCX_PATH,
            chunk_size=1024,
            chunk_overlap=128,
        )
        print(f"  DOCX uploaded: {result}")
    except Exception as e:
        print(f"  DOCX upload: {e}")

    try:
        results = kb.query(
            query="What are the main topics covered in this document?",
            top_k=5,
            retrieval_type="basic",
        )
        print(f"\n  Query results: {len(results)} chunks")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score: {r.score:.4f} | {r.text[:150]}...")
    except Exception as e:
        print(f"  Query error: {e}")


# =============================================================================
# 5. Parser Selection Guide
# =============================================================================

def print_parser_guide():
    """Print a quick reference for parser selection."""
    print("\n" + "=" * 60)
    print("  PARSER SELECTION GUIDE")
    print("=" * 60)
    print("""
  Use this guide to choose the right parser for your documents:

  pymupdf:
    + Fastest processing speed
    + Good for clean, text-only PDFs
    + Lowest resource usage
    - No layout awareness
    - Cannot handle tables or images
    Best for: Text reports, articles, plain documentation

  llmsherpa (DEFAULT):
    + Layout-aware parsing
    + Preserves tables, headers, sections
    + Good balance of speed and quality
    - No image understanding
    Best for: Structured documents, manuals, forms with tables

  lyzr_parse (MULTI-MODAL):
    + AI-powered visual understanding (Docling + VLM)
    + Extracts and describes images, charts, diagrams
    + Best table extraction accuracy
    + Handles complex multi-column layouts
    - Slowest processing speed
    - Higher resource usage
    Best for: Engineering drawings, reports with charts/images,
              scanned documents, complex layouts

  Rule of thumb:
    - Text-only docs        → pymupdf
    - Docs with tables      → llmsherpa
    - Docs with images/charts → lyzr_parse
    - Not sure              → llmsherpa (safe default)
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #4: Multi-Modal Document Parsing")
    print("=" * 60)

    studio = Studio(api_key=API_KEY)

    # Demo 1: Compare all 3 parsers on the same document
    compare_parsers(studio)

    # Demo 2: Multi-modal parsing deep dive
    multimodal_parsing(studio)

    # Demo 3: Chunking strategy comparison
    chunking_comparison(studio)

    # Demo 4: DOCX parsing
    docx_parsing(studio)

    # Reference: Parser selection guide
    print_parser_guide()

    print("\n" + "=" * 60)
    print("  Cookbook #4 complete!")
    print("=" * 60)
