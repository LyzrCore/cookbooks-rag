# Lyzr ADK — RAG Cookbooks

A collection of hands-on Python cookbooks demonstrating **Retrieval-Augmented Generation (RAG)** pipelines using the [Lyzr ADK](https://pypi.org/project/lyzr-adk/) SDK. Each cookbook is a standalone script that progressively introduces more advanced capabilities — from basic vector search to full end-to-end pipelines with knowledge graphs and persistent memory.

## Cookbooks

| # | File | Description |
|---|------|-------------|
| 01 | `01_basic_rag.py` | **Basic RAG Pipeline** — Knowledge base creation, document ingestion (PDF, DOCX, TXT, website), and querying with multiple retrieval strategies |
| 02 | `02_agentic_rag.py` | **Agentic RAG** — AI agents with single/multi-KB configurations, custom retrieval settings, multi-turn sessions, and streaming |
| 03 | `03_knowledge_graph.py` | **Knowledge Graph (Neo4j)** — Entity/relationship extraction, schema-guided ingestion, and graph-powered agent answers |
| 04 | `04_multimodal_parsing.py` | **Multi-Modal Parsing** — Side-by-side comparison of all 3 PDF parsers, chunking strategies, and parser selection guide |
| 05 | `05_cognis_memory.py` | **Cognis Memory** — Persistent semantic memory across sessions, scoped retrieval, memory management, and async patterns |
| 06 | `06_full_pipeline.py` | **Full End-to-End Pipeline** — Combines all capabilities: multi-modal parsing, knowledge graph, agentic RAG, Cognis memory, and streaming |

## Architecture

```
                        ┌──────────────┐
                        │  Lyzr Studio │
                        │   (Cloud)    │
                        └──────┬───────┘
                               │ LYZR_API_KEY
          ┌────────────────────┼────────────────────┐
          │                    │                     │
   ┌──────▼──────┐     ┌──────▼──────┐      ┌──────▼──────┐
   │  Knowledge  │     │    Agent    │      │   Cognis    │
   │    Base     │     │   Engine    │      │   Memory    │
   └──────┬──────┘     └──────┬──────┘      └─────────────┘
          │                   │
    ┌─────┼─────┐             │
    │     │     │             │
 ┌──▼─┐ ┌▼──┐ ┌▼────┐  ┌────▼────┐
 │Qdr.│ │Neo│ │Parse│  │Sessions │
 │ant │ │4j │ │ rs  │  │Streaming│
 └────┘ └───┘ └─────┘  └─────────┘
```

## Key Capabilities

**Retrieval Strategies:** `basic` · `mmr` (maximal marginal relevance) · `hyde` (hypothetical document embeddings) · `time_aware`

**Document Parsers:** `pymupdf` (fast text extraction) · `llmsherpa` (layout-aware) · `lyzr_parse` (multi-modal)

**Supported Formats:** PDF · DOCX · TXT · Websites · Raw text

**Vector Stores:** Qdrant (default) · Neo4j (knowledge graph)

## Prerequisites

- **Python 3.8+**
- **lyzr-adk** — `pip install lyzr-adk`
- **LYZR_API_KEY** — obtain from [Lyzr Studio](https://studio.lyzr.ai)
- **Neo4j Aura** — required for cookbooks 03 and 06 ([console.neo4j.io](https://console.neo4j.io))
- **Cognis** — used in cookbooks 05 and 06 (included in lyzr-adk, no extra setup)

> **Note:** No `OPENAI_API_KEY` is needed — LLM access is managed through Lyzr Studio.

## Quick Start

```bash
# Install the SDK
pip install lyzr-adk

# Set your API key
export LYZR_API_KEY="your-api-key"

# Run any cookbook
python 01_basic_rag.py
```

## License

This project is licensed under the MIT License.
