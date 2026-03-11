"""
Lyzr ADK Cookbook #6: Full End-to-End Pipeline
===============================================

Combines all Lyzr ADK capabilities into a complete industrial AI pipeline:

1. Document ingestion with multi-modal parsing (lyzr_parse)
2. Knowledge Graph construction with Neo4j
3. Agent creation with RAG + KG knowledge bases
4. Cognis memory for cross-session conversation continuity
5. Multi-turn session management
6. Streaming responses

This cookbook brings together Cookbooks #1-#5 into a unified workflow.

Prerequisites:
    pip install lyzr-adk

Setup:
    1. Configure Neo4j data connector in Studio UI (see Cookbook #3)
    2. Set LYZR_API_KEY environment variable

Usage:
    export LYZR_API_KEY="your-api-key"
    python 06_full_pipeline.py
"""

import os
import sys

from lyzr import Cognis, Studio


# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.environ.get("LYZR_API_KEY")
if not API_KEY:
    print("Error: Set the LYZR_API_KEY environment variable.")
    sys.exit(1)

# Placeholder document paths
EQUIPMENT_MANUAL_PDF = "data/equipment_manual.pdf"
SAFETY_STANDARDS_PDF = "data/safety_standards.pdf"
MAINTENANCE_GUIDE_DOCX = "data/maintenance_guide.docx"
OPERATING_PROCEDURES_TXT = "data/operating_procedures.txt"
DOCUMENTATION_URL = "https://example.com/industrial-docs"


# =============================================================================
# 1. Initialize All Clients
# =============================================================================

def initialize():
    """Initialize Studio and Cognis clients."""
    print("\n" + "=" * 60)
    print("  STEP 1: INITIALIZE CLIENTS")
    print("=" * 60)

    studio = Studio(api_key=API_KEY)
    cognis = Cognis(api_key=API_KEY)

    print("  Studio initialized.")
    print("  Cognis initialized.")
    return studio, cognis


# =============================================================================
# 2. Create Knowledge Bases (RAG + Knowledge Graph)
# =============================================================================

def create_knowledge_bases(studio):
    """
    Create two knowledge bases:
    - A vector RAG KB (Qdrant) for fast similarity search
    - A Knowledge Graph KB (Neo4j) for entity-relationship queries
    """
    print("\n" + "=" * 60)
    print("  STEP 2: CREATE KNOWLEDGE BASES")
    print("=" * 60)

    # --- Vector RAG Knowledge Base ---
    print("\n--- Creating Vector RAG KB (Qdrant) ---")
    rag_kb = studio.create_knowledge_base(
        name="industrial_rag_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Vector RAG KB for industrial operations documentation",
    )
    print(f"  RAG KB created: {rag_kb.name}")

    # --- Knowledge Graph KB ---
    print("\n--- Creating Knowledge Graph KB (Neo4j) ---")
    kg_kb = studio.create_knowledge_base(
        name="industrial_kg_kb",
        vector_store="neo4j",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Knowledge graph for industrial equipment relationships",
    )
    print(f"  KG KB created: {kg_kb.name}")

    return rag_kb, kg_kb


# =============================================================================
# 3. Ingest Documents into Both KBs
# =============================================================================

def ingest_documents(rag_kb, kg_kb):
    """
    Add documents to both knowledge bases:
    - RAG KB: Uses lyzr_parse for multi-modal parsing (best quality)
    - KG KB:  Uses llmsherpa for structured extraction + schema_prompt
    """
    print("\n" + "=" * 60)
    print("  STEP 3: INGEST DOCUMENTS")
    print("=" * 60)

    # --- RAG KB: Multi-modal parsing for comprehensive text extraction ---
    print("\n--- RAG KB: Adding documents with lyzr_parse (multi-modal) ---")

    rag_docs = [
        {
            "method": "add_pdf",
            "kwargs": {
                "file_path": EQUIPMENT_MANUAL_PDF,
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "data_parser": "lyzr_parse",  # Multi-modal for best quality
            },
            "label": "Equipment Manual (PDF, lyzr_parse)",
        },
        {
            "method": "add_pdf",
            "kwargs": {
                "file_path": SAFETY_STANDARDS_PDF,
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "data_parser": "llmsherpa",  # Layout-aware for structured safety docs
            },
            "label": "Safety Standards (PDF, llmsherpa)",
        },
        {
            "method": "add_docx",
            "kwargs": {
                "file_path": MAINTENANCE_GUIDE_DOCX,
                "chunk_size": 1024,
                "chunk_overlap": 128,
            },
            "label": "Maintenance Guide (DOCX)",
        },
        {
            "method": "add_txt",
            "kwargs": {
                "file_path": OPERATING_PROCEDURES_TXT,
                "chunk_size": 512,
                "chunk_overlap": 64,
            },
            "label": "Operating Procedures (TXT)",
        },
    ]

    for doc in rag_docs:
        try:
            method = getattr(rag_kb, doc["method"])
            result = method(**doc["kwargs"])
            print(f"  Added to RAG KB: {doc['label']}")
        except Exception as e:
            print(f"  RAG KB ({doc['label']}): {e}")

    # Add website content
    print("\n  Adding website content to RAG KB...")
    try:
        rag_kb.add_website(
            url=DOCUMENTATION_URL,
            max_pages=20,
            max_depth=2,
            chunk_size=1024,
            chunk_overlap=128,
        )
        print(f"  Added website: {DOCUMENTATION_URL}")
    except Exception as e:
        print(f"  Website: {e}")

    # --- KG KB: Schema-guided extraction for entity/relationship graph ---
    print("\n--- KG KB: Adding documents with schema-guided extraction ---")

    industrial_schema = (
        "Entity types: Equipment, Component, Facility, Manufacturer, "
        "MaintenanceProcedure, SafetyStandard, Sensor, FailureMode, "
        "SparePartNumber, OperatingParameter. "
        "Relationship types: HAS_COMPONENT, MANUFACTURED_BY, LOCATED_IN, "
        "REQUIRES_MAINTENANCE, FOLLOWS_STANDARD, MONITORS, CAN_CAUSE_FAILURE, "
        "USES_PART, HAS_PARAMETER, DEPENDS_ON, TRIGGERS_ALERT."
    )

    kg_docs = [
        {
            "file_path": EQUIPMENT_MANUAL_PDF,
            "label": "Equipment Manual",
        },
        {
            "file_path": SAFETY_STANDARDS_PDF,
            "label": "Safety Standards",
        },
    ]

    for doc in kg_docs:
        try:
            kg_kb.add_pdf(
                file_path=doc["file_path"],
                chunk_size=1024,
                chunk_overlap=128,
                data_parser="llmsherpa",
                extra_info=f'{{"schema_prompt": "{industrial_schema}"}}',
            )
            print(f"  Added to KG KB: {doc['label']}")
        except Exception as e:
            print(f"  KG KB ({doc['label']}): {e}")

    # Add structured text to KG
    try:
        kg_kb.add_text(
            text=(
                "The Siemens SGT-800 gas turbine at Plant Alpha has a 57 MW output "
                "and requires major overhaul every 32,000 hours. It is monitored by "
                "vibration sensor VS-801 (threshold: 4.5 mm/s) and temperature sensor "
                "TS-802 (max: 650C). Maintenance procedure MP-001 covers routine "
                "inspection, while MP-002 covers emergency shutdown procedures."
            ),
            source="structured_equipment_data",
            chunk_size=512,
            chunk_overlap=64,
        )
        print(f"  Added structured text to KG KB.")
    except Exception as e:
        print(f"  KG KB (text): {e}")

    return rag_kb, kg_kb


# =============================================================================
# 4. Create the Agent
# =============================================================================

def create_agent(studio):
    """
    Create an intelligent agent that can leverage both RAG and KG knowledge bases.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: CREATE AGENT")
    print("=" * 60)

    agent = studio.create_agent(
        name="IndustrialAI",
        provider="gpt-4o",
        role="Senior Industrial Operations AI Assistant",
        goal=(
            "Provide comprehensive support for industrial operations including "
            "equipment maintenance, safety compliance, troubleshooting, and "
            "operational optimization. Leverage knowledge bases for accurate, "
            "evidence-based answers."
        ),
        instructions=(
            "You are an expert AI assistant for industrial operations at a large "
            "manufacturing facility. You have access to equipment manuals, safety "
            "standards, maintenance guides, and a knowledge graph of equipment "
            "relationships.\n\n"
            "Guidelines:\n"
            "- Always cite sources from the knowledge base when providing answers\n"
            "- For equipment questions, trace component relationships in the KG\n"
            "- For safety questions, reference specific standards and regulations\n"
            "- Provide actionable, step-by-step recommendations\n"
            "- Flag any safety concerns proactively\n"
            "- If information is not in the knowledge base, clearly state the limitation"
        ),
        temperature=0.2,
    )
    print(f"  Agent created: {agent.name}")
    print(f"  Role: {agent.role}")
    return agent


# =============================================================================
# 5. Run Agent with Dual Knowledge Bases
# =============================================================================

def run_dual_kb_queries(agent, rag_kb, kg_kb):
    """
    Run agent queries that leverage both RAG (vector search) and KG (graph traversal)
    knowledge bases simultaneously.
    """
    print("\n" + "=" * 60)
    print("  STEP 5: DUAL-KB AGENT QUERIES")
    print("=" * 60)

    queries = [
        {
            "message": (
                "What maintenance procedures apply to the SGT-800 gas turbine? "
                "Include component-level details and related safety standards."
            ),
            "label": "Equipment Maintenance (multi-hop)",
        },
        {
            "message": (
                "A vibration sensor is reading 5.8 mm/s on a turbine bearing. "
                "What should we do? What safety protocols apply?"
            ),
            "label": "Troubleshooting + Safety",
        },
        {
            "message": (
                "Generate a maintenance schedule for the next quarter. "
                "Prioritize by equipment criticality and failure risk."
            ),
            "label": "Planning + Analysis",
        },
    ]

    session_id = None

    for q in queries:
        print(f"\n--- {q['label']} ---")
        print(f"  Q: {q['message']}")

        try:
            response = agent.run(
                message=q["message"],
                session_id=session_id,  # Maintain conversation context
                knowledge_bases=[
                    # RAG KB: diverse retrieval with high top_k
                    rag_kb.with_config(
                        top_k=10,
                        retrieval_type="mmr",
                        score_threshold=0.4,
                    ),
                    # KG KB: relationship-aware retrieval
                    kg_kb.with_config(
                        top_k=10,
                        retrieval_type="basic",
                        score_threshold=0.3,
                    ),
                ],
            )

            session_id = response.session_id  # Carry session across queries
            print(f"\n  A: {response.response[:500]}...")
            print(f"  Session: {response.session_id}")

        except Exception as e:
            print(f"  Error: {e}")

    return session_id


# =============================================================================
# 6. Cognis Memory — Save and Recall
# =============================================================================

def memory_workflow(cognis, agent, rag_kb, kg_kb):
    """
    Use Cognis memory to:
    1. Save the current conversation for future reference
    2. Retrieve past memories to enrich new conversations
    """
    print("\n" + "=" * 60)
    print("  STEP 6: COGNIS MEMORY WORKFLOW")
    print("=" * 60)

    owner_id = "plant_alpha_team"

    # --- Save past interaction as memory ---
    print("\n--- Saving conversation to Cognis ---")
    try:
        cognis.add(
            messages=[
                {
                    "role": "user",
                    "content": "The SGT-800 turbine at Plant Alpha had a vibration alert at 5.8 mm/s on bearing 3.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "This exceeds the 4.5 mm/s threshold. Immediate actions: "
                        "1) Reduce load to 70%, 2) Schedule bearing inspection within 48 hours, "
                        "3) Check lubrication system. Root cause is likely bearing wear or "
                        "misalignment. Ref: MP-001 Section 4.3."
                    ),
                },
                {
                    "role": "user",
                    "content": "We found the bearing had excessive wear. Replaced it and vibration is now at 1.9 mm/s.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Good — 1.9 mm/s is well within normal range. I recommend: "
                        "1) Monitor daily for the next 2 weeks, 2) Update the CMMS record, "
                        "3) Reduce the bearing replacement interval from 16,000 to 12,000 hours "
                        "for this unit based on this early failure."
                    ),
                },
            ],
            owner_id=owner_id,
            agent_id="industrial_ai",
            session_id="incident_turbine_001",
        )
        print("  Conversation saved to Cognis.")
    except Exception as e:
        print(f"  Save error: {e}")

    # --- Retrieve memories for a new conversation ---
    print("\n--- Retrieving past memories for new conversation ---")
    try:
        memories = cognis.search(
            query="turbine bearing issues and maintenance history",
            owner_id=owner_id,
            limit=5,
            cross_session=True,
        )

        if memories:
            print(f"  Found {len(memories)} relevant memories:")
            for i, m in enumerate(memories, 1):
                print(f"  [{i}] {m.content[:150]}...")

            # Build memory context for the agent
            memory_context = "Relevant history from past conversations:\n"
            for m in memories:
                memory_context += f"- {m.content}\n"

            # Run agent with memory-enriched context
            print("\n--- Running agent with memory context ---")
            response = agent.run(
                message=(
                    f"{memory_context}\n\n"
                    "Current question: It's been 3 months since we replaced the bearing "
                    "on the SGT-800 turbine. Vibration readings have been stable at 2.0 mm/s. "
                    "Should we adjust our monitoring frequency?"
                ),
                knowledge_bases=[
                    rag_kb.with_config(top_k=5, retrieval_type="basic"),
                    kg_kb.with_config(top_k=5, retrieval_type="basic"),
                ],
            )
            print(f"\n  Agent Response (memory-enriched):")
            print(f"  {response.response[:500]}...")

            # Save this new interaction too
            cognis.add(
                messages=[
                    {
                        "role": "user",
                        "content": "3 months post bearing replacement, vibration stable at 2.0 mm/s. Adjust monitoring?",
                    },
                    {"role": "assistant", "content": response.response},
                ],
                owner_id=owner_id,
                agent_id="industrial_ai",
                session_id="followup_turbine_001",
            )
            print("\n  Follow-up saved to Cognis.")
        else:
            print("  No past memories found.")

    except Exception as e:
        print(f"  Memory workflow error: {e}")


# =============================================================================
# 7. Streaming Response Demo
# =============================================================================

def streaming_demo(agent, rag_kb, kg_kb):
    """
    Demonstrate streaming responses for real-time output.
    """
    print("\n" + "=" * 60)
    print("  STEP 7: STREAMING RESPONSE")
    print("=" * 60)

    print("\n  Q: Create a comprehensive maintenance checklist for the SGT-800.\n")
    print("  Streaming response:\n  ", end="")

    try:
        response = agent.run(
            message=(
                "Create a comprehensive daily, weekly, and monthly maintenance "
                "checklist for the SGT-800 gas turbine. Include specific measurements, "
                "thresholds, and references to procedures."
            ),
            knowledge_bases=[
                rag_kb.with_config(top_k=15, retrieval_type="mmr"),
                kg_kb.with_config(top_k=10, retrieval_type="basic"),
            ],
            stream=True,
        )

        if hasattr(response, "__iter__"):
            for chunk in response:
                print(chunk, end="", flush=True)
            print()
        else:
            print(response.response)

    except Exception as e:
        print(f"\n  Streaming error: {e}")


# =============================================================================
# 8. Pipeline Summary
# =============================================================================

def print_summary():
    """Print a summary of the complete pipeline."""
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print("""
  This pipeline demonstrated:

  1. DOCUMENT INGESTION
     - PDF parsing with lyzr_parse (multi-modal) and llmsherpa (layout-aware)
     - DOCX, TXT, and website ingestion
     - Schema-guided extraction for knowledge graphs

  2. DUAL KNOWLEDGE BASES
     - Vector RAG (Qdrant): Fast similarity search with MMR diversity
     - Knowledge Graph (Neo4j): Entity-relationship traversal

  3. INTELLIGENT AGENT
     - Role-based agent with domain expertise
     - Dual-KB queries combining vector + graph retrieval
     - Multi-turn session management

  4. COGNIS MEMORY
     - Conversation memory persistence
     - Cross-session recall for continuity
     - Memory-enriched agent responses

  5. STREAMING
     - Real-time response streaming for interactive applications

  Architecture:
     Documents → [lyzr_parse/llmsherpa] → Qdrant (vectors) + Neo4j (graph)
                                              ↓                    ↓
                                         Agent (GPT-4o) ←── Dual KB Retrieval
                                              ↓
                                    Cognis Memory ←→ Cross-Session Recall
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #6: Full End-to-End Pipeline")
    print("=" * 60)
    print()
    print("  This cookbook combines RAG, Knowledge Graph, Agent, and Memory")
    print("  into a complete industrial AI pipeline.")
    print()

    # Step 1: Initialize
    studio, cognis = initialize()

    # Step 2: Create knowledge bases
    rag_kb, kg_kb = create_knowledge_bases(studio)

    # Step 3: Ingest documents
    ingest_documents(rag_kb, kg_kb)

    # Step 4: Create agent
    agent = create_agent(studio)

    # Step 5: Run dual-KB queries
    run_dual_kb_queries(agent, rag_kb, kg_kb)

    # Step 6: Cognis memory workflow
    memory_workflow(cognis, agent, rag_kb, kg_kb)

    # Step 7: Streaming demo
    streaming_demo(agent, rag_kb, kg_kb)

    # Summary
    print_summary()

    print("=" * 60)
    print("  Cookbook #6 complete! Full pipeline demonstrated.")
    print("=" * 60)
