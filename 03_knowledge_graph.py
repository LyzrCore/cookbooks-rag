"""
Lyzr ADK Cookbook #3: Knowledge Graph with Neo4j
=================================================

Demonstrates building and querying knowledge graphs using Lyzr's ADK SDK
with Neo4j as the graph database backend, including:

- Setting up Neo4j credentials via Studio UI
- Creating a knowledge base with Neo4j vector store
- Adding documents for automatic entity/relationship extraction
- Using schema_prompt for guided extraction (industrial domain)
- Querying the knowledge graph
- Adding website and text sources to the graph
- Combining KG with an agent for graph-powered answers

Prerequisites:
    pip install lyzr-adk

Neo4j Setup:
    You need a Neo4j Aura instance. Create one at https://console.neo4j.io
    or use the credentials provided below.

Usage:
    export LYZR_API_KEY="your-api-key"
    python 03_knowledge_graph.py
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

# Neo4j Aura credentials
# These should be configured in Studio UI under Data Connectors (see setup steps below)
NEO4J_URI = "neo4j+s://0618163e.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "_xI3Dq5CkSoJb64JnzhSsRTUVEq_uMPCSmKPoo9iyf0"

# Placeholder document paths
PDF_PATH = "data/equipment_manual.pdf"
SAFETY_PDF_PATH = "data/safety_standards.pdf"
WEBSITE_URL = "https://example.com/industrial-docs"


# =============================================================================
# Studio UI Setup Instructions
# =============================================================================
#
# Before using the SDK to create a Neo4j knowledge base, you must configure
# the Neo4j data connector in the Lyzr Studio UI:
#
# 1. Navigate to https://studio.lyzr.ai
# 2. Go to Settings > Data Connectors (or Infrastructure > Data Connectors)
# 3. Click "Create New" or "Add Connector"
# 4. Select "Neo4j" as the connector type
# 5. Enter your Neo4j Aura credentials:
#    - URI:      neo4j+s://0618163e.databases.neo4j.io
#    - Username: neo4j
#    - Password: _xI3Dq5CkSoJb64JnzhSsRTUVEq_uMPCSmKPoo9iyf0
# 6. Test the connection and save
#
# Once the connector is saved, the SDK will automatically use these
# credentials when you create a knowledge base with vector_store="neo4j".
#


# =============================================================================
# 1. Create a Knowledge Graph Knowledge Base
# =============================================================================

def create_kg_knowledge_base(studio):
    """
    Create a knowledge base backed by Neo4j for knowledge graph storage.

    When vector_store="neo4j", Lyzr automatically:
    - Connects to the configured Neo4j instance
    - Extracts entities and relationships from documents using LLM
    - Stores them as nodes and edges in Neo4j
    - Creates vector indexes for hybrid search
    """
    kb = studio.create_knowledge_base(
        name="industrial_knowledge_graph",
        vector_store="neo4j",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Knowledge graph for industrial equipment, maintenance, and safety",
    )
    print(f"Knowledge Graph KB created: {kb.name}")
    print(f"  Vector store: neo4j")
    print(f"  Embedding model: text-embedding-3-large")
    return kb


# =============================================================================
# 2. Add Documents with Schema-Guided Extraction
# =============================================================================

def add_documents_with_schema(kb):
    """
    Add documents to the knowledge graph with a schema_prompt that guides
    the LLM entity/relationship extraction process.

    The schema_prompt tells the extraction LLM what types of entities and
    relationships to look for. This dramatically improves extraction quality
    for domain-specific documents.
    """

    # Define an industrial domain schema
    # This guides the LLM to extract specific entity types and relationship types
    industrial_schema = (
        "Entity types: Equipment, Component, Facility, Manufacturer, "
        "MaintenanceProcedure, SafetyStandard, Sensor, Material, "
        "FailureMode, Technician, SparePartNumber, OperatingParameter. "
        "Relationship types: HAS_COMPONENT, MANUFACTURED_BY, LOCATED_IN, "
        "REQUIRES_MAINTENANCE, FOLLOWS_STANDARD, MONITORS, MADE_OF, "
        "CAN_CAUSE_FAILURE, PERFORMED_BY, USES_PART, HAS_PARAMETER, "
        "DEPENDS_ON, REPLACES, TRIGGERS_ALERT."
    )

    # --- Add PDF with schema-guided extraction ---
    print("\n--- Adding equipment manual (with schema prompt) ---")
    try:
        result = kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
            extra_info=f'{{"schema_prompt": "{industrial_schema}"}}',
        )
        print(f"  Document added: {result}")
        print(f"  Schema-guided extraction will identify entities like:")
        print(f"    Equipment, Component, MaintenanceProcedure, FailureMode, etc.")
    except Exception as e:
        print(f"  Upload: {e}")

    # --- Add safety standards document ---
    print("\n--- Adding safety standards document ---")
    safety_schema = (
        "Entity types: SafetyStandard, Regulation, HazardType, "
        "ProtectiveEquipment, RiskLevel, ComplianceRequirement, "
        "InspectionSchedule, EmergencyProcedure. "
        "Relationship types: ADDRESSES_HAZARD, REQUIRES_EQUIPMENT, "
        "HAS_RISK_LEVEL, MANDATES_INSPECTION, INCLUDES_PROCEDURE, "
        "SUPERSEDES, REFERENCES."
    )
    try:
        result = kb.add_pdf(
            file_path=SAFETY_PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
            extra_info=f'{{"schema_prompt": "{safety_schema}"}}',
        )
        print(f"  Document added: {result}")
    except Exception as e:
        print(f"  Upload: {e}")


# =============================================================================
# 3. Add Documents without Schema (Open Extraction)
# =============================================================================

def add_documents_open_extraction(kb):
    """
    Without a schema_prompt, the LLM extracts entities and relationships
    freely based on the document content. This is useful when you don't
    know the domain structure upfront.
    """
    print("\n--- Adding document (open extraction, no schema) ---")
    try:
        result = kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="pymupdf",
        )
        print(f"  Document added with open extraction: {result}")
        print(f"  LLM will freely identify entity types and relationships.")
    except Exception as e:
        print(f"  Upload: {e}")


# =============================================================================
# 4. Add Website to Knowledge Graph
# =============================================================================

def add_website_to_kg(kb):
    """
    Crawl a website and add its content to the knowledge graph.
    Each page is processed for entity/relationship extraction.
    """
    print("\n--- Adding website to knowledge graph ---")
    try:
        result = kb.add_website(
            url=WEBSITE_URL,
            max_pages=20,
            max_depth=2,
            chunk_size=1024,
            chunk_overlap=128,
        )
        print(f"  Website added: {result}")
    except Exception as e:
        print(f"  Website addition: {e}")


# =============================================================================
# 5. Add Raw Text to Knowledge Graph
# =============================================================================

def add_text_to_kg(kb):
    """
    Add raw text directly to the knowledge graph.
    Useful for adding structured data, reports, or manual entries.
    """
    print("\n--- Adding raw text to knowledge graph ---")

    text_entries = [
        {
            "text": (
                "The Siemens SGT-800 gas turbine has a power output of 57 MW "
                "and thermal efficiency of 40%. It requires major overhaul every "
                "32,000 equivalent operating hours. Key components include the "
                "compressor section with 15 stages, combustion chamber with "
                "30 DLE burners, and a 4-stage turbine section."
            ),
            "source": "equipment_specs",
        },
        {
            "text": (
                "Maintenance procedure MP-2024-001: Hydraulic pump inspection. "
                "Step 1: Isolate the pump and depressurize the system. "
                "Step 2: Check oil level and quality using ISO 4406 cleanliness code. "
                "Step 3: Inspect seals for wear. Replace if Shore A hardness < 60. "
                "Step 4: Measure vibration at bearing points. Alert threshold: 4.5 mm/s RMS. "
                "Step 5: Record findings in CMMS and schedule follow-up if needed."
            ),
            "source": "maintenance_procedures",
        },
    ]

    for entry in text_entries:
        try:
            result = kb.add_text(
                text=entry["text"],
                source=entry["source"],
                chunk_size=512,
                chunk_overlap=64,
            )
            print(f"  Added text from '{entry['source']}': {result}")
        except Exception as e:
            print(f"  Text addition ({entry['source']}): {e}")


# =============================================================================
# 6. Query the Knowledge Graph
# =============================================================================

def query_knowledge_graph(kb):
    """
    Query the knowledge graph using natural language.
    The query engine translates questions into graph traversals combined
    with vector similarity search for hybrid retrieval.
    """
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH QUERIES")
    print("=" * 60)

    queries = [
        {
            "query": "What components does the SGT-800 gas turbine have?",
            "description": "Entity-relationship traversal",
        },
        {
            "query": "What maintenance procedures apply to hydraulic pumps?",
            "description": "Cross-entity relationship query",
        },
        {
            "query": "Which safety standards apply to high-pressure equipment?",
            "description": "Domain-specific entity lookup",
        },
        {
            "query": "What are the failure modes for compressor components?",
            "description": "Chain-of-relationship query",
        },
    ]

    for q in queries:
        print(f"\n--- {q['description']} ---")
        print(f"  Q: {q['query']}")
        try:
            results = kb.query(
                query=q["query"],
                top_k=5,
                retrieval_type="basic",
                score_threshold=0.3,
            )
            if results:
                for i, r in enumerate(results, 1):
                    print(f"  [{i}] Score: {r.score:.4f} | {r.text[:150]}...")
            else:
                print("  No results found.")
        except Exception as e:
            print(f"  Query error: {e}")


# =============================================================================
# 7. Agent-Powered Knowledge Graph Queries
# =============================================================================

def agent_with_kg(studio, kb):
    """
    Create an agent that uses the knowledge graph KB for graph-aware answers.
    The agent can traverse relationships to answer complex, multi-hop questions.
    """
    print("\n" + "=" * 60)
    print("  AGENT + KNOWLEDGE GRAPH")
    print("=" * 60)

    agent = studio.create_agent(
        name="KGExpert",
        provider="gpt-4o",
        role="Industrial Knowledge Graph Expert",
        goal=(
            "Answer questions using the knowledge graph to trace relationships "
            "between equipment, components, maintenance procedures, and safety standards"
        ),
        instructions=(
            "You have access to a knowledge graph containing industrial equipment data. "
            "When answering questions, trace relationships between entities to provide "
            "comprehensive answers. Reference specific entities and relationships. "
            "If a question requires multi-hop reasoning (e.g., 'what maintenance applies "
            "to components of turbine X'), walk through the relationship chain."
        ),
        temperature=0.2,
    )

    # Multi-hop query that benefits from graph traversal
    response = agent.run(
        message=(
            "For the SGT-800 gas turbine: list all components, their associated "
            "maintenance procedures, and any relevant safety standards."
        ),
        knowledge_bases=[kb.with_config(top_k=15, retrieval_type="basic")],
    )
    print(f"\n  Agent Response:\n  {response.response}")

    # Follow-up with failure analysis
    response_2 = agent.run(
        message=(
            "Based on the maintenance procedures, what are the most critical "
            "failure modes to watch for, and how can they be detected early?"
        ),
        session_id=response.session_id,
        knowledge_bases=[kb.with_config(top_k=15, retrieval_type="mmr")],
    )
    print(f"\n  Follow-up Response:\n  {response_2.response}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #3: Knowledge Graph with Neo4j")
    print("=" * 60)
    print()
    print("  NOTE: Ensure Neo4j credentials are configured in Studio UI")
    print("  (see setup instructions in this script).")
    print()

    studio = Studio(api_key=API_KEY)

    # Step 1: Create KG knowledge base
    kb = create_kg_knowledge_base(studio)

    # Step 2: Add documents with schema-guided extraction
    add_documents_with_schema(kb)

    # Step 3: Add documents with open extraction
    add_documents_open_extraction(kb)

    # Step 4: Add website content
    add_website_to_kg(kb)

    # Step 5: Add raw text
    add_text_to_kg(kb)

    # Step 6: Query the knowledge graph
    query_knowledge_graph(kb)

    # Step 7: Agent-powered KG queries
    agent_with_kg(studio, kb)

    print("\n" + "=" * 60)
    print("  Cookbook #3 complete!")
    print("=" * 60)
