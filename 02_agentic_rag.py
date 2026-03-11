"""
Lyzr ADK Cookbook #2: Agentic RAG
=================================

Demonstrates how to build AI agents that leverage knowledge bases for
context-aware responses, including:

- Agent creation with role, goal, and instructions
- Single-KB and multi-KB agent configurations
- Custom retrieval settings via kb.with_config()
- Session-based multi-turn conversations
- Streaming responses with knowledge base context

Prerequisites:
    pip install lyzr-adk

Usage:
    export LYZR_API_KEY="your-api-key"
    python 02_agentic_rag.py
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

PDF_PATH = "data/sample_document.pdf"
POLICY_PDF_PATH = "data/safety_policies.pdf"


# =============================================================================
# 1. Setup: Studio, Agent, and Knowledge Base
# =============================================================================

def setup():
    """Initialize Studio, create an agent and a knowledge base."""
    studio = Studio(api_key=API_KEY)

    # Create a knowledge base with industrial documents
    kb = studio.create_knowledge_base(
        name="industrial_ops_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Industrial operations documentation",
    )

    # Add a document
    try:
        kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
        )
        print(f"Document added to KB: {kb.name}")
    except Exception as e:
        print(f"Document upload: {e}")

    # Create an agent with a specific role
    agent = studio.create_agent(
        name="IndustrialAssistant",
        provider="gpt-4o",
        role="Industrial Operations Expert",
        goal="Help engineers with maintenance procedures, safety protocols, and operational best practices",
        instructions=(
            "You are an expert in industrial operations. Use the provided knowledge base "
            "to give accurate, specific answers. Always cite the source document when possible. "
            "If the knowledge base doesn't contain relevant information, say so clearly."
        ),
        temperature=0.3,
    )
    print(f"Agent created: {agent.name}")

    return studio, agent, kb


# =============================================================================
# 2. Basic Agent + KB Query
# =============================================================================

def basic_agent_query(agent, kb):
    """
    Run an agent query with knowledge base context.
    The agent automatically retrieves relevant chunks and uses them to answer.
    """
    print("\n" + "=" * 60)
    print("  BASIC AGENT + KB QUERY")
    print("=" * 60)

    response = agent.run(
        message="What are the recommended maintenance intervals for hydraulic systems?",
        knowledge_bases=[kb],
    )

    print(f"\n  Agent Response:\n  {response.response}")
    print(f"\n  Session ID: {response.session_id}")
    print(f"  Message ID: {response.message_id}")

    if response.tool_calls:
        print(f"  Tool Calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"    - {tc}")

    return response


# =============================================================================
# 3. Custom Retrieval Configuration
# =============================================================================

def custom_retrieval_query(agent, kb):
    """
    Use kb.with_config() to customize retrieval parameters at runtime
    without modifying the knowledge base itself.
    """
    print("\n" + "=" * 60)
    print("  CUSTOM RETRIEVAL CONFIGURATION")
    print("=" * 60)

    # --- MMR retrieval with high top_k for comprehensive answers ---
    print("\n--- MMR Retrieval (diverse, comprehensive) ---")
    response = agent.run(
        message="Summarize all safety procedures related to high-pressure equipment.",
        knowledge_bases=[
            kb.with_config(
                top_k=15,
                retrieval_type="mmr",
                score_threshold=0.5,
            )
        ],
    )
    print(f"\n  Response:\n  {response.response[:500]}...")

    # --- HyDE retrieval for abstract questions ---
    print("\n--- HyDE Retrieval (for complex questions) ---")
    response = agent.run(
        message="How can predictive analytics reduce unplanned downtime?",
        knowledge_bases=[
            kb.with_config(
                top_k=10,
                retrieval_type="hyde",
                score_threshold=0.4,
            )
        ],
    )
    print(f"\n  Response:\n  {response.response[:500]}...")

    # --- Time-aware retrieval for recent information ---
    print("\n--- Time-Aware Retrieval (recency-boosted) ---")
    response = agent.run(
        message="What are the latest regulatory changes for equipment safety?",
        knowledge_bases=[
            kb.with_config(
                top_k=10,
                retrieval_type="time_aware",
                score_threshold=0.3,
                time_decay_factor=0.5,
            )
        ],
    )
    print(f"\n  Response:\n  {response.response[:500]}...")


# =============================================================================
# 4. Multi-KB Agent
# =============================================================================

def multi_kb_agent(studio, agent):
    """
    Create multiple knowledge bases and give the agent access to all of them.
    The agent searches across all KBs to find the most relevant information.
    """
    print("\n" + "=" * 60)
    print("  MULTI-KB AGENT")
    print("=" * 60)

    # Create a second KB for safety policies
    safety_kb = studio.create_knowledge_base(
        name="safety_policies_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Safety policies and compliance documentation",
    )

    try:
        safety_kb.add_pdf(
            file_path=POLICY_PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
        )
        print(f"Safety KB created: {safety_kb.name}")
    except Exception as e:
        print(f"Safety KB upload: {e}")

    # Create the original operations KB
    ops_kb = studio.create_knowledge_base(
        name="operations_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Operations and maintenance documentation",
    )

    try:
        ops_kb.add_pdf(
            file_path=PDF_PATH,
            chunk_size=1024,
            chunk_overlap=128,
            data_parser="llmsherpa",
        )
        print(f"Operations KB created: {ops_kb.name}")
    except Exception as e:
        print(f"Operations KB upload: {e}")

    # Query agent with both KBs — agent searches across both
    response = agent.run(
        message=(
            "A hydraulic pump is showing abnormal vibration patterns. "
            "What maintenance steps should I take, and what safety protocols apply?"
        ),
        knowledge_bases=[
            ops_kb.with_config(top_k=10, retrieval_type="mmr"),
            safety_kb.with_config(top_k=5, retrieval_type="basic"),
        ],
    )

    print(f"\n  Multi-KB Response:\n  {response.response}")
    return response


# =============================================================================
# 5. Session-Based Conversation (Multi-Turn)
# =============================================================================

def session_conversation(agent, kb):
    """
    Use session_id for multi-turn conversations.
    The agent remembers previous messages in the same session.
    """
    print("\n" + "=" * 60)
    print("  SESSION-BASED CONVERSATION")
    print("=" * 60)

    # Turn 1: Initial question
    response_1 = agent.run(
        message="What are the key components of a hydraulic system?",
        knowledge_bases=[kb],
    )
    session_id = response_1.session_id
    print(f"\n  Turn 1 (session={session_id}):")
    print(f"  Q: What are the key components of a hydraulic system?")
    print(f"  A: {response_1.response[:300]}...")

    # Turn 2: Follow-up (agent remembers context from Turn 1)
    response_2 = agent.run(
        message="What are the common failure modes for those components?",
        session_id=session_id,  # Same session — maintains context
        knowledge_bases=[kb],
    )
    print(f"\n  Turn 2 (same session):")
    print(f"  Q: What are the common failure modes for those components?")
    print(f"  A: {response_2.response[:300]}...")

    # Turn 3: Another follow-up
    response_3 = agent.run(
        message="How can I detect these failures early using sensor data?",
        session_id=session_id,
        knowledge_bases=[kb],
    )
    print(f"\n  Turn 3 (same session):")
    print(f"  Q: How can I detect these failures early using sensor data?")
    print(f"  A: {response_3.response[:300]}...")

    return session_id


# =============================================================================
# 6. Streaming Response with KB
# =============================================================================

def streaming_with_kb(agent, kb):
    """
    Stream agent responses in real-time while using knowledge base context.
    Useful for long answers or chat UIs that show progressive output.
    """
    print("\n" + "=" * 60)
    print("  STREAMING RESPONSE WITH KB")
    print("=" * 60)

    print("\n  Streaming response:\n  ", end="")

    response = agent.run(
        message="Explain the complete process for commissioning a new industrial pump.",
        knowledge_bases=[kb.with_config(top_k=10, retrieval_type="mmr")],
        stream=True,
    )

    # When stream=True, iterate over the response chunks
    if hasattr(response, "__iter__"):
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
    else:
        # Fallback if streaming returns a complete response
        print(response.response)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #2: Agentic RAG")
    print("=" * 60)

    # Setup
    studio, agent, kb = setup()

    # Demo 1: Basic agent + KB query
    basic_agent_query(agent, kb)

    # Demo 2: Custom retrieval settings
    custom_retrieval_query(agent, kb)

    # Demo 3: Multi-KB agent
    multi_kb_agent(studio, agent)

    # Demo 4: Session-based conversation
    session_conversation(agent, kb)

    # Demo 5: Streaming with KB
    streaming_with_kb(agent, kb)

    print("\n" + "=" * 60)
    print("  Cookbook #2 complete!")
    print("=" * 60)
