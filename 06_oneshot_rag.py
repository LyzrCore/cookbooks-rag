"""
Lyzr ADK Cookbook #6: One-Shot Planner+Synthesizer RAG
======================================================

Replacement for the agentic (ReAct-loop) RAG in cookbook #2. The agent makes
exactly two LLM calls per turn:

    1. Planner (fast, small model, structured JSON output)
       — decides which KBs to query, generates per-KB sub-queries and
         metadata filters, or flips `needs_retrieval=false` for chitchat.
    2. Parallel retrieval via asyncio.gather (no tool loop, no waiting).
    3. Synthesizer (gpt-4o, streamed) — answers from chunks with citations.

Compare wall-clock and `response.tool_calls` against `02_agentic_rag.py`:
oneshot has no `tool_calls` and should land in single-digit seconds.

Prerequisites:
    pip install lyzr-adk

Usage:
    export LYZR_API_KEY="your-api-key"
    python 06_oneshot_rag.py
"""

import os
import sys
import time

from lyzr import Studio


API_KEY = os.environ.get("LYZR_API_KEY")
if not API_KEY:
    print("Error: Set the LYZR_API_KEY environment variable.")
    sys.exit(1)

PDF_PATH = "data/sample_document.pdf"
POLICY_PDF_PATH = "data/safety_policies.pdf"


# =============================================================================
# Setup
# =============================================================================


def setup():
    """Create a Studio, one or two KBs, and an agent. Identical to #2 so the
    comparison is apples-to-apples — only the `oneshot=True` flag differs at
    call time.
    """
    studio = Studio(api_key=API_KEY)

    kb = studio.create_knowledge_base(
        name="industrial_ops_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Industrial operations and maintenance documentation.",
    )

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

    agent = studio.create_agent(
        name="IndustrialAssistant",
        provider="gpt-4o",
        role="Industrial Operations Expert",
        goal="Help engineers with maintenance procedures, safety protocols, and operational best practices",
        instructions=(
            "Answer concisely. When the knowledge base is queried, cite each "
            "claim using the [doc_id:chunk_id] markers in the retrieved chunks."
        ),
        temperature=0.3,
    )
    print(f"Agent created: {agent.name}")
    return studio, agent, kb


# =============================================================================
# 1. Basic one-shot query
# =============================================================================


def basic_oneshot_query(agent, kb):
    print("\n" + "=" * 60)
    print("  ONE-SHOT BASIC QUERY")
    print("=" * 60)

    t0 = time.perf_counter()
    response = agent.run(
        message="What are the recommended maintenance intervals for hydraulic systems?",
        knowledge_bases=[kb],
        oneshot=True,
    )
    dt = time.perf_counter() - t0

    print(f"\n  Total wall-clock: {dt:.2f}s")
    print(f"  tool_calls (should be empty): {response.tool_calls!r}")
    if getattr(response, "plan", None):
        print(f"  Planner: needs_retrieval={response.plan.get('needs_retrieval')}")
        for sq in response.plan.get("sub_queries", []):
            print(f"    - kb={sq['kb_id']!r}  q={sq['query']!r}")
    print(f"\n  Answer:\n  {response.response}")


# =============================================================================
# 2. Per-call planner hints (filter_fields, examples)
# =============================================================================


def planner_hints(agent, kb):
    print("\n" + "=" * 60)
    print("  ONE-SHOT WITH PLANNER HINTS (filter_fields + examples)")
    print("=" * 60)

    # `with_config` collects hints that go straight to the planner's
    # KB catalog so it knows which filter keys are legal and what kinds of
    # queries this KB answers.
    configured = kb.with_config(
        top_k=20,
        retrieval_type="basic",
        filter_fields=["source", "doc_type"],
        examples=[
            "preventive maintenance for hydraulic pumps",
            "torque specification for valve V-12",
        ],
    )

    t0 = time.perf_counter()
    response = agent.run(
        message=(
            "Compare the maintenance and safety procedures for hydraulic vs "
            "pneumatic systems."
        ),
        knowledge_bases=[configured],
        oneshot=True,
        merge_top_k=8,            # 4-8 is the documented sweet spot.
        planner_model="gpt-4o-mini",
    )
    dt = time.perf_counter() - t0

    print(f"\n  Total wall-clock: {dt:.2f}s")
    if getattr(response, "plan", None):
        print(f"  Planner sub-queries:")
        for sq in response.plan.get("sub_queries", []):
            print(f"    - kb={sq['kb_id']!r}  q={sq['query']!r}")
    print(f"\n  Answer:\n  {response.response[:500]}...")


# =============================================================================
# 3. Multi-KB one-shot
# =============================================================================


def multi_kb_oneshot(studio, agent, ops_kb):
    print("\n" + "=" * 60)
    print("  MULTI-KB ONE-SHOT")
    print("=" * 60)

    safety_kb = studio.create_knowledge_base(
        name="safety_policies_kb",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description="Safety policies and compliance documentation.",
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

    t0 = time.perf_counter()
    response = agent.run(
        message=(
            "A hydraulic pump is showing abnormal vibration patterns. What "
            "maintenance steps should I take, and what safety protocols apply?"
        ),
        knowledge_bases=[
            ops_kb.with_config(
                examples=["hydraulic pump vibration diagnosis", "bearing wear procedure"],
            ),
            safety_kb.with_config(
                examples=["high-pressure equipment safety", "lockout-tagout"],
                filter_fields=["doc_type", "policy_year"],
            ),
        ],
        oneshot=True,
    )
    dt = time.perf_counter() - t0

    print(f"\n  Total wall-clock: {dt:.2f}s")
    if getattr(response, "plan", None):
        print(f"  Planner routed to {len(response.plan.get('sub_queries', []))} sub-queries:")
        for sq in response.plan.get("sub_queries", []):
            print(f"    - kb={sq['kb_id']!r}  q={sq['query']!r}")
    print(f"\n  Answer:\n  {response.response[:600]}...")


# =============================================================================
# 4. Chitchat short-circuit (no retrieval needed)
# =============================================================================


def chitchat_short_circuit(agent, kb):
    print("\n" + "=" * 60)
    print("  CHITCHAT SHORT-CIRCUIT (planner should set needs_retrieval=false)")
    print("=" * 60)

    t0 = time.perf_counter()
    response = agent.run(
        message="Hi, what can you help me with?",
        knowledge_bases=[kb],
        oneshot=True,
    )
    dt = time.perf_counter() - t0

    print(f"\n  Total wall-clock: {dt:.2f}s")
    if getattr(response, "plan", None):
        print(
            f"  Planner: needs_retrieval={response.plan.get('needs_retrieval')} "
            f"(should be False for greetings)"
        )
    print(f"\n  Answer:\n  {response.response[:300]}")


# =============================================================================
# 5. Streaming one-shot
# =============================================================================


def streaming_oneshot(agent, kb):
    print("\n" + "=" * 60)
    print("  STREAMING ONE-SHOT")
    print("=" * 60)

    print("\n  Streaming: ", end="", flush=True)
    t0 = time.perf_counter()
    first_token_t = None
    for chunk in agent.run(
        message="Explain the complete process for commissioning a new industrial pump.",
        knowledge_bases=[kb],
        oneshot=True,
        stream=True,
    ):
        if first_token_t is None and chunk.content:
            first_token_t = time.perf_counter()
        print(chunk.content, end="", flush=True)
    total_t = time.perf_counter() - t0
    ttft = (first_token_t - t0) if first_token_t else total_t
    print()
    print(f"\n  TTFT: {ttft:.2f}s   Total: {total_t:.2f}s")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #6: One-Shot Planner+Synthesizer RAG")
    print("=" * 60)

    studio, agent, kb = setup()

    basic_oneshot_query(agent, kb)
    planner_hints(agent, kb)
    multi_kb_oneshot(studio, agent, kb)
    chitchat_short_circuit(agent, kb)
    streaming_oneshot(agent, kb)

    print("\n" + "=" * 60)
    print("  Cookbook #6 complete!")
    print("=" * 60)
