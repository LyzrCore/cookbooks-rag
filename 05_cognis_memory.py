"""
Lyzr ADK Cookbook #5: Cognis Memory System
==========================================

Demonstrates Lyzr's Cognis memory system for persistent, semantic
conversation memory across sessions, including:

- Adding conversation memories (multi-turn)
- Semantic search across memories
- Cross-session memory retrieval
- Memory scoping (owner_id, agent_id, session_id)
- Memory management (get, update, delete)
- Async usage patterns
- Integration with agents

Prerequisites:
    pip install lyzr-adk

Usage:
    export LYZR_API_KEY="your-api-key"
    python 05_cognis_memory.py
"""

import asyncio
import os
import sys

from lyzr import Cognis, CognisMessage, Studio


# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.environ.get("LYZR_API_KEY")
if not API_KEY:
    print("Error: Set the LYZR_API_KEY environment variable.")
    sys.exit(1)


# =============================================================================
# 1. Initialize Cognis
# =============================================================================

def init_cognis():
    """
    Initialize the Cognis memory client.

    Cognis provides semantic memory storage that:
    - Extracts key facts from conversations
    - Stores them as searchable memory units
    - Supports scoping by owner, agent, and session
    - Enables cross-session memory retrieval
    """
    cog = Cognis(api_key=API_KEY)
    print("Cognis initialized.")
    return cog


# =============================================================================
# 2. Add Conversation Memories
# =============================================================================

def add_memories(cog):
    """
    Add multi-turn conversation memories.
    Cognis extracts key facts from the conversation and stores them
    as individual memory units.
    """
    print("\n" + "=" * 60)
    print("  ADDING CONVERSATION MEMORIES")
    print("=" * 60)

    # --- Session 1: Equipment troubleshooting ---
    print("\n--- Session 1: Equipment troubleshooting ---")
    try:
        result = cog.add(
            messages=[
                {"role": "user", "content": "The hydraulic pump on Line 3 is showing abnormal vibration. Readings are at 6.2 mm/s RMS."},
                {"role": "assistant", "content": "That's above the 4.5 mm/s alert threshold. This could indicate bearing wear, misalignment, or cavitation. I recommend an immediate inspection."},
                {"role": "user", "content": "We inspected it and found the coupling alignment was off by 0.15mm. We realigned it and vibration dropped to 2.1 mm/s."},
                {"role": "assistant", "content": "Good catch. The alignment tolerance for this pump model is 0.05mm, so 0.15mm was significantly out of spec. I'd recommend checking the alignment monthly for the next quarter to ensure it holds."},
            ],
            owner_id="engineer_001",
            agent_id="maintenance_bot",
            session_id="session_troubleshoot_001",
        )
        print(f"  Memories added: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Session 2: Safety review ---
    print("\n--- Session 2: Safety review ---")
    try:
        result = cog.add(
            messages=[
                {"role": "user", "content": "We need to update our lockout/tagout procedures for the new press line."},
                {"role": "assistant", "content": "For the new press line, you'll need LOTO procedures that cover the hydraulic, pneumatic, and electrical energy sources. OSHA 29 CFR 1910.147 requires documented procedures for each machine."},
                {"role": "user", "content": "We also need to train the night shift crew. There are 12 operators."},
                {"role": "assistant", "content": "I'd suggest a hands-on training session with certification. Each operator should demonstrate proper lockout/tagout on the actual equipment before being authorized to work independently."},
            ],
            owner_id="engineer_001",
            agent_id="safety_bot",
            session_id="session_safety_001",
        )
        print(f"  Memories added: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    # --- Session 3: Different user (cross-owner demo) ---
    print("\n--- Session 3: Different user ---")
    try:
        result = cog.add(
            messages=[
                {"role": "user", "content": "What's the recommended oil change interval for the Caterpillar 320 excavators?"},
                {"role": "assistant", "content": "For the Cat 320, engine oil should be changed every 500 hours with Cat DEO-ULS 15W-40. Hydraulic oil should be checked at 500 hours and changed at 2000 hours using Cat HYDO Advanced 10W."},
            ],
            owner_id="engineer_002",
            agent_id="maintenance_bot",
            session_id="session_maint_002",
        )
        print(f"  Memories added: {result}")
    except Exception as e:
        print(f"  Error: {e}")


# =============================================================================
# 3. Search Memories
# =============================================================================

def search_memories(cog):
    """
    Search memories using natural language queries.
    Cognis uses semantic similarity to find relevant memories.
    """
    print("\n" + "=" * 60)
    print("  SEARCHING MEMORIES")
    print("=" * 60)

    # --- Search within a specific owner ---
    print("\n--- Search: 'vibration issues' (owner: engineer_001) ---")
    try:
        results = cog.search(
            query="vibration problems on equipment",
            owner_id="engineer_001",
            limit=5,
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r.score:.4f}")
                print(f"      Content: {r.content[:200]}...")
                print(f"      Owner: {r.owner_id} | Created: {r.created_at}")
        else:
            print("  No matching memories found.")
    except Exception as e:
        print(f"  Search error: {e}")

    # --- Search for safety-related memories ---
    print("\n--- Search: 'lockout tagout procedures' (owner: engineer_001) ---")
    try:
        results = cog.search(
            query="lockout tagout training requirements",
            owner_id="engineer_001",
            limit=5,
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r.score:.4f}")
                print(f"      Content: {r.content[:200]}...")
        else:
            print("  No matching memories found.")
    except Exception as e:
        print(f"  Search error: {e}")

    # --- Search across a different owner ---
    print("\n--- Search: 'oil change' (owner: engineer_002) ---")
    try:
        results = cog.search(
            query="oil change intervals for excavators",
            owner_id="engineer_002",
            limit=5,
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r.score:.4f}")
                print(f"      Content: {r.content[:200]}...")
        else:
            print("  No matching memories found.")
    except Exception as e:
        print(f"  Search error: {e}")


# =============================================================================
# 4. Cross-Session Memory Retrieval
# =============================================================================

def cross_session_search(cog):
    """
    Search across all sessions for a given owner.
    This enables an agent to recall information from any past conversation.
    """
    print("\n" + "=" * 60)
    print("  CROSS-SESSION MEMORY RETRIEVAL")
    print("=" * 60)

    print("\n--- Cross-session search for engineer_001 ---")
    print("  (Searching across troubleshooting + safety sessions)\n")

    try:
        results = cog.search(
            query="What equipment issues and procedures were discussed?",
            owner_id="engineer_001",
            limit=10,
            cross_session=True,  # Search across all sessions
        )
        if results:
            for i, r in enumerate(results, 1):
                print(f"  [{i}] Score: {r.score:.4f}")
                print(f"      Content: {r.content[:200]}...")
                if hasattr(r, "metadata") and r.metadata:
                    print(f"      Metadata: {r.metadata}")
        else:
            print("  No cross-session memories found.")
    except Exception as e:
        print(f"  Cross-session search error: {e}")


# =============================================================================
# 5. Memory Scoping
# =============================================================================

def memory_scoping(cog):
    """
    Demonstrate memory scoping with owner_id, agent_id, and session_id.

    Scoping hierarchy:
    - owner_id:   User or entity that owns the memory (required)
    - agent_id:   The agent that participated in the conversation
    - session_id: The specific session/conversation
    """
    print("\n" + "=" * 60)
    print("  MEMORY SCOPING")
    print("=" * 60)

    # Scope by agent_id — find only memories from the maintenance bot
    print("\n--- Scope: agent_id='maintenance_bot' ---")
    try:
        results = cog.search(
            query="equipment maintenance",
            owner_id="engineer_001",
            agent_id="maintenance_bot",
            limit=5,
        )
        print(f"  Maintenance bot memories: {len(results) if results else 0}")
        if results:
            for r in results:
                print(f"    - {r.content[:150]}...")
    except Exception as e:
        print(f"  Error: {e}")

    # Scope by agent_id — find only memories from the safety bot
    print("\n--- Scope: agent_id='safety_bot' ---")
    try:
        results = cog.search(
            query="safety procedures",
            owner_id="engineer_001",
            agent_id="safety_bot",
            limit=5,
        )
        print(f"  Safety bot memories: {len(results) if results else 0}")
        if results:
            for r in results:
                print(f"    - {r.content[:150]}...")
    except Exception as e:
        print(f"  Error: {e}")


# =============================================================================
# 6. Memory Management (Get, Update, Delete)
# =============================================================================

def memory_management(cog):
    """
    Manage individual memories: retrieve, update, and delete.
    """
    print("\n" + "=" * 60)
    print("  MEMORY MANAGEMENT")
    print("=" * 60)

    # First, search to find a memory ID to work with
    try:
        results = cog.search(
            query="hydraulic pump vibration",
            owner_id="engineer_001",
            limit=1,
        )
        if not results:
            print("  No memories found to manage.")
            return

        memory_id = results[0].id
        print(f"\n  Found memory ID: {memory_id}")

        # --- Get a specific memory by ID ---
        print("\n--- Get memory by ID ---")
        memory = cog.get(memory_id=memory_id)
        print(f"  Content: {memory.content[:200]}...")
        print(f"  Owner: {memory.owner_id}")
        print(f"  Created: {memory.created_at}")

        # --- Update a memory ---
        print("\n--- Update memory ---")
        updated = cog.update(
            memory_id=memory_id,
            content=(
                "Hydraulic pump on Line 3 had vibration at 6.2 mm/s RMS "
                "(threshold: 4.5 mm/s). Root cause: coupling misalignment "
                "at 0.15mm (tolerance: 0.05mm). Fixed by realignment. "
                "Post-fix reading: 2.1 mm/s. Monthly alignment checks recommended."
            ),
        )
        print(f"  Updated memory: {updated}")

        # --- Delete a memory ---
        print("\n--- Delete memory ---")
        # Uncomment the line below to actually delete:
        # deleted = cog.delete(memory_id=memory_id)
        # print(f"  Deleted: {deleted}")
        print("  (Skipped — uncomment to actually delete)")

    except Exception as e:
        print(f"  Memory management error: {e}")


# =============================================================================
# 7. Async Usage Pattern
# =============================================================================

async def async_memory_operations():
    """
    Demonstrate async usage of Cognis for high-throughput applications.
    """
    print("\n" + "=" * 60)
    print("  ASYNC MEMORY OPERATIONS")
    print("=" * 60)

    cog = Cognis(api_key=API_KEY)

    # Add multiple conversations concurrently
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Boiler B-12 pressure dropped below 150 PSI."},
                {"role": "assistant", "content": "Check the pressure relief valve and feedwater pump. Normal operating pressure should be 180-200 PSI."},
            ],
            "owner_id": "engineer_003",
            "session_id": "async_session_1",
        },
        {
            "messages": [
                {"role": "user", "content": "Conveyor belt on Line 7 keeps slipping."},
                {"role": "assistant", "content": "Check belt tension and alignment. Tension should be 2-3% elongation. Also inspect the drive pulley for wear."},
            ],
            "owner_id": "engineer_003",
            "session_id": "async_session_2",
        },
    ]

    # Add memories sequentially (async pattern shown for structure)
    for conv in conversations:
        try:
            result = cog.add(
                messages=conv["messages"],
                owner_id=conv["owner_id"],
                session_id=conv["session_id"],
            )
            print(f"  Added memory for session {conv['session_id']}: {result}")
        except Exception as e:
            print(f"  Error for session {conv['session_id']}: {e}")

    # Search across all async sessions
    try:
        results = cog.search(
            query="equipment pressure and tension issues",
            owner_id="engineer_003",
            limit=5,
            cross_session=True,
        )
        print(f"\n  Cross-session search results: {len(results) if results else 0}")
        if results:
            for r in results:
                print(f"    - {r.content[:150]}...")
    except Exception as e:
        print(f"  Search error: {e}")


# =============================================================================
# 8. Agent + Cognis Integration
# =============================================================================

def agent_with_memory():
    """
    Show how Cognis memory can enhance agent conversations by providing
    context from past interactions.
    """
    print("\n" + "=" * 60)
    print("  AGENT + COGNIS INTEGRATION")
    print("=" * 60)

    studio = Studio(api_key=API_KEY)
    cog = Cognis(api_key=API_KEY)

    # Create an agent
    agent = studio.create_agent(
        name="MemoryAgent",
        provider="gpt-4o",
        role="Industrial Assistant with Memory",
        goal="Help engineers while remembering past conversations",
        instructions=(
            "You have access to conversation history from previous sessions. "
            "Use this context to provide personalized, continuity-aware responses. "
            "Reference past issues and solutions when relevant."
        ),
        temperature=0.3,
    )

    owner_id = "engineer_001"

    # Retrieve relevant memories before running the agent
    print("\n--- Retrieving past memories for context ---")
    try:
        memories = cog.search(
            query="equipment issues and maintenance procedures",
            owner_id=owner_id,
            limit=5,
            cross_session=True,
        )

        # Build context from memories
        memory_context = ""
        if memories:
            memory_context = "Previous conversation context:\n"
            for m in memories:
                memory_context += f"- {m.content}\n"
            print(f"  Found {len(memories)} relevant memories.")
        else:
            print("  No past memories found.")

        # Run agent with memory context injected into the message
        response = agent.run(
            message=(
                f"{memory_context}\n\n"
                "Current question: The hydraulic pump on Line 3 is vibrating again. "
                "What should we check first based on our previous experience?"
            ),
        )
        print(f"\n  Agent Response (with memory context):")
        print(f"  {response.response}")

        # Save this new conversation to Cognis
        cog.add(
            messages=[
                {"role": "user", "content": "The hydraulic pump on Line 3 is vibrating again. What should we check first?"},
                {"role": "assistant", "content": response.response},
            ],
            owner_id=owner_id,
            agent_id="memory_agent",
            session_id="session_followup_001",
        )
        print("\n  New conversation saved to Cognis memory.")

    except Exception as e:
        print(f"  Error: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Lyzr ADK Cookbook #5: Cognis Memory System")
    print("=" * 60)

    cog = init_cognis()

    # Demo 1: Add conversation memories
    add_memories(cog)

    # Demo 2: Search memories
    search_memories(cog)

    # Demo 3: Cross-session retrieval
    cross_session_search(cog)

    # Demo 4: Memory scoping
    memory_scoping(cog)

    # Demo 5: Memory management
    memory_management(cog)

    # Demo 6: Async usage
    asyncio.run(async_memory_operations())

    # Demo 7: Agent + Cognis integration
    agent_with_memory()

    print("\n" + "=" * 60)
    print("  Cookbook #5 complete!")
    print("=" * 60)
