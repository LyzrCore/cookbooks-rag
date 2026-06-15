"""
Fallback: build the multi-KB fixture WITHOUT calling get_knowledge_base
(which was hanging on dev). The agentic/oneshot feature-config dicts are
constructed directly from KB_SPECS + the known kb_ids — they only need
rag_id, name, description, and retrieval params.

Includes the SDK monkeypatch (inject api_key the client holds) so
create_agent's post-create deserialization doesn't fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from lyzr.agents import AgentModule
_orig_make = AgentModule._make_smart_agent
def _patched_make(self, agent_data, response_model=None):
    if isinstance(agent_data, dict) and "api_key" not in agent_data:
        agent_data["api_key"] = getattr(self._http, "api_key", None) or os.environ.get("LYZR_API_KEY")
    return _orig_make(self, agent_data, response_model=response_model)
AgentModule._make_smart_agent = _patched_make

from lyzr import Studio
from setup_dev_multi_kb import (
    KB_SPECS, AGENT_ROLE, AGENT_GOAL, AGENT_INSTRUCTIONS, AGENT_PROVIDER, AGENT_TEMPERATURE,
)
from finalize_multi_kb import KB_IDS


def agentic_cfg(spec, kb_id):
    return {
        "rag_id": kb_id, "name": spec.name,
        "description": spec.description,
        "top_k": 10, "retrieval_type": "basic",
        "score_threshold": 0.0, "time_decay_factor": 0.4,
    }


def oneshot_cfg(spec, kb_id):
    return {
        "rag_id": kb_id, "name": spec.name,
        "description": spec.description,
        "top_k": 20, "retrieval_type": "basic", "score_threshold": 0.0,
        "filter_fields": spec.filter_fields, "examples": spec.examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="dev")
    ap.add_argument("--out", default="/tmp/dev_multi_kb_fixture.json")
    args = ap.parse_args()
    if not os.environ.get("LYZR_API_KEY"):
        print("Error: LYZR_API_KEY required.", file=sys.stderr)
        return 2

    studio = Studio(api_key=os.environ["LYZR_API_KEY"], env=args.env)
    specs = {s.idx: s for s in KB_SPECS}
    agentic_cfgs = [agentic_cfg(specs[i], KB_IDS[i]) for i in range(1, 12)]
    oneshot_cfgs = [oneshot_cfg(specs[i], KB_IDS[i]) for i in range(1, 12)]

    common = dict(provider=AGENT_PROVIDER, role=AGENT_ROLE, goal=AGENT_GOAL,
                  instructions=AGENT_INSTRUCTIONS, temperature=AGENT_TEMPERATURE)
    ts = int(time.time())

    print(f"[direct] Creating agentic_rag agent ({len(agentic_cfgs)} KBs)...", flush=True)
    a = studio.create_agent(
        name=f"multikb_agentic_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE", "config": {"lyzr_rag": {}, "agentic_rag": agentic_cfgs}, "priority": 0}],
        **common)
    print(f"[direct]   agentic_agent_id={a.id}", flush=True)

    print(f"[direct] Creating oneshot_rag agent ({len(oneshot_cfgs)} KBs)...", flush=True)
    b = studio.create_agent(
        name=f"multikb_oneshot_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE", "config": {"lyzr_rag": {}, "oneshot_rag": oneshot_cfgs,
                   "planner_model": "gpt-4o-mini", "merge_top_k": 6}, "priority": 0}],
        **common)
    print(f"[direct]   oneshot_agent_id={b.id}", flush=True)

    fixture = {
        "env": args.env, "ts": ts,
        "agentic_agent_id": a.id, "oneshot_agent_id": b.id,
        "kbs": [{"idx": i, "name": specs[i].name, "kb_id": KB_IDS[i],
                 "description": specs[i].description, "examples": specs[i].examples,
                 "filter_fields": specs[i].filter_fields} for i in range(1, 12)],
    }
    json.dump(fixture, open(args.out, "w"), indent=2, default=str)
    print(f"\n[direct] Fixture -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
