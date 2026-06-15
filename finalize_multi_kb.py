"""
Finalize the multi-KB fixture after recovery:
  - Monkeypatch the SDK Agent deserialization bug (GET response omits api_key,
    which the pydantic Agent model requires). We inject the api_key the HTTP
    client already holds.
  - Serially retry the files that hit client-side ReadError/ReadTimeout during
    the parallel recovery (KB9 internship, KB10 commercial wire).
  - Create both agents over all 11 KB ids, write the fixture.

All 11 kb_ids are known from the recovery log.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# ---- SDK monkeypatch: inject api_key the client holds into agent deserialization ----
from lyzr.agents import AgentModule
_orig_make = AgentModule._make_smart_agent
def _patched_make(self, agent_data, response_model=None):
    if isinstance(agent_data, dict) and "api_key" not in agent_data:
        agent_data["api_key"] = getattr(self._http, "api_key", None) or os.environ.get("LYZR_API_KEY")
    return _orig_make(self, agent_data, response_model=response_model)
AgentModule._make_smart_agent = _patched_make
# -------------------------------------------------------------------------------------

from lyzr import Studio
from setup_dev_multi_kb import (
    KB_SPECS, ingest_kb,
    AGENT_ROLE, AGENT_GOAL, AGENT_INSTRUCTIONS, AGENT_PROVIDER, AGENT_TEMPERATURE,
)

# All 11 kb_ids (9 from the first run + 10/11 created in recovery).
KB_IDS = {
    1: "6a2ddc23955188b6c9726407",
    2: "6a2ddc76955188b6c972640f",
    3: "6a2ddcaa955188b6c9726413",
    4: "6a2ddd38955188b6c9726425",
    5: "6a2ddda5955188b6c972642d",
    6: "6a2de03c955188b6c972642f",
    7: "6a2de14f955188b6c9726433",
    8: "6a2de4c4955188b6c972643b",
    9: "6a2de582955188b6c972643f",
    10: "6a2e3989955188b6c9726455",
    11: "6a2e4c0f955188b6c972646d",
}

# Files that hit client-side ReadError/ReadTimeout during the parallel recovery.
# Retried serially here. (parser per doc type: KB9 born-digital -> standard, KB10 scans -> advanced)
RETRY = {
    9: ("standard", [
        "Demo_Internship_Report_2.pdf",
        "Feedback from industry expert.pdf",
        "3180701_Internship_Report_Guidelines.pdf",
        "Part_4_All_chapters.pdf",
        "Part_2_Certificate_and_Completion_Certificate.pdf",
        "STUDENT’S DAILY DIARY LOG.pdf",
        "Demo_Internship_Report_1.pdf",
        "Weekly log.pdf",
    ]),
    10: ("advanced", [
        "Commercial Sample 2 BravoEnterprisesInc01122026pdf.pdf",
        "Commercial Sample 1 Wire-Checklist.pdf",
        "Commercial Sample 3 Wire Request.pdf",
    ]),
}


def spec_by_idx(idx):
    return next(s for s in KB_SPECS if s.idx == idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["dev", "prod", "local"], default="dev")
    ap.add_argument("--out", default="/tmp/dev_multi_kb_fixture.json")
    ap.add_argument("--skip-retry", action="store_true", help="Skip serial re-ingest of dropped files.")
    args = ap.parse_args()

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY required.", file=sys.stderr)
        return 2

    studio = Studio(api_key=api_key, env=args.env)
    base_url = studio._http.base_url  # type: ignore[attr-defined]
    print(f"[finalize] env={args.env} base_url={base_url}")

    retry_summary = {}
    if not args.skip_retry:
        for idx, (parser, basenames) in RETRY.items():
            s = spec_by_idx(idx)
            want = set(basenames)
            files = [f for f in s.files if os.path.basename(f) in want and os.path.exists(f)]
            kb_id = KB_IDS[idx]
            print(f"\n[finalize] KB{idx} {s.name}: serial retry {len(files)} file(s) via {parser}", flush=True)
            # parallelism=1 to avoid the gateway connection drops we hit at 4
            results = asyncio.run(ingest_kb(base_url, api_key, kb_id, files, parser, 1))
            ok = [r for r in results if r.error is None]
            retry_summary[idx] = {"retry_ok": len(ok), "retry_total": len(results)}

    print("\n[finalize] Building feature configs over all 11 KBs...", flush=True)
    agentic_cfgs, oneshot_cfgs, kb_records = [], [], []
    for idx in range(1, 12):
        s = spec_by_idx(idx)
        kb = studio.get_knowledge_base(KB_IDS[idx])
        agentic_cfgs.append(kb.to_agentic_config(top_k=10, retrieval_type="basic"))
        oneshot_cfgs.append(kb.to_oneshot_config(
            top_k=20, retrieval_type="basic",
            filter_fields=s.filter_fields, examples=s.examples,
        ))
        kb_records.append({
            "idx": idx, "name": s.name, "kb_id": KB_IDS[idx],
            "description": s.description, "examples": s.examples,
            "filter_fields": s.filter_fields,
        })

    common = dict(provider=AGENT_PROVIDER, role=AGENT_ROLE, goal=AGENT_GOAL,
                  instructions=AGENT_INSTRUCTIONS, temperature=AGENT_TEMPERATURE)
    ts = int(time.time())

    print(f"[finalize] Creating agentic_rag agent ({len(agentic_cfgs)} KBs)...", flush=True)
    agent_a = studio.create_agent(
        name=f"multikb_agentic_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "agentic_rag": agentic_cfgs}, "priority": 0}],
        **common,
    )
    print(f"[finalize]   agentic_agent_id={agent_a.id}", flush=True)

    print(f"[finalize] Creating oneshot_rag agent ({len(oneshot_cfgs)} KBs)...", flush=True)
    agent_b = studio.create_agent(
        name=f"multikb_oneshot_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "oneshot_rag": oneshot_cfgs,
                              "planner_model": "gpt-4o-mini", "merge_top_k": 6}, "priority": 0}],
        **common,
    )
    print(f"[finalize]   oneshot_agent_id={agent_b.id}", flush=True)

    fixture = {
        "env": args.env, "ts": ts,
        "agentic_agent_id": agent_a.id, "oneshot_agent_id": agent_b.id,
        "retry_summary": retry_summary,
        "kbs": kb_records,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, default=str)
    print(f"\n[finalize] Fixture -> {args.out}")
    print(f"[finalize] retry_summary={retry_summary}")
    print(f"[finalize] Next: python benchmark_multi_kb.py --fixture {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
