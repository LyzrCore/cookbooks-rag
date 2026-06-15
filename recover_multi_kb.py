"""
Surgical recovery for the multi-KB fixture after the setup crash.

State after the crashed paddleocr run:
  KB1-4, 6, 8  : fully ingested (paddleocr) — REUSE as-is.
  KB5          : 0/3 (3 giant ML books hard-failed paddleocr) — re-ingest via LiteParse.
  KB7          : 3/4 ("Python for Gemini and Bard" failed) — re-ingest that 1 via LiteParse.
  KB9          : created but empty (crash) — ingest all 10 via LiteParse.
  KB10, KB11   : never created — create + ingest via PaddleOCR (genuine scans).

Then create both agents wired to all 11 kb_ids and write the fixture.

Born-digital papers/books/reports -> LiteParse (standard); scanned wire docs -> PaddleOCR (advanced).

Usage:
    export LYZR_API_KEY="dev-key"
    python recover_multi_kb.py --out /tmp/dev_multi_kb_fixture.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from lyzr import Studio

from setup_dev_multi_kb import (
    KB_SPECS, ingest_kb,
    AGENT_ROLE, AGENT_GOAL, AGENT_INSTRUCTIONS, AGENT_PROVIDER, AGENT_TEMPERATURE,
)

# kb_ids created by the crashed run (from /tmp/multi_kb_setup.log)
EXISTING_KB_IDS = {
    1: "6a2ddc23955188b6c9726407",   # transformers      (3/3 ok)
    2: "6a2ddc76955188b6c972640f",   # llm_evaluation    (1/1 ok)
    3: "6a2ddcaa955188b6c9726413",   # dem_superres      (8/8 ok)
    4: "6a2ddd38955188b6c9726425",   # authors_project   (3/3 ok)
    5: "6a2ddda5955188b6c972642d",   # ml_textbooks      (0/3 — re-ingest liteparse)
    6: "6a2de03c955188b6c972642f",   # nlp               (1/1 ok)
    7: "6a2de14f955188b6c9726433",   # python            (3/4 — re-ingest 1 liteparse)
    8: "6a2de4c4955188b6c972643b",   # webrtc            (1/1 ok)
    9: "6a2de582955188b6c972643f",   # internship        (empty — ingest 10 liteparse)
}

# Re-ingest plans into EXISTING kb_ids (parser, list-of-basenames or None=all spec files)
REINGEST = {
    5: ("standard", None),  # all 3 ML books via liteparse
    7: ("standard", ["Python for Gemini and Bard.pdf"]),  # only the failed one
    9: ("standard", None),  # all 10 internship files via liteparse
}

# KBs to create fresh + ingest (parser)
CREATE = {
    10: "advanced",  # commercial wire scans
    11: "advanced",  # consumer wire scans
}

PARALLELISM = 4


def spec_by_idx(idx: int):
    return next(s for s in KB_SPECS if s.idx == idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=["dev", "prod", "local"], default="dev")
    ap.add_argument("--out", default="/tmp/dev_multi_kb_fixture.json")
    args = ap.parse_args()

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY required.", file=sys.stderr)
        return 2

    studio = Studio(api_key=api_key, env=args.env)
    base_url = studio._http.base_url  # type: ignore[attr-defined]
    print(f"[recover] env={args.env} base_url={base_url}")

    kb_id_by_idx: dict[int, str] = dict(EXISTING_KB_IDS)
    ingest_summary: dict[int, dict] = {}

    # 1. Re-ingest empty/partial existing KBs via LiteParse
    for idx, (parser, only_files) in REINGEST.items():
        s = spec_by_idx(idx)
        files = [f for f in s.files if os.path.exists(f)]
        if only_files is not None:
            files = [f for f in files if os.path.basename(f) in set(only_files)]
        kb_id = kb_id_by_idx[idx]
        print(f"\n[recover] KB{idx} {s.name}: re-ingest {len(files)} file(s) via {parser} into {kb_id}", flush=True)
        results = asyncio.run(ingest_kb(base_url, api_key, kb_id, files, parser, PARALLELISM))
        ok = [r for r in results if r.error is None]
        ingest_summary[idx] = {"reingested_ok": len(ok), "reingested_total": len(results)}

    # 2. Create + ingest the wire KBs via PaddleOCR
    for idx, parser in CREATE.items():
        s = spec_by_idx(idx)
        files = [f for f in s.files if os.path.exists(f)]
        print(f"\n[recover] KB{idx} {s.name}: CREATE + ingest {len(files)} file(s) via {parser}", flush=True)
        kb = studio.create_knowledge_base(
            name=f"{s.name}_{int(time.time())}",
            vector_store="qdrant", embedding_model="text-embedding-3-large",
            llm_model="gpt-4o", description=s.description,
        )
        kb_id_by_idx[idx] = kb.id
        print(f"[recover]   kb_id={kb.id}  ingesting...", flush=True)
        results = asyncio.run(ingest_kb(base_url, api_key, kb.id, files, parser, PARALLELISM))
        ok = [r for r in results if r.error is None]
        ingest_summary[idx] = {"created_ok": len(ok), "created_total": len(results)}

    # 3. Build feature configs over all 11 KBs
    print("\n[recover] Building feature configs over all 11 KBs...", flush=True)
    agentic_cfgs, oneshot_cfgs, kb_records = [], [], []
    for idx in range(1, 12):
        s = spec_by_idx(idx)
        kb_id = kb_id_by_idx[idx]
        kb = studio.get_knowledge_base(kb_id)
        agentic_cfgs.append(kb.to_agentic_config(top_k=10, retrieval_type="basic"))
        oneshot_cfgs.append(kb.to_oneshot_config(
            top_k=20, retrieval_type="basic",
            filter_fields=s.filter_fields, examples=s.examples,
        ))
        kb_records.append({
            "idx": idx, "name": s.name, "kb_id": kb_id,
            "description": s.description, "examples": s.examples,
            "filter_fields": s.filter_fields,
            "ingest": ingest_summary.get(idx, {"reused": True}),
        })

    common = dict(provider=AGENT_PROVIDER, role=AGENT_ROLE, goal=AGENT_GOAL,
                  instructions=AGENT_INSTRUCTIONS, temperature=AGENT_TEMPERATURE)
    ts = int(time.time())

    print(f"[recover] Creating agentic_rag agent ({len(agentic_cfgs)} KBs)...", flush=True)
    agent_a = studio.create_agent(
        name=f"multikb_agentic_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "agentic_rag": agentic_cfgs}, "priority": 0}],
        **common,
    )
    print(f"[recover]   agentic_agent_id={agent_a.id}", flush=True)

    print(f"[recover] Creating oneshot_rag agent ({len(oneshot_cfgs)} KBs)...", flush=True)
    agent_b = studio.create_agent(
        name=f"multikb_oneshot_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "oneshot_rag": oneshot_cfgs,
                              "planner_model": "gpt-4o-mini", "merge_top_k": 6}, "priority": 0}],
        **common,
    )
    print(f"[recover]   oneshot_agent_id={agent_b.id}", flush=True)

    fixture = {
        "env": args.env, "ts": ts,
        "agentic_agent_id": agent_a.id, "oneshot_agent_id": agent_b.id,
        "kbs": kb_records,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, default=str)
    print(f"\n[recover] Fixture -> {args.out}")
    print(f"[recover] Next: python benchmark_multi_kb.py --fixture {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
