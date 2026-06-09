"""
One-time setup of the dev environment fixture used by `benchmark_oneshot_vs_agentic.py`.

Builds a single shared KB ingested from the loadtest corpus (paddleocr by default,
"advanced" tier of lyzrparse), then creates two persistent agents wired to that KB:

  * `bench_agentic_rag_<ts>` — `feature.config.agentic_rag = [...]` (legacy ReAct loop)
  * `bench_oneshot_rag_<ts>` — `feature.config.oneshot_rag = [...]` (planner+synthesizer)

Both agents share the same role, goal, instructions, provider, and temperature so any
benchmark delta is attributable to the retrieval path, not the agent persona.

The script writes a fixture JSON the benchmark reads with `--fixture`:

    {"kb_id": "...", "agentic_agent_id": "...", "oneshot_agent_id": "...",
     "env": "dev", "corpus_root": "...", "uploaded_files": [...]}

Why a one-time setup vs. inline-in-the-harness fixture? PaddleOCR ingestion of the
full loadtest corpus is slow (~10-20 minutes). Building once and reusing across
benchmark runs is faster, cheaper, and avoids paddleocr saturation from concurrent
ephemeral KBs piling up on the dev tier.

Usage
-----

    export LYZR_API_KEY="dev-key"
    python setup_dev_bench_agents.py                         # full corpus
    python setup_dev_bench_agents.py --max-files 6           # smoke setup
    python setup_dev_bench_agents.py --include-ext pdf       # PDFs only
    python setup_dev_bench_agents.py --parser standard       # liteparse instead

Outputs
-------

  /tmp/dev_bench_fixture.json  (path overridable via --out)

Then run the benchmark:

    python benchmark_oneshot_vs_agentic.py --fixture /tmp/dev_bench_fixture.json --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from lyzr import Studio

# Reuse the upload+poll helper already in the bench harness; both files live
# in the same dir so a sibling import is fine.
from benchmark_oneshot_vs_agentic import (  # type: ignore[import-not-found]
    ASSETS_UPLOAD_PATH,
    PARSE_STATUS_PATH,
)


# ---------------------------------------------------------------------------
# Shared persona for both agents — only the feature config differs.
# ---------------------------------------------------------------------------

BENCH_AGENT_ROLE = "Enterprise Document Specialist"
BENCH_AGENT_GOAL = (
    "Answer staff questions about commercial and consumer wire-transfer "
    "procedures, internship policies, technical training materials, and the "
    "other enterprise documents loaded into the knowledge base."
)
BENCH_AGENT_INSTRUCTIONS = (
    "Use the configured knowledge base to ground every factual claim. Cite "
    "the document(s) you used. If the knowledge base does not cover a "
    "question, explain plainly what is missing instead of refusing flatly. "
    "Do not invent details, dates, or amounts."
)
BENCH_AGENT_PROVIDER = "gpt-4o"
BENCH_AGENT_TEMPERATURE = 0.2

KB_NAME_TEMPLATE = "oneshot_bench_kb_{ts}"
KB_DESCRIPTION = (
    "Mixed-domain enterprise corpus from rag/loadtest: commercial and "
    "consumer wire-transfer policies and request samples, Python/ML "
    "textbook excerpts, internship reports, and product certificates. "
    "Use for staff questions about wire-transfer procedures, training "
    "materials, and operational documents."
)
KB_FILTER_FIELDS = ["source"]
KB_EXAMPLES = [
    "documents required for a commercial wire transfer",
    "wire transfer approval workflow",
    "OFAC screening process",
    "wire request from Bath Planet of Chicago",
    "Bravo Enterprises wire transfer",
]


# ---------------------------------------------------------------------------
# Async upload pool — paddleocr saturates if we fan out too aggressively, so
# the parallelism cap defaults to 4.
# ---------------------------------------------------------------------------


@dataclass
class UploadResult:
    path: str
    asset_id: Optional[str]
    parsing_status: Optional[str]
    duration_s: float
    error: Optional[str] = None


async def _async_upload_one(
    client: httpx.AsyncClient,
    agent_base_url: str,
    api_key: str,
    kb_id: str,
    pdf_path: str,
    provider: str,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> UploadResult:
    t0 = time.perf_counter()
    headers = {"x-api-key": api_key}
    parse_config = {
        "provider": provider,
        "rag_id": kb_id,
        "extract_text": True,
        "label_pages": False,
    }
    mime = "application/pdf" if pdf_path.lower().endswith(".pdf") else "application/octet-stream"
    try:
        with open(pdf_path, "rb") as fh:
            files = {"files": (os.path.basename(pdf_path), fh.read(), mime)}
        data = {"parse_config": json.dumps(parse_config)}
        r = await client.post(
            f"{agent_base_url}{ASSETS_UPLOAD_PATH}",
            headers=headers,
            files=files,
            data=data,
        )
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results") or []
        asset_id = next((res.get("asset_id") for res in results if res.get("success")), None)
        if not asset_id:
            return UploadResult(pdf_path, None, None, time.perf_counter() - t0,
                                 error=f"upload returned no asset_id: {payload}")

        # Poll status
        deadline = time.time() + poll_timeout_s
        last_status_payload: dict = {}
        while time.time() < deadline:
            sr = await client.get(
                f"{agent_base_url}{PARSE_STATUS_PATH.format(asset_id=asset_id)}",
                headers=headers,
            )
            if sr.status_code == 404:
                await asyncio.sleep(poll_interval_s)
                continue
            sr.raise_for_status()
            last_status_payload = sr.json()
            status = (last_status_payload.get("parsing_status") or "").lower()
            if status in {"success", "completed", "done"}:
                return UploadResult(pdf_path, asset_id, status,
                                     time.perf_counter() - t0)
            if status in {"failed", "error"}:
                return UploadResult(pdf_path, asset_id, status,
                                     time.perf_counter() - t0,
                                     error=f"parse failed: {last_status_payload}")
            await asyncio.sleep(poll_interval_s)
        return UploadResult(pdf_path, asset_id, None,
                             time.perf_counter() - t0,
                             error=f"parse-status timeout; last={last_status_payload}")
    except Exception as exc:
        return UploadResult(pdf_path, None, None,
                             time.perf_counter() - t0,
                             error=f"{type(exc).__name__}: {exc}")


async def ingest_corpus(
    *,
    agent_base_url: str,
    api_key: str,
    kb_id: str,
    files: list[str],
    provider: str,
    parallelism: int,
    poll_interval_s: float = 4.0,
    poll_timeout_s: float = 900.0,
) -> list[UploadResult]:
    """Upload files with a Semaphore-bounded async pool."""
    sem = asyncio.Semaphore(parallelism)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0),
    ) as client:
        async def bound(path: str) -> UploadResult:
            async with sem:
                t0 = time.perf_counter()
                r = await _async_upload_one(
                    client=client,
                    agent_base_url=agent_base_url,
                    api_key=api_key,
                    kb_id=kb_id,
                    pdf_path=path,
                    provider=provider,
                    poll_interval_s=poll_interval_s,
                    poll_timeout_s=poll_timeout_s,
                )
                label = os.path.basename(path)
                if r.error:
                    print(f"  [FAIL]  {label}  ({r.duration_s:.1f}s)  err={r.error[:120]}")
                else:
                    print(f"  [ok]    {label}  ({r.duration_s:.1f}s, status={r.parsing_status!r})")
                return r

        return await asyncio.gather(*[bound(p) for p in files])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collect_files(corpus_root: str, include_ext: list[str], max_files: Optional[int]) -> list[str]:
    """Walk corpus_root recursively, collect matching files, return sorted paths."""
    matches: list[str] = []
    for ext in include_ext:
        ext = ext.lstrip(".").lower()
        matches.extend(glob.glob(os.path.join(corpus_root, "**", f"*.{ext}"), recursive=True))
    matches = sorted(set(matches))
    if max_files:
        matches = matches[:max_files]
    return matches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env",
        choices=["dev", "prod", "local"],
        default="dev",
        help="Studio environment. Default: dev (where feature/oneshot-rag is deployed).",
    )
    ap.add_argument(
        "--corpus-root",
        default="/Users/parshva/lyzr/rag/loadtest/corpus",
        help="Root dir to walk recursively for ingestible files.",
    )
    ap.add_argument(
        "--include-ext",
        default="pdf,docx",
        help="Comma-separated file extensions to ingest. Default: pdf,docx (xlsx routes through Excel tooling; pptx coverage is patchy).",
    )
    ap.add_argument(
        "--parser",
        choices=["advanced", "standard"],
        default="advanced",
        help="lyzrparse tier. advanced=PaddleOCR (default, user-stated preference); standard=LiteParse.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap ingestion at N files. Useful for smoke setups.",
    )
    ap.add_argument(
        "--parallel-uploads",
        type=int,
        default=4,
        help="Concurrent uploads against /v3/assets/upload. paddleocr saturates fast; 4 is a safe default.",
    )
    ap.add_argument(
        "--out",
        default="/tmp/dev_bench_fixture.json",
        help="Where to write the fixture JSON the benchmark reads with --fixture.",
    )
    ap.add_argument(
        "--kb-name",
        default=None,
        help="Override KB name. Default: oneshot_bench_kb_<unix_ts>.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the file list and the agent specs, do nothing.",
    )
    args = ap.parse_args()

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY env var is required.", file=sys.stderr)
        return 2

    include_ext = [e.strip() for e in args.include_ext.split(",") if e.strip()]
    files = collect_files(args.corpus_root, include_ext, args.max_files)
    if not files:
        print(f"Error: no files matched {args.corpus_root!r} include_ext={include_ext}", file=sys.stderr)
        return 2

    print(f"[setup] env={args.env}  corpus_root={args.corpus_root}")
    print(f"[setup] {len(files)} file(s) to ingest (parser={args.parser}, parallelism={args.parallel_uploads}):")
    for p in files:
        print(f"   - {os.path.relpath(p, args.corpus_root)}")
    if args.dry_run:
        print("[dry-run] not creating KB / agents / fixture.")
        return 0

    studio = Studio(api_key=api_key, env=args.env)
    agent_base_url = studio._http.base_url  # type: ignore[attr-defined]
    print(f"[setup] agent_base_url={agent_base_url}")

    ts = int(time.time())
    kb_name = args.kb_name or KB_NAME_TEMPLATE.format(ts=ts)
    print(f"[setup] Creating KB '{kb_name}'...")
    kb = studio.create_knowledge_base(
        name=kb_name,
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        description=KB_DESCRIPTION,
    )
    print(f"[setup] kb_id={kb.id}")

    print(f"[setup] Ingesting {len(files)} file(s)...")
    t_ingest_start = time.perf_counter()
    upload_results = asyncio.run(
        ingest_corpus(
            agent_base_url=agent_base_url,
            api_key=api_key,
            kb_id=kb.id,
            files=files,
            provider=args.parser,
            parallelism=args.parallel_uploads,
        )
    )
    t_ingest = time.perf_counter() - t_ingest_start
    ok = sum(1 for r in upload_results if r.error is None)
    print(f"[setup] Ingestion finished in {t_ingest:.1f}s — {ok}/{len(upload_results)} ok")

    # Sanity check: how many docs landed in the KB?
    try:
        docs = kb.list_documents()
        print(f"[setup] KB now reports {len(docs)} document(s).")
    except Exception as e:
        print(f"[setup] WARN: list_documents failed: {e}")

    # ---- Build feature configs ----
    agentic_kb_config = kb.to_agentic_config(top_k=10, retrieval_type="basic")
    oneshot_kb_config = kb.to_oneshot_config(
        top_k=20,
        retrieval_type="basic",
        filter_fields=KB_FILTER_FIELDS,
        examples=KB_EXAMPLES,
    )

    common_agent_kwargs = dict(
        provider=BENCH_AGENT_PROVIDER,
        role=BENCH_AGENT_ROLE,
        goal=BENCH_AGENT_GOAL,
        instructions=BENCH_AGENT_INSTRUCTIONS,
        temperature=BENCH_AGENT_TEMPERATURE,
    )

    print(f"[setup] Creating Agent A (agentic_rag baseline)...")
    agent_a = studio.create_agent(
        name=f"bench_agentic_rag_{ts}",
        features=[{
            "type": "KNOWLEDGE_BASE",
            "config": {"lyzr_rag": {}, "agentic_rag": [agentic_kb_config]},
            "priority": 0,
        }],
        **common_agent_kwargs,
    )
    print(f"[setup] agentic_agent_id={agent_a.id}")

    print(f"[setup] Creating Agent B (oneshot_rag candidate)...")
    agent_b = studio.create_agent(
        name=f"bench_oneshot_rag_{ts}",
        features=[{
            "type": "KNOWLEDGE_BASE",
            "config": {
                "lyzr_rag": {},
                "oneshot_rag": [oneshot_kb_config],
                "planner_model": "gpt-4o-mini",
                "merge_top_k": 6,
            },
            "priority": 0,
        }],
        **common_agent_kwargs,
    )
    print(f"[setup] oneshot_agent_id={agent_b.id}")

    # ---- Persist fixture ----
    fixture = {
        "env": args.env,
        "corpus_root": args.corpus_root,
        "kb_id": kb.id,
        "kb_name": kb.name,
        "agentic_agent_id": agent_a.id,
        "oneshot_agent_id": agent_b.id,
        "parser": args.parser,
        "ts": ts,
        "uploaded_files": [
            {"path": r.path, "asset_id": r.asset_id, "status": r.parsing_status,
             "duration_s": round(r.duration_s, 2), "error": r.error}
            for r in upload_results
        ],
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, default=str)
    print(f"\n[setup] Fixture written to {args.out}")
    print(f"[setup] Now run: python benchmark_oneshot_vs_agentic.py --fixture {args.out} --runs 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
