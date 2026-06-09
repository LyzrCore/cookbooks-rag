"""
1:1 latency benchmark: oneshot_rag vs agentic_rag.

Runs the same set of queries twice against the same agent + KB — once with
`agent.run(oneshot=False)` (the legacy ReAct tool loop) and once with
`agent.run(oneshot=True)` (the new planner+synthesizer path) — and produces
a side-by-side comparison of:

  * TTFT (time to first user-visible token, via stream=True)
  * Total wall-clock
  * tool_calls count   (proves oneshot has 0; agentic typically >0)
  * Answer text (for manual quality eyeballing)

Fixtures are built fresh on each run: creates a KB, ingests a themed subset
of the rag/loadtest corpus using paddleocr or liteparse (NOT llmsherpa,
per ops constraint), creates an agent, then runs the query set.

Usage
-----

    pip install lyzr-adk
    export LYZR_API_KEY="dev-env-key"
    export LYZR_AGENT_URL="https://dev-agent.lyzr.ai"   # optional override
    python benchmark_oneshot_vs_agentic.py \\
        --parser paddleocr \\
        --corpus-glob '/Users/parshva/lyzr/rag/loadtest/corpus/pdf/medium/*Wire*.pdf' \\
        --runs 3 \\
        --out results_oneshot_vs_agentic.json

Notes
-----

* The dev env MUST be running lyzr-agent feature/oneshot-rag (or merged) —
  otherwise `oneshot=True` is silently ignored on the server, falls back to
  agentic_rag, and the numbers will look suspiciously identical.
* `--runs N` repeats every query N times per path so we can compute median /
  p95 instead of trusting one cold sample.
* Streaming is used to measure TTFT honestly. For oneshot, the planner is
  non-streaming (≤1.5s) and the synthesizer streams; for agentic, the model
  must finish each tool round-trip before any text streams.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

from lyzr import Studio


# Direct hit to the agent-side assets/upload endpoint. The SDK's kb.add_pdf
# uses the legacy rag /v3/train/pdf/ path which silently 0-chunks on the
# advanced (PaddleOCR) tier. The proper async parse pipeline is:
#   POST /v3/assets/upload (multipart) → SQS → Lambda → rag → KB chunks
ASSETS_UPLOAD_PATH = "/v3/assets/upload"
PARSE_STATUS_PATH = "/v3/assets/{asset_id}/parse-status"


def upload_and_wait(
    *,
    agent_base_url: str,
    api_key: str,
    kb_id: str,
    pdf_path: str,
    provider: str,            # 'advanced' (paddleocr) | 'standard' (liteparse)
    poll_interval_s: float = 3.0,
    poll_timeout_s: float = 600.0,
) -> dict:
    """Upload one PDF via /v3/assets/upload, poll until parse completes.

    Returns the final parse_status dict. Raises if parsing fails or times out.
    """
    headers = {"x-api-key": api_key}
    parse_config = {
        "provider": provider,
        "rag_id": kb_id,
        "extract_text": True,
        "label_pages": False,
    }
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0)) as client:
        with open(pdf_path, "rb") as fh:
            files = {"files": (os.path.basename(pdf_path), fh, "application/pdf")}
            data = {"parse_config": json.dumps(parse_config)}
            r = client.post(
                f"{agent_base_url}{ASSETS_UPLOAD_PATH}",
                headers=headers,
                files=files,
                data=data,
            )
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results") or []
        asset_id = None
        for res in results:
            if res.get("success") and res.get("asset_id"):
                asset_id = res["asset_id"]
                break
        if not asset_id:
            raise RuntimeError(f"upload returned no asset_id: {payload}")

        # Poll
        deadline = time.time() + poll_timeout_s
        last_status: dict = {}
        while time.time() < deadline:
            r = client.get(
                f"{agent_base_url}{PARSE_STATUS_PATH.format(asset_id=asset_id)}",
                headers=headers,
            )
            if r.status_code == 404:
                # status row not created yet; wait a beat
                time.sleep(poll_interval_s)
                continue
            r.raise_for_status()
            last_status = r.json()
            status = (last_status.get("parsing_status") or "").lower()
            if status in {"success", "completed", "done"}:
                return last_status
            if status in {"failed", "error"}:
                raise RuntimeError(f"parse failed for {pdf_path}: {last_status}")
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"parse-status did not complete in {poll_timeout_s}s; last={last_status}"
        )


# -----------------------------------------------------------------------------
# Default query set — wire-transfer themed since the loadtest corpus has
# a good cluster of commercial/consumer wire docs. Categorized so the report
# can split timing by question type.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchQuery:
    category: str
    question: str


DEFAULT_QUERIES: list[BenchQuery] = [
    # Factoid: single, direct fact lookups
    BenchQuery("factoid", "What documents are required to initiate a commercial wire transfer?"),
    BenchQuery("factoid", "Who needs to approve a wire transfer request?"),
    BenchQuery("factoid", "What is the approval limit for a commercial wire transfer?"),
    BenchQuery("factoid", "What information must a wire request form include about the beneficiary?"),
    BenchQuery("factoid", "What is the cutoff time for same-day wire transfers?"),
    BenchQuery("factoid", "Who is the originator on the Bravo Enterprises wire?"),

    # Comparative: requires fanout across multiple docs
    BenchQuery("comparative", "Compare the wire transfer requirements for Bravo Enterprises and Bath Planet of Chicago."),
    BenchQuery("comparative", "What is the difference between a commercial wire and a consumer wire?"),
    BenchQuery("comparative", "How do the approval workflows differ between Commercial Sample 1 and Commercial Sample 3?"),

    # Enumeration: list-style answers, often need broad recall
    BenchQuery("enumeration", "List every field a wire request form must include."),
    BenchQuery("enumeration", "What are all the parties involved in a wire transfer approval?"),
    BenchQuery("enumeration", "Enumerate the verification steps for a high-value wire."),

    # Multi-topic: separable sub-questions, planner should decompose
    BenchQuery("multi_topic", "What documents and approvals are needed, and what happens if a wire is rejected?"),
    BenchQuery("multi_topic", "How do you initiate, approve, and verify a commercial wire transfer?"),
    BenchQuery("multi_topic", "Explain wire transfer fees and the dispute process."),

    # Multi-hop: hop 2 depends on hop 1
    BenchQuery("multi_hop", "Who approved the wire transfer for the customer named in Commercial Sample 2?"),
    BenchQuery("multi_hop", "For the largest wire amount mentioned in the corpus, what was the approval chain?"),

    # Specific lookup: requires precise filter-style retrieval
    BenchQuery("specific", "What is the wire amount on the Bath Planet of Chicago request dated 01/21/2026?"),
    BenchQuery("specific", "What account number is referenced in the Consumer Sample 1 wire?"),

    # Ambiguous: terse, requires query rewriting
    BenchQuery("ambiguous", "Wire limit?"),
    BenchQuery("ambiguous", "Approval flow?"),

    # Temporal: date/time-bound
    BenchQuery("temporal", "What wire transfers were submitted in January 2026?"),

    # Citation: tests whether the synthesizer surfaces source markers
    BenchQuery("citation", "What documents mention 'OFAC screening', and what do they say about it?"),

    # Chitchat: planner should set needs_retrieval=false in oneshot
    BenchQuery("chitchat", "Hi, what can you help me with today?"),
    BenchQuery("chitchat", "Thanks, that was useful."),

    # Out-of-corpus: stresses the 'insufficient context' refusal
    BenchQuery("oos", "What is the capital of France?"),
    BenchQuery("oos", "Who won the 2025 NBA Finals?"),

    # Repetition: same query twice → tests prompt cache / planner cache behavior
    BenchQuery("repetition", "What documents are required to initiate a commercial wire transfer?"),
]


# -----------------------------------------------------------------------------
# Per-run measurement
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    category: str
    question: str
    path: str                  # "agentic_rag" | "oneshot_rag"
    run_idx: int               # 0..N-1
    ttft_s: Optional[float]
    total_s: float
    tool_calls_count: int
    response_text: str
    plan: Optional[dict]       # only populated on oneshot path
    error: Optional[str] = None


def _time_agent_run(
    agent,
    kb,
    question: str,
    oneshot: bool,
    use_persistent: bool = False,
) -> tuple[Optional[float], float, str, list[dict], Optional[dict], Optional[str]]:
    """One scored agent.run call. Streams so TTFT is real.

    Returns (ttft_s, total_s, response_text, tool_calls, plan, error).

    When `use_persistent=True`, the call assumes the agent already has the
    correct feature config baked in (created by `setup_dev_bench_agents.py`)
    and does NOT inject `knowledge_bases=` or `oneshot=True` per-call —
    those would duplicate the persistent feature on the server.
    """
    t0 = time.perf_counter()
    first_token_t: Optional[float] = None
    deltas: list[str] = []
    final_chunk_content: str = ""
    tool_calls: list[dict] = []
    plan: Optional[dict] = None
    err: Optional[str] = None

    try:
        if use_persistent:
            # Persistent agent already has the right KB feature (agentic_rag or
            # oneshot_rag) in its stored config. Pass NO knowledge_bases= and
            # NO oneshot= flag; the server-side dispatch reads the persisted
            # feature and behaves accordingly.
            kwargs = dict(message=question, stream=True)
        else:
            # Ephemeral runtime path — inject feature payload per call. Keeps
            # `with_config` planner hints so oneshot's KB catalog has context.
            kb_arg = kb.with_config(
                filter_fields=["source"],
                examples=[
                    "documents required for a commercial wire transfer",
                    "wire transfer approval workflow",
                    "wire request from Bath Planet of Chicago",
                    "manual wire transfer agreement",
                ],
            ) if oneshot else kb
            kwargs = dict(
                message=question,
                knowledge_bases=[kb_arg],
                stream=True,
            )
            if oneshot:
                kwargs["oneshot"] = True

        for chunk in agent.run(**kwargs):
            if first_token_t is None and chunk.content:
                first_token_t = time.perf_counter()
            # The SDK yields per-chunk content for streaming, then a final
            # done=True chunk whose content is the *full accumulated* string.
            # If we naïvely concatenate every chunk.content we double-count.
            if getattr(chunk, "done", False):
                final_chunk_content = chunk.content or ""
            else:
                deltas.append(chunk.content or "")
            md = getattr(chunk, "metadata", None) or {}
            if isinstance(md, dict):
                tcs = md.get("tool_calls")
                if isinstance(tcs, list) and tcs and not tool_calls:
                    tool_calls = tcs
                p = md.get("plan")
                if isinstance(p, dict) and plan is None:
                    plan = p
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    total = time.perf_counter() - t0
    ttft = (first_token_t - t0) if first_token_t is not None else None
    # Prefer per-chunk deltas; fall back to the final done=True chunk if the
    # SDK degraded to single-chunk delivery (no real streaming).
    response_text = "".join(deltas) if deltas else final_chunk_content
    return ttft, total, response_text, tool_calls, plan, err


# -----------------------------------------------------------------------------
# Fixture: KB + agent
# -----------------------------------------------------------------------------

def build_fixture(studio: Studio, pdfs: list[str], parser: str, agent_base_url: str, api_key: str):
    """Create a fresh KB, upload + parse PDFs via /v3/assets/upload, create an agent.

    Returns (agent, kb).
    """
    print(f"[fixture] Creating KB (parser={parser})...")
    kb = studio.create_knowledge_base(
        name=f"oneshot_bench_{int(time.time())}",
        vector_store="qdrant",
        embedding_model="text-embedding-3-large",
        llm_model="gpt-4o",
        # Domain-specific description so the oneshot planner's KB catalog
        # actually signals what's inside; a generic 'benchmark KB' label
        # makes the planner unable to route.
        description=(
            "Bank wire-transfer policy documents: commercial and consumer wire "
            "transfer agreements, submission checklists, approval workflows, "
            "and customer-specific wire requests (e.g., Bravo Enterprises, "
            "Bath Planet of Chicago)."
        ),
    )

    for path in pdfs:
        try:
            t0 = time.perf_counter()
            status = upload_and_wait(
                agent_base_url=agent_base_url,
                api_key=api_key,
                kb_id=kb.id,
                pdf_path=path,
                provider=parser,
            )
            dt = time.perf_counter() - t0
            print(
                f"[fixture]   ingested {os.path.basename(path)} ({dt:.1f}s, "
                f"status={status.get('parsing_status')!r})"
            )
        except Exception as exc:
            print(f"[fixture]   FAILED {os.path.basename(path)}: {exc}", file=sys.stderr)

    print("[fixture] Creating agent...")
    agent = studio.create_agent(
        name=f"oneshot_bench_agent_{int(time.time())}",
        provider="gpt-4o",
        role="Wire Transfer Operations Specialist",
        goal="Help bank staff answer questions about commercial and consumer wire transfer procedures using the supplied knowledge base.",
        instructions=(
            "Use ONLY the knowledge base for factual claims. Cite document IDs "
            "when the chunks provide them. If the knowledge base does not "
            "cover a question, say so plainly. Do not invent details."
        ),
        temperature=0.2,
    )
    print(f"[fixture] Done. agent_id={agent.id} kb_id={kb.id}")
    return agent, kb


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    sxs = sorted(xs)
    n = len(sxs)
    def pct(p: float) -> float:
        if n == 1:
            return sxs[0]
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return sxs[k]
    return {
        "n": n,
        "p50": round(statistics.median(sxs), 2),
        "p95": round(pct(95), 2),
        "min": round(sxs[0], 2),
        "max": round(sxs[-1], 2),
        "mean": round(statistics.mean(sxs), 2),
    }


def print_report(results: list[RunResult]) -> None:
    by_path: dict[str, list[RunResult]] = {"agentic_rag": [], "oneshot_rag": []}
    for r in results:
        by_path.setdefault(r.path, []).append(r)

    print("\n" + "=" * 78)
    print("  Aggregate (across all queries × runs)")
    print("=" * 78)
    print(f"{'metric':<22}{'agentic_rag':>26}{'oneshot_rag':>28}")
    for label, key in [("TTFT (s)", "ttft_s"), ("Total wall-clock (s)", "total_s")]:
        a = _stats([getattr(r, key) for r in by_path["agentic_rag"] if getattr(r, key) is not None])
        o = _stats([getattr(r, key) for r in by_path["oneshot_rag"] if getattr(r, key) is not None])
        a_fmt = f"p50={a.get('p50','-'):>5} p95={a.get('p95','-'):>5}"
        o_fmt = f"p50={o.get('p50','-'):>5} p95={o.get('p95','-'):>5}"
        print(f"{label:<22}{a_fmt:>26}{o_fmt:>28}")
    a_tcs = [r.tool_calls_count for r in by_path["agentic_rag"]]
    o_tcs = [r.tool_calls_count for r in by_path["oneshot_rag"]]
    a_tc_mean = round(statistics.mean(a_tcs), 2) if a_tcs else "-"
    o_tc_mean = round(statistics.mean(o_tcs), 2) if o_tcs else "-"
    print(f"{'tool_calls mean':<22}{a_tc_mean:>26}{o_tc_mean:>28}")

    print("\n" + "=" * 78)
    print("  Per-query (median across runs)")
    print("=" * 78)
    by_q: dict[str, dict[str, list[RunResult]]] = {}
    for r in results:
        by_q.setdefault(r.question, {}).setdefault(r.path, []).append(r)
    header = f"{'category':<13}{'question':<42}{'agentic total/TTFT':<22}{'oneshot total/TTFT':<22}{'tc(a/o)':<9}"
    print(header)
    print("-" * len(header))
    for question, paths in by_q.items():
        a_r = paths.get("agentic_rag", [])
        o_r = paths.get("oneshot_rag", [])
        cat = (a_r or o_r)[0].category if (a_r or o_r) else ""
        def med(rs: list[RunResult], k: str) -> str:
            vals = [getattr(r, k) for r in rs if getattr(r, k) is not None]
            return f"{statistics.median(vals):.1f}" if vals else "-"
        a_fmt = f"{med(a_r, 'total_s'):>5}/{med(a_r, 'ttft_s'):>5}s"
        o_fmt = f"{med(o_r, 'total_s'):>5}/{med(o_r, 'ttft_s'):>5}s"
        a_tc = statistics.median([r.tool_calls_count for r in a_r]) if a_r else "-"
        o_tc = statistics.median([r.tool_calls_count for r in o_r]) if o_r else "-"
        q_short = (question[:39] + "...") if len(question) > 42 else question
        print(f"{cat:<13}{q_short:<42}{a_fmt:<22}{o_fmt:<22}{a_tc}/{o_tc:<7}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parser",
        choices=["advanced", "standard"],
        default="advanced",
        help=(
            "Ingestion tier (lyzrparse). 'advanced' = PaddleOCR, 'standard' = LiteParse. "
            "NOT llmsherpa (ops constraint). Default: advanced."
        ),
    )
    ap.add_argument(
        "--corpus-glob",
        default="/Users/parshva/lyzr/rag/loadtest/corpus/pdf/medium/*Wire*.pdf",
        help="Glob of PDFs to ingest. Default: wire-transfer themed subset of the rag loadtest corpus.",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=6,
        help="Cap ingestion at this many PDFs (loadtest corpus can be very large).",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repeat each query this many times per path (compute median/p95). Default: 3.",
    )
    ap.add_argument(
        "--out",
        default="results_oneshot_vs_agentic.json",
        help="Where to write the raw results JSON.",
    )
    ap.add_argument(
        "--existing-agent-id",
        default=None,
        help="Skip fixture creation; use this agent + the KB it's wired to. (Mutually exclusive with --corpus-glob.)",
    )
    ap.add_argument(
        "--existing-kb-id",
        default=None,
        help="Required if --existing-agent-id is set: the KB id to query.",
    )
    ap.add_argument(
        "--env",
        choices=["dev", "prod", "local"],
        default="dev",
        help="Studio environment. Default: dev (where feature/oneshot-rag is deployed).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the query set to the first N (smoke testing). Default: run all queries.",
    )
    ap.add_argument(
        "--fixture",
        default=None,
        help=(
            "Path to a fixture JSON produced by setup_dev_bench_agents.py. "
            "When set, benchmark uses the persistent agents named in the fixture "
            "(skips fixture-building) and omits per-call knowledge_bases= so "
            "the persistent feature config drives retrieval."
        ),
    )
    ap.add_argument(
        "--probe-once",
        action="store_true",
        help=(
            "Before the streaming bench, run each query ONCE with stream=False, "
            "oneshot=True to capture response.plan (the planner's structured "
            "output) for diagnosis. Adds ~3-6s per query. Plans are written to "
            "the results JSON under 'probe_plans'."
        ),
    )
    args = ap.parse_args()

    queries = DEFAULT_QUERIES[: args.limit] if args.limit else DEFAULT_QUERIES

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY env var is required.", file=sys.stderr)
        return 2

    print(f"[setup] Studio env={args.env}")
    studio = Studio(api_key=api_key, env=args.env)
    # Agent base URL is needed for the assets/upload endpoint (not exposed by the SDK).
    agent_base_url = studio._http.base_url  # type: ignore[attr-defined]
    print(f"[setup] agent_base_url={agent_base_url}")

    # Fixture setup
    fixture_data: Optional[dict] = None
    use_persistent = False
    agent_pair: Optional[tuple] = None  # (agentic_agent, oneshot_agent) when persistent
    if args.fixture:
        with open(args.fixture) as f:
            fixture_data = json.load(f)
        kb_id = fixture_data["kb_id"]
        agentic_id = fixture_data["agentic_agent_id"]
        oneshot_id = fixture_data["oneshot_agent_id"]
        print(f"[fixture] Using persistent agents from {args.fixture}")
        print(f"   kb={kb_id}  agentic={agentic_id}  oneshot={oneshot_id}")
        agentic_agent = studio.get_agent(agentic_id)
        oneshot_agent = studio.get_agent(oneshot_id)
        kb = studio.get_knowledge_base(kb_id)
        agent_pair = (agentic_agent, oneshot_agent)
        use_persistent = True
    elif args.existing_agent_id and args.existing_kb_id:
        print(f"[fixture] Using existing agent={args.existing_agent_id} kb={args.existing_kb_id}")
        agent = studio.get_agent(args.existing_agent_id)
        kb = studio.get_knowledge_base(args.existing_kb_id)
    else:
        pdfs = sorted(glob.glob(args.corpus_glob))[: args.max_docs]
        if not pdfs:
            print(f"Error: no PDFs matched --corpus-glob {args.corpus_glob!r}", file=sys.stderr)
            return 2
        print(f"[fixture] {len(pdfs)} PDF(s) selected for ingestion:")
        for p in pdfs:
            print(f"   - {os.path.basename(p)}")
        agent, kb = build_fixture(studio, pdfs, args.parser, agent_base_url, api_key)

    # Optional probe: capture response.plan via a single non-streaming run per query.
    probe_plans: dict[str, dict] = {}
    if args.probe_once:
        probe_agent = agent_pair[1] if use_persistent else agent  # type: ignore[index]
        print(f"\n[probe] capturing planner output for {len(queries)} queries (stream=False, oneshot=True)")
        for q in queries:
            try:
                kwargs = dict(message=q.question, stream=False)
                if not use_persistent:
                    kwargs["knowledge_bases"] = [kb]
                    kwargs["oneshot"] = True
                resp = probe_agent.run(**kwargs)
                p = getattr(resp, "plan", None)
                probe_plans[q.question] = p if isinstance(p, dict) else {}
                summary = "no plan"
                if isinstance(p, dict):
                    nr = p.get("needs_retrieval")
                    sqs = p.get("sub_queries") or []
                    summary = f"needs_retrieval={nr} sub_queries={len(sqs)}"
                print(f"   [{q.category}] {q.question[:60]!r:<62} {summary}")
            except Exception as exc:
                probe_plans[q.question] = {"_error": f"{type(exc).__name__}: {exc}"}
                print(f"   [probe-err] {q.question[:60]!r}: {exc}")

    # Run benchmark
    results: list[RunResult] = []
    print(f"\n[bench] {len(queries)} queries × 2 paths × {args.runs} runs = "
          f"{len(queries) * 2 * args.runs} agent.run calls")
    for q_idx, q in enumerate(queries, 1):
        print(f"\n[bench] ({q_idx}/{len(queries)}) [{q.category}] {q.question}")
        for run_idx in range(args.runs):
            for oneshot, path in [(False, "agentic_rag"), (True, "oneshot_rag")]:
                # Pick the right agent: persistent pair if fixture, else the
                # shared ephemeral agent (which gets per-call feature injection).
                if use_persistent and agent_pair is not None:
                    target_agent = agent_pair[0] if not oneshot else agent_pair[1]
                else:
                    target_agent = agent  # noqa: F821
                ttft, total, text, tcs, plan, err = _time_agent_run(
                    target_agent, kb, q.question,
                    oneshot=oneshot,
                    use_persistent=use_persistent,
                )
                results.append(RunResult(
                    category=q.category,
                    question=q.question,
                    path=path,
                    run_idx=run_idx,
                    ttft_s=ttft,
                    total_s=total,
                    tool_calls_count=len(tcs) if tcs else 0,
                    response_text=text,
                    plan=plan,
                    error=err,
                ))
                ttft_str = f"{ttft:.2f}s" if ttft is not None else "n/a"
                err_str = f" ERR={err}" if err else ""
                print(f"   {path:<14} run={run_idx} ttft={ttft_str:<7} total={total:.2f}s tc={len(tcs) if tcs else 0}{err_str}")

    # Persist
    out_path = args.out
    if use_persistent and agent_pair is not None:
        agentic_id_out = agent_pair[0].id
        oneshot_id_out = agent_pair[1].id
    else:
        agentic_id_out = getattr(agent, "id", None)  # noqa: F821
        oneshot_id_out = None
    with open(out_path, "w") as f:
        json.dump({
            "queries": [asdict(q) for q in queries],
            "runs": args.runs,
            "parser": args.parser,
            "fixture": args.fixture,
            "use_persistent": use_persistent,
            "agentic_agent_id": agentic_id_out,
            "oneshot_agent_id": oneshot_id_out,
            "kb_id": getattr(kb, "id", None),
            "probe_plans": probe_plans,
            "results": [asdict(r) for r in results],
        }, f, indent=2, default=str)
    print(f"\n[bench] Raw results written to {out_path}")

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
