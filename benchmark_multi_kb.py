"""
Multi-KB benchmark: oneshot_rag vs agentic_rag across 11 knowledge bases.

Two measurement passes per query:
  1. PROBE  (oneshot, stream=False)  -> captures the planner's `plan`
     (needs_retrieval + sub_queries[].kb_id) for ROUTING-ACCURACY scoring.
  2. BENCH  (both paths, stream=True) -> TTFT, total wall-clock, refusal.

Routing accuracy is the headline multi-KB metric the single-KB run could not
measure: did the planner pick the RIGHT knowledge base(s)? Ground truth comes
from multi_kb_queries.expected_kb.

Usage:
    export LYZR_API_KEY="dev-key"
    python benchmark_multi_kb.py --fixture /tmp/dev_multi_kb_fixture.json --out /tmp/multi_kb_results.json
    python benchmark_multi_kb.py --fixture ... --limit 10          # smoke
    python benchmark_multi_kb.py --fixture ... --no-bench          # routing only (probe pass)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

# SDK workaround: GET /v3/agents/{id} intermittently omits `api_key`, which the
# pydantic Agent model requires. Inject the api_key the HTTP client already holds.
from lyzr.agents import AgentModule
_orig_make = AgentModule._make_smart_agent
def _patched_make(self, agent_data, response_model=None):
    if isinstance(agent_data, dict) and "api_key" not in agent_data:
        agent_data["api_key"] = getattr(self._http, "api_key", None) or os.environ.get("LYZR_API_KEY")
    return _orig_make(self, agent_data, response_model=response_model)
AgentModule._make_smart_agent = _patched_make

from lyzr import Studio

from multi_kb_queries import QUERIES, MKQuery
from setup_dev_multi_kb import KB_SPECS

# Build basename -> KB-name map for routing inference. The oneshot planner's
# chosen KB(s) are observed indirectly via which documents were retrieved:
# raw_response.module_outputs.documents[].metadata.source. That is the most
# meaningful "did the right KB's content reach the answer" signal.
def _norm(name: str) -> str:
    import os as _os
    b = _os.path.basename(str(name or "")).lower().strip()
    if b.endswith(".pdf"):
        b = b[:-4]
    # normalize spaces/underscores/dashes so server-side renames still match
    for ch in (" ", "_", "-", ".", ","):
        b = b.replace(ch, "")
    return b

FILE2KB = {}
for _s in KB_SPECS:
    for _f in _s.files:
        FILE2KB[_norm(_f)] = _s.name

# (normalized_basename, kb_name) pairs for inferring which KB an agentic answer
# drew from — agentic_rag cites source filenames like "[Source: Document 1, X.pdf]".
_BASENAME_KB = [(_norm(_f), _s.name) for _s in KB_SPECS for _f in _s.files]


def kbs_cited_in_text(text: str) -> list[str]:
    """Best-effort: which KBs are referenced by filename citations in an answer."""
    if not text:
        return []
    t = _norm(text)  # strips spaces/_/-/. and lowercases — matches cited filenames
    found = set()
    for base, kb in _BASENAME_KB:
        if len(base) >= 8 and base in t:
            found.add(kb)
    return sorted(found)

REFUSAL_TAG = "[insufficient_context]"
REFUSAL_PHRASES = ["i don't have enough information", "do not provide", "not enough information"]
ANTI_PATTERNS = ["the documents say", "according to the knowledge", "based on the provided"]

ROUTING_CATS = {"factoid", "enumeration", "specific", "sibling", "cross_kb"}
SINGLE_CATS = {"factoid", "enumeration", "specific"}


def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return REFUSAL_TAG in t[:120] or any(p in t for p in REFUSAL_PHRASES)


@dataclass
class QResult:
    category: str
    question: str
    expected: list
    # routing (from probe, inferred from retrieved documents' sources)
    needs_retrieval: Optional[bool] = None
    planned_kbs: list = field(default_factory=list)
    n_docs: int = 0
    plan_error: Optional[str] = None
    # bench (from stream)
    oneshot_ttft: Optional[float] = None
    oneshot_total: Optional[float] = None
    oneshot_text: str = ""
    oneshot_refusal: bool = False
    agentic_ttft: Optional[float] = None
    agentic_total: Optional[float] = None
    agentic_text: str = ""
    agentic_refusal: bool = False
    agentic_routed_kbs: list = field(default_factory=list)  # inferred from cited filenames


_TRANSIENT = ("503", "502", "504", "service temporarily", "timeout", "timed out",
              "temporarily unavailable", "connection", "rate limit", "429")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in _TRANSIENT)


def _retry(fn, tries: int = 5, base: float = 4.0):
    """Call fn(); retry on transient errors (503/502/504/timeout/conn) with
    exponential backoff. Re-raises the last non-transient error or after tries."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient(exc) or i == tries - 1:
                raise
            time.sleep(base * (2 ** i))  # 4, 8, 16, 32s
    if last:
        raise last


def stream_run(agent, message: str):
    """Return (ttft, total, text, error). Persistent agent, no per-call KBs.
    Retries the whole stream on a transient error *before any token* arrived."""
    def attempt():
        t0 = time.perf_counter()
        first = None
        deltas, final = [], ""
        for chunk in agent.run(message=message, stream=True):
            if first is None and chunk.content:
                first = time.perf_counter()
            if getattr(chunk, "done", False):
                final = chunk.content or ""
            else:
                deltas.append(chunk.content or "")
        total = time.perf_counter() - t0
        ttft = (first - t0) if first else None
        # The SDK emits incremental deltas then a final done=True chunk with the
        # FULL accumulated text. On some paths the deltas arrive truncated while
        # the final chunk is complete — so use whichever is longer as the
        # authoritative answer text. (TTFT still comes from the first delta.)
        joined = "".join(deltas)
        text = final if len(final) >= len(joined) else joined
        return ttft, total, text

    try:
        ttft, total, text = _retry(attempt)
        return ttft, total, text, None
    except Exception as exc:
        return None, 0.0, "", f"{type(exc).__name__}: {exc}"


def probe_routing(agent, message: str):
    """Non-streaming oneshot call. Prefer the AUTHORITATIVE routing signal the
    module now surfaces (module_outputs.plan.routed_kbs / .routed_kbs — actual
    KB names that contributed). Fall back to inferring from retrieved document
    sources for older deployments that don't surface it.

    Returns (retrieved_any: bool|None, routed_kb_names: list[str], n_docs, error).
    """
    try:
        resp = _retry(lambda: agent.run(message=message, stream=False))
        rr = getattr(resp, "raw_response", None) or {}
        mo = rr.get("module_outputs") if isinstance(rr, dict) else None
        mo = mo or {}
        docs = mo.get("documents") or []

        # 1. Authoritative: the module reports the KB names it routed to.
        routed = mo.get("routed_kbs")
        if not routed:
            plan = mo.get("plan") or {}
            routed = plan.get("routed_kbs") if isinstance(plan, dict) else None
        if routed:
            return (len(docs) > 0), sorted(set(routed)), len(docs), None

        # 2. Fallback: infer from retrieved document sources.
        inferred = set()
        for d in docs:
            if not isinstance(d, dict):
                continue
            md = d.get("metadata") or {}
            src = md.get("source") or md.get("doc_id") or ""
            kb = FILE2KB.get(_norm(src))
            if kb:
                inferred.add(kb)
        return (len(docs) > 0), sorted(inferred), len(docs), None
    except Exception as exc:
        return None, [], 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--env", choices=["dev", "prod", "local"], default="dev")
    ap.add_argument("--out", default="/tmp/multi_kb_results.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-cat", type=int, default=None,
                    help="Stratified sample: first N queries of each category (for a tractable latency bench).")
    ap.add_argument("--coverage", action="store_true",
                    help="KB-coverage subset: 1 single-target query per KB + 4 cross_kb + 3 oos + 3 chitchat. "
                         "Ensures every KB is exercised once per path for the side-by-side.")
    ap.add_argument("--no-bench", action="store_true", help="Routing probe only; skip streaming latency pass.")
    ap.add_argument("--no-probe", action="store_true", help="Latency only; skip routing probe.")
    ap.add_argument("--throttle", type=float, default=1.5,
                    help="Seconds to sleep between queries (avoid overloading dev -> 503s). Default 1.5.")
    args = ap.parse_args()

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY required.", file=sys.stderr)
        return 2

    fx = json.load(open(args.fixture))
    rag2name = {k["kb_id"]: k["name"] for k in fx["kbs"]}
    print(f"[setup] fixture has {len(rag2name)} KBs; "
          f"agentic={fx['agentic_agent_id']} oneshot={fx['oneshot_agent_id']}")

    studio = Studio(api_key=api_key, env=args.env)
    agentic = studio.get_agent(fx["agentic_agent_id"])
    oneshot = studio.get_agent(fx["oneshot_agent_id"])

    if args.coverage:
        # One single-target query per KB (first match), then 4 cross_kb, 3 oos, 3 chitchat.
        queries = []
        for kb in [k["name"] for k in fx["kbs"]]:
            hit = next((q for q in QUERIES
                        if len(q.expected_kb) == 1 and next(iter(q.expected_kb)) == kb
                        and q.category in ("factoid", "enumeration", "specific")), None)
            if hit:
                queries.append(hit)
        for cat, n in (("cross_kb", 4), ("oos", 3), ("chitchat", 3)):
            queries += [q for q in QUERIES if q.category == cat][:n]
    elif args.sample_per_cat:
        from collections import defaultdict as _dd
        seen = _dd(int)
        queries = []
        for q in QUERIES:
            if seen[q.category] < args.sample_per_cat:
                queries.append(q)
                seen[q.category] += 1
    else:
        queries = QUERIES[: args.limit] if args.limit else QUERIES
    print(f"[bench] {len(queries)} queries  (probe={'off' if args.no_probe else 'on'}, "
          f"bench={'off' if args.no_bench else 'on'})")

    results: list[QResult] = []
    for i, q in enumerate(queries, 1):
        r = QResult(category=q.category, question=q.question, expected=sorted(q.expected_kb))

        if not args.no_probe:
            retrieved_any, routed_kbs, n_docs, perr = probe_routing(oneshot, q.question)
            r.needs_retrieval = retrieved_any
            r.planned_kbs = routed_kbs
            r.n_docs = n_docs
            r.plan_error = perr

        if not args.no_bench:
            a_ttft, a_total, a_text, a_err = stream_run(agentic, q.question)
            r.agentic_ttft, r.agentic_total, r.agentic_text = a_ttft, a_total, a_text
            r.agentic_refusal = is_refusal(a_text)
            r.agentic_routed_kbs = kbs_cited_in_text(a_text)
            o_ttft, o_total, o_text, o_err = stream_run(oneshot, q.question)
            r.oneshot_ttft, r.oneshot_total, r.oneshot_text = o_ttft, o_total, o_text
            r.oneshot_refusal = is_refusal(o_text)

        results.append(r)
        ab = lambda s: s.replace("kb_", "").replace("_", " ")[:18]
        exp = ",".join(ab(e) for e in r.expected) or "—"
        plan = ",".join(ab(p) for p in r.planned_kbs) or "—"
        hit = "✓" if (set(r.expected) & set(r.planned_kbs)) else ("·" if not r.expected else "✗")
        errflag = " ERR" if r.plan_error else ""
        print(f"  ({i}/{len(queries)}) [{q.category:<11}] {hit} n={r.n_docs:<2} exp=[{exp}] plan=[{plan}]{errflag}", flush=True)
        if args.throttle:
            time.sleep(args.throttle)

    json.dump(
        {"fixture": args.fixture, "n": len(results),
         "kbs": [k["name"] for k in fx["kbs"]],
         "results": [asdict(r) for r in results]},
        open(args.out, "w"), indent=2, default=str,
    )
    print(f"\n[bench] results -> {args.out}")
    report(results)
    return 0


def report(results: list[QResult]) -> None:
    def planned(r): return set(r.planned_kbs)
    def expected(r): return set(r.expected)

    # ---- routing ----
    single = [r for r in results if r.category in SINGLE_CATS and r.expected]
    sibling = [r for r in results if r.category == "sibling"]
    cross = [r for r in results if r.category == "cross_kb"]
    oos = [r for r in results if r.category == "oos"]
    chit = [r for r in results if r.category == "chitchat"]

    def hit_rate(rs):
        if not rs: return None
        return round(100 * sum(1 for r in rs if expected(r) & planned(r)) / len(rs), 1)

    def exact_rate(rs):
        if not rs: return None
        return round(100 * sum(1 for r in rs if planned(r) == expected(r)) / len(rs), 1)

    def recall(rs):
        if not rs: return None
        vals = [len(expected(r) & planned(r)) / len(expected(r)) for r in rs if expected(r)]
        return round(100 * statistics.mean(vals), 1) if vals else None

    print("\n" + "=" * 80)
    print("  ROUTING ACCURACY (oneshot planner)")
    print("=" * 80)
    print(f"  Single-target hit-rate (right KB present)   : {hit_rate(single)}%   (n={len(single)})")
    print(f"  Single-target exact-match (only right KB)    : {exact_rate(single)}%")
    print(f"  Sibling hit-rate (right one of close pair)   : {hit_rate(sibling)}%   (n={len(sibling)})")
    print(f"  Sibling exact-match (precision under tension): {exact_rate(sibling)}%")
    print(f"  Cross-KB recall (fan-out to all needed KBs)  : {recall(cross)}%   (n={len(cross)})")
    print(f"  Cross-KB any-hit                             : {hit_rate(cross)}%")

    # abstain behavior
    if oos:
        ab = round(100 * sum(1 for r in oos if (r.needs_retrieval is False) or not r.planned_kbs) / len(oos), 1)
        print(f"  OOS correct-abstain (no/empty retrieval)     : {ab}%   (n={len(oos)})")
    if chit:
        nr_false = round(100 * sum(1 for r in chit if r.needs_retrieval is False) / len(chit), 1)
        print(f"  Chitchat needs_retrieval=False rate          : {nr_false}%   (n={len(chit)})")

    # ---- per-KB routing hit-rate ----
    print("\n  Per-KB hit-rate (single-target + sibling queries whose ground truth is that KB):")
    by_kb = {}
    for r in results:
        if r.category in (SINGLE_CATS | {"sibling"}) and len(r.expected) == 1:
            kb = r.expected[0]
            by_kb.setdefault(kb, []).append(1 if (expected(r) & planned(r)) else 0)
    for kb in sorted(by_kb):
        hits = by_kb[kb]
        print(f"    {kb:<36} {round(100*sum(hits)/len(hits),1):>5}%  ({sum(hits)}/{len(hits)})")

    # ---- latency + refusal ----
    benched = [r for r in results if r.oneshot_ttft is not None or r.agentic_ttft is not None]
    if benched:
        def pct(xs, q):
            if not xs: return 0.0
            s = sorted(xs); k = max(0, min(len(s)-1, int(round(q/100*(len(s)-1))))); return round(s[k], 2)
        o_ttft = [r.oneshot_ttft for r in benched if r.oneshot_ttft]
        a_ttft = [r.agentic_ttft for r in benched if r.agentic_ttft]
        o_tot = [r.oneshot_total for r in benched if r.oneshot_total]
        a_tot = [r.agentic_total for r in benched if r.agentic_total]
        o_ref = round(100 * sum(1 for r in benched if r.oneshot_refusal) / len(benched), 1)
        a_ref = round(100 * sum(1 for r in benched if r.agentic_refusal) / len(benched), 1)
        print("\n" + "=" * 80)
        print("  LATENCY & REFUSAL")
        print("=" * 80)
        print(f"{'metric':<26}{'agentic_rag':>18}{'oneshot_rag':>18}")
        print(f"{'TTFT p50 (s)':<26}{pct(a_ttft,50):>18}{pct(o_ttft,50):>18}")
        print(f"{'TTFT p95 (s)':<26}{pct(a_ttft,95):>18}{pct(o_ttft,95):>18}")
        print(f"{'Total p50 (s)':<26}{pct(a_tot,50):>18}{pct(o_tot,50):>18}")
        print(f"{'Total p95 (s)':<26}{pct(a_tot,95):>18}{pct(o_tot,95):>18}")
        print(f"{'Refusal rate (%)':<26}{a_ref:>18}{o_ref:>18}")


if __name__ == "__main__":
    sys.exit(main())
