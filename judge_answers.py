"""
LLM-as-judge accuracy comparison: oneshot_rag vs agentic_rag answers.

Reads a side-by-side benchmark JSON (both paths' answer texts per query),
grades each answer with a gpt-4o judge agent on dev, and reports correctness +
groundedness per path plus head-to-head win/tie/loss.

Note on ground truth: for general-knowledge questions (transformers, webrtc,
python, nlp, ml) the judge assesses correctness directly. For corpus-specific
questions (exact wire amounts, the author's project) it can only assess
specificity/plausibility, not verify the fact — flagged in the per-query notes.

Usage:
    export LYZR_API_KEY=...
    python judge_answers.py --bench /tmp/mkb_sidebyside_v2.json --out /tmp/mkb_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from lyzr.agents import AgentModule
_o = AgentModule._make_smart_agent
def _p(self, ad, response_model=None):
    if isinstance(ad, dict) and "api_key" not in ad:
        ad["api_key"] = getattr(self._http, "api_key", None) or os.environ.get("LYZR_API_KEY")
    return _o(self, ad, response_model=response_model)
AgentModule._make_smart_agent = _p

from lyzr import Studio

JUDGE_INSTRUCTIONS = (
    "You are a strict grader comparing two AI assistant answers (A and B) to the "
    "same user question. Grade each on:\n"
    "  correctness 0-2: 0 = wrong, refused, or off-topic; 1 = partially correct "
    "or vague; 2 = correct and specific.\n"
    "  grounded true/false: true if the answer is specific and substantive; "
    "false if it is generic boilerplate, hedged, or a refusal.\n"
    "Then pick the better answer: 'A', 'B', or 'tie'.\n"
    "Output ONLY a JSON object: "
    '{"a_score":0-2,"a_grounded":bool,"b_score":0-2,"b_grounded":bool,'
    '"winner":"A|B|tie","reason":"<=25 words"}'
)


def _retry(fn, tries=5, base=4.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            if i == tries - 1 or not any(t in str(exc).lower() for t in
                ("503", "502", "504", "timeout", "temporarily", "connection", "429")):
                raise
            time.sleep(base * (2 ** i))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="/tmp/mkb_sidebyside_v2.json")
    ap.add_argument("--env", default="dev")
    ap.add_argument("--out", default="/tmp/mkb_accuracy.json")
    ap.add_argument("--throttle", type=float, default=1.5)
    args = ap.parse_args()
    if not os.environ.get("LYZR_API_KEY"):
        print("LYZR_API_KEY required", file=sys.stderr); return 2

    B = json.load(open(args.bench))["results"]
    studio = Studio(api_key=os.environ["LYZR_API_KEY"], env=args.env)
    judge = studio.create_agent(
        name=f"mkb_judge_{int(time.time())}",
        provider="gpt-4o", role="Answer Grader",
        goal="Grade and compare two RAG answers objectively.",
        instructions=JUDGE_INSTRUCTIONS, temperature=0.0,
    )
    print(f"[judge] agent={judge.id}  grading {len(B)} queries")

    verdicts = []
    for i, r in enumerate(B, 1):
        q = r["question"]
        a = (r.get("oneshot_text") or "").strip() or "(no answer)"
        b = (r.get("agentic_text") or "").strip() or "(no answer)"
        prompt = (
            f"Question: {q}\n\n"
            f"Answer A (oneshot):\n{a[:1500]}\n\n"
            f"Answer B (agentic):\n{b[:1500]}\n\n"
            "Grade now. JSON only."
        )
        try:
            resp = _retry(lambda: judge.run(message=prompt, stream=False))
            txt = (resp.response or "").strip()
            # tolerate code fences
            if txt.startswith("```"):
                txt = txt.strip("`")
                txt = txt[txt.find("{"):]
            v = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception as exc:
            v = {"error": f"{type(exc).__name__}: {exc}"}
        v["category"] = r["category"]
        v["question"] = q
        verdicts.append(v)
        w = v.get("winner", "?")
        print(f"  ({i}/{len(B)}) [{r['category']:<11}] A={v.get('a_score','?')} B={v.get('b_score','?')} win={w}", flush=True)
        time.sleep(args.throttle)

    # Aggregate
    valid = [v for v in verdicts if "error" not in v]
    def mean(key):
        xs = [v[key] for v in valid if isinstance(v.get(key), (int, float))]
        return round(sum(xs) / len(xs), 2) if xs else None
    def rate(key):
        xs = [1 for v in valid if v.get(key) is True]
        return round(100 * len(xs) / len(valid), 1) if valid else None
    from collections import Counter
    wins = Counter(v.get("winner") for v in valid)

    print("\n" + "=" * 60)
    print(f"  ACCURACY (LLM-judge, n={len(valid)} graded)")
    print("=" * 60)
    print(f"{'metric':<26}{'oneshot (A)':>16}{'agentic (B)':>16}")
    print(f"{'mean correctness /2':<26}{str(mean('a_score')):>16}{str(mean('b_score')):>16}")
    print(f"{'grounded %':<26}{str(rate('a_grounded')):>16}{str(rate('b_grounded')):>16}")
    print(f"\nhead-to-head: oneshot {wins.get('A',0)} | tie {wins.get('tie',0)} | agentic {wins.get('B',0)}")

    json.dump({"verdicts": verdicts,
               "agg": {"oneshot_correctness": mean("a_score"), "agentic_correctness": mean("b_score"),
                       "oneshot_grounded_pct": rate("a_grounded"), "agentic_grounded_pct": rate("b_grounded"),
                       "wins": dict(wins)}},
              open(args.out, "w"), indent=2, default=str)
    print(f"\n[judge] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
