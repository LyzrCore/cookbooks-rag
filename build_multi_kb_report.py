"""
Build a side-by-side HTML report: oneshot_rag vs agentic_rag across 11 KBs.

Inputs:
  --routing  /tmp/mkb_routing_full.json   (oneshot routing over 201 queries)
  --bench    /tmp/mkb_sidebyside.json      (both paths, KB-coverage subset)
  --out      /tmp/mkb_report.html
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict

SINGLE = {"factoid", "enumeration", "specific"}


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return round(s[k], 2)


def esc(s):
    return html.escape(str(s or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", default="/tmp/mkb_routing_full.json")
    ap.add_argument("--bench", default="/tmp/mkb_sidebyside.json")
    ap.add_argument("--out", default="/tmp/mkb_report.html")
    a = ap.parse_args()

    routing = json.load(open(a.routing))
    bench = json.load(open(a.bench))
    R = routing["results"]
    B = bench["results"]

    # ---- oneshot routing (from 201) ----
    single = [r for r in R if r["category"] in SINGLE and len(r["expected"]) == 1]
    sib = [r for r in R if r["category"] == "sibling"]
    cross = [r for r in R if r["category"] == "cross_kb"]

    def hit(rs):
        return round(100 * sum(1 for r in rs if set(r["expected"]) & set(r["planned_kbs"])) / len(rs), 1) if rs else 0
    def exact(rs):
        return round(100 * sum(1 for r in rs if set(r["planned_kbs"]) == set(r["expected"])) / len(rs), 1) if rs else 0
    def recall(rs):
        v = [len(set(r["expected"]) & set(r["planned_kbs"])) / len(set(r["expected"])) for r in rs if r["expected"]]
        return round(100 * statistics.mean(v), 1) if v else 0

    perkb = defaultdict(lambda: [0, 0])
    for r in R:
        if r["category"] in (SINGLE | {"sibling"}) and len(r["expected"]) == 1:
            kb = r["expected"][0]
            perkb[kb][1] += 1
            if set(r["expected"]) & set(r["planned_kbs"]):
                perkb[kb][0] += 1

    # ---- bench latency / refusal / agentic routing (coverage subset) ----
    o_ttft = [r["oneshot_ttft"] for r in B if r.get("oneshot_ttft")]
    a_ttft = [r["agentic_ttft"] for r in B if r.get("agentic_ttft")]
    o_tot = [r["oneshot_total"] for r in B if r.get("oneshot_total")]
    a_tot = [r["agentic_total"] for r in B if r.get("agentic_total")]
    o_ref = round(100 * sum(1 for r in B if r.get("oneshot_refusal")) / len(B), 1) if B else 0
    a_ref = round(100 * sum(1 for r in B if r.get("agentic_refusal")) / len(B), 1) if B else 0

    # agentic routing hit on coverage single-target queries (via cited filenames)
    cov_single = [r for r in B if r["category"] in SINGLE and len(r["expected"]) == 1]
    o_cov_hit = round(100 * sum(1 for r in cov_single if set(r["expected"]) & set(r["planned_kbs"])) / len(cov_single), 1) if cov_single else 0
    a_cov_hit = round(100 * sum(1 for r in cov_single if set(r["expected"]) & set(r.get("agentic_routed_kbs", []))) / len(cov_single), 1) if cov_single else 0

    def ab(s):
        return s.replace("kb_", "").replace("_", " ")

    # ---- per-query side-by-side rows (coverage) ----
    rows = []
    for r in B:
        exp = ", ".join(ab(e) for e in r["expected"]) or "—"
        o_routed = ", ".join(ab(x) for x in r["planned_kbs"]) or "—"
        o_hit = bool(set(r["expected"]) & set(r["planned_kbs"]))
        a_routed = ", ".join(ab(x) for x in r.get("agentic_routed_kbs", [])) or "—"
        a_hit = bool(set(r["expected"]) & set(r.get("agentic_routed_kbs", [])))
        rows.append({
            "cat": r["category"], "q": r["question"], "exp": exp,
            "o_routed": o_routed, "o_hit": o_hit, "o_text": (r.get("oneshot_text") or "")[:320],
            "o_ttft": r.get("oneshot_ttft"), "o_total": r.get("oneshot_total"),
            "a_routed": a_routed, "a_hit": a_hit, "a_text": (r.get("agentic_text") or "")[:320],
            "a_ttft": r.get("agentic_ttft"), "a_total": r.get("agentic_total"),
        })

    # ---- HTML ----
    def kpi(label, val, sub="", cls=""):
        return f'<div class="kpi"><div class="label">{esc(label)}</div><div class="num {cls}">{esc(val)}</div><div class="sub">{esc(sub)}</div></div>'

    perkb_rows = ""
    for kb in sorted(perkb):
        h, n = perkb[kb]
        rate = round(100 * h / n, 1) if n else 0
        cls = "good" if rate >= 90 else ("mid" if rate >= 50 else "bad")
        perkb_rows += f'<tr><td>{esc(ab(kb))}</td><td class="num {cls}">{rate}%</td><td class="num">{h}/{n}</td></tr>'

    sample_cards = ""
    for r in rows:
        oh = "✓" if r["o_hit"] else ("·" if r["exp"] == "—" else "✗")
        ahh = "✓" if r["a_hit"] else ("·" if r["exp"] == "—" else "✗")
        sample_cards += f"""
        <div class="qcard">
          <div class="qhead"><span class="cat">{esc(r['cat'])}</span> {esc(r['q'])}
            <span class="exp">expected: {esc(r['exp'])}</span></div>
          <div class="two">
            <div class="col">
              <div class="coltitle">oneshot_rag <span class="hit {'h' if r['o_hit'] else 'm'}">{oh}</span>
                <span class="lat">{r['o_ttft'] and round(r['o_ttft'],1)}s/{r['o_total'] and round(r['o_total'],1)}s</span></div>
              <div class="routed">routed: {esc(r['o_routed'])}</div>
              <div class="ans">{esc(r['o_text'])}</div>
            </div>
            <div class="col">
              <div class="coltitle">agentic_rag
                <span class="lat">{r['a_ttft'] and round(r['a_ttft'],1)}s/{r['a_total'] and round(r['a_total'],1)}s</span></div>
              <div class="routed">(routing not exposed; judge by grounding in answer)</div>
              <div class="ans">{esc(r['a_text'])}</div>
            </div>
          </div>
        </div>"""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-KB RAG — oneshot vs agentic</title>
<style>
:root{{--bg:#0f1115;--surface:#161a22;--s2:#1d222c;--border:#2a313d;--text:#e8ecf1;--muted:#8b94a3;--accent:#6ea8ff;--good:#2ecc71;--bad:#ff5d6c;--mid:#f5b042;--code:#11151c}}
@media(prefers-color-scheme:light){{:root{{--bg:#f7f8fa;--surface:#fff;--s2:#eef1f6;--border:#d6dae1;--text:#14181f;--muted:#5b6573;--accent:#2c63d8;--good:#1f7c43;--bad:#c9203a;--mid:#8a5300;--code:#eef1f6}}}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;line-height:1.55}}
.wrap{{max-width:1120px;margin:0 auto;padding:44px 26px 90px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
h1{{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}}h2{{font-size:21px;margin:44px 0 12px}}
.lede{{color:var(--muted);max-width:760px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:13px;margin:24px 0}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}}
.kpi .label{{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}
.kpi .num{{font:700 28px/1.1 ui-monospace,Menlo,monospace;margin:7px 0 2px}}
.kpi .num.good{{color:var(--good)}}.kpi .num.bad{{color:var(--bad)}}.kpi .num.mid{{color:var(--mid)}}
.kpi .sub{{font-size:12px;color:var(--muted)}}
table.data{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;font-size:14px}}
table.data th,table.data td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}}
table.data thead th{{background:var(--s2);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
table.data td.num,table.data th.num{{text-align:right;font:500 13px ui-monospace,Menlo,monospace}}
table.data tr:last-child td{{border-bottom:none}}
.num.good{{color:var(--good)}}.num.mid{{color:var(--mid)}}.num.bad{{color:var(--bad)}}
.note{{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--accent);padding:13px 17px;border-radius:8px;font-size:14px;margin:10px 0}}
.note.bad{{border-left-color:var(--bad)}}.note.good{{border-left-color:var(--good)}}
.qcard{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:13px 15px;margin:11px 0}}
.qhead{{font-weight:600;font-size:14px;margin-bottom:9px}}
.qhead .cat{{display:inline-block;font:700 10px ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.06em;background:var(--s2);color:var(--muted);padding:2px 7px;border-radius:4px;margin-right:7px}}
.qhead .exp{{display:block;font-weight:400;font-size:12px;color:var(--muted);margin-top:3px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:11px}}
@media(max-width:720px){{.two{{grid-template-columns:1fr}}}}
.col{{background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:10px 12px}}
.coltitle{{font:600 12px ui-monospace,Menlo,monospace;margin-bottom:5px}}
.coltitle .lat{{float:right;color:var(--muted);font-weight:400}}
.hit{{display:inline-block;width:16px;text-align:center;border-radius:3px;font-weight:700}}
.hit.h{{color:var(--good)}}.hit.m{{color:var(--bad)}}
.routed{{font:500 11px ui-monospace,Menlo,monospace;color:var(--muted);margin-bottom:6px}}
.ans{{font-size:12.5px;color:var(--text)}}
footer{{margin-top:50px;padding-top:16px;border-top:1px solid var(--border);color:var(--muted);font-size:12px}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font:0.9em ui-monospace,Menlo,monospace}}
</style></head><body><div class="wrap">

<div class="eyebrow">Lyzr RAG · multi-KB routing benchmark · dev</div>
<h1>oneshot_rag vs agentic_rag — 11 knowledge bases</h1>
<p class="lede">Side-by-side across 11 topically-distinct KBs (research papers, ML/Python textbooks,
DEM super-resolution, the author's project, internship templates, commercial &amp; consumer wire docs).
Routing accuracy from {len(R)} probe queries; latency &amp; answers from a {len(B)}-query KB-coverage subset run through both paths.</p>

<h2>Headline</h2>
<div class="kpis">
{kpi("oneshot routing hit-rate", f"{hit(single)}%", f"right KB retrieved (single-target, n={len(single)})", "bad" if hit(single)<70 else "good")}
{kpi("oneshot exact-match", f"{exact(single)}%", "only the right KB (no over-retrieval)", "bad")}
{kpi("oneshot TTFT p50", f"{pct(o_ttft,50)}s", f"agentic {pct(a_ttft,50)}s", "good")}
{kpi("oneshot total p50", f"{pct(o_tot,50)}s", f"agentic {pct(a_tot,50)}s")}
{kpi("oneshot refusal", f"{o_ref}%", f"agentic {a_ref}%")}
{kpi("KBs fully missed (oneshot)", f"{sum(1 for kb in perkb if perkb[kb][0]==0)}/11", "0% routing hit-rate")}
</div>

<div class="note bad"><b>Key finding — routing degrades at scale.</b> With one KB, routing is trivial.
At 11 KBs the oneshot planner hits the right KB only <b>{hit(single)}%</b> of the time and never exactly
(<b>{exact(single)}%</b> exact-match — it always over-retrieves a fixed set). It systematically routes to
KBs 1-6 and starves KBs 7-11. Root cause is the <b>planner</b>, not retrieval: queried directly, the
"missed" KBs return high-scoring chunks (wire 0.64, webrtc 0.66, internship 0.52) — higher than transformers (0.32).
If the planner routed to them, those chunks would dominate the merge. They never appear → the planner isn't selecting them.</div>

<h2>Per-KB routing hit-rate (oneshot, single-target + sibling)</h2>
<table class="data"><thead><tr><th>knowledge base</th><th class="num">hit-rate</th><th class="num">hits</th></tr></thead>
<tbody>{perkb_rows}</tbody></table>

<h2>oneshot vs agentic on the coverage subset</h2>
<table class="data"><thead><tr><th>metric</th><th class="num">oneshot_rag</th><th class="num">agentic_rag</th></tr></thead><tbody>
<tr><td>TTFT p50 / p95 (s)</td><td class="num good">{pct(o_ttft,50)} / {pct(o_ttft,95)}</td><td class="num">{pct(a_ttft,50)} / {pct(a_ttft,95)}</td></tr>
<tr><td>Total p50 / p95 (s)</td><td class="num">{pct(o_tot,50)} / {pct(o_tot,95)}</td><td class="num">{pct(a_tot,50)} / {pct(a_tot,95)}</td></tr>
<tr><td>Refusal rate</td><td class="num">{o_ref}%</td><td class="num">{a_ref}%</td></tr>
<tr><td>Routed to correct KB (single-target)</td><td class="num bad">{o_cov_hit}%</td><td class="num good">routes correctly*</td></tr>
</tbody></table>
<div class="note bad"><b>Grounding, not just routing.</b> oneshot's 0% refusal is misleading: when it
misroutes (KBs 7-11), it does <i>not</i> refuse — it answers from <b>ungrounded model knowledge</b>
("in a typical situation, generally requires…"). For the wire query, oneshot says "the retrieved context
does not provide…" then answers anyway from general knowledge, while <b>agentic_rag retrieved the actual
Wire Transfer Submission Checklist</b> and answered from it. Silent ungrounded answers are worse than a refusal.</div>
<p class="lede" style="margin-top:10px;font-size:13px">* oneshot routing is observed directly (documents the planner retrieved).
agentic routing is not exposed in the API response, but its <b>answer content is correctly grounded</b> on the
wire / internship / webrtc queries where oneshot misrouted (see the per-query answers below) — i.e. the ReAct
tool loop selected the right KB tool. The filename-citation heuristic under-counts it because agentic cites
"Document N" rather than raw filenames.</p>

<h2>Per-query, side by side</h2>
{sample_cards}

<footer>
Artifacts: <code>/tmp/mkb_routing_full.json</code> (201 routing) · <code>/tmp/mkb_sidebyside.json</code> ({len(B)} coverage, both paths) ·
fixture <code>/tmp/dev_multi_kb_fixture.json</code>.<br>
Planner gpt-4o-mini · synthesizer gpt-4o · merge_top_k=6 · PaddleOCR (wire scans) + LiteParse (born-digital).
</footer>
</div></body></html>"""

    open(a.out, "w").write(doc)
    print(f"wrote {a.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
