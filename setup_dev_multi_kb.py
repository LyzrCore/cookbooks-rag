"""
Multi-KB dev fixture builder for the oneshot_rag vs agentic_rag routing test.

Creates 11 topically-distinct knowledge bases from the user's local corpus,
ingests each via /v3/assets/upload (PaddleOCR / "advanced" tier), then creates
two persistent agents — one agentic_rag, one oneshot_rag — BOTH wired to all
11 KBs. This is the test the single-KB run could not do: it exercises the
oneshot planner's *routing* decision (which KB(s) to query) across many KBs,
several of which are deliberately close siblings.

KB design (close-sibling tensions in brackets):
   1  kb_transformers_foundation_models   [~2 llm_evaluation]
   2  kb_llm_evaluation                    [~1]
   3  kb_dem_superresolution               [~4 authors_dem_project]
   4  kb_authors_dem_project               [~3]
   5  kb_ml_textbooks                      [~6 nlp, ~7 python]
   6  kb_nlp                               [~5]
   7  kb_python_programming                [~5]
   8  kb_webrtc_networking                 (distinct)
   9  kb_internship_report_templates       (distinct)
  10  kb_wire_commercial                   [~11 wire_consumer]
  11  kb_wire_consumer                     [~10]

Usage:
    export LYZR_API_KEY="dev-key"
    python setup_dev_multi_kb.py                     # full build (long; PaddleOCR)
    python setup_dev_multi_kb.py --dry-run           # print KB plan + file lists
    python setup_dev_multi_kb.py --only-kbs 1,10,11  # build a subset (1-indexed)
    python setup_dev_multi_kb.py --skip-ingest       # (re)create agents from existing KBs in fixture

Outputs /tmp/dev_multi_kb_fixture.json:
    {"env": "dev", "kbs": [{"idx","name","kb_id","description","files_ok","files_fail"}...],
     "agentic_agent_id": "...", "oneshot_agent_id": "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from lyzr import Studio

ASSETS_UPLOAD_PATH = "/v3/assets/upload"
PARSE_STATUS_PATH = "/v3/assets/{asset_id}/parse-status"

# Big born-digital textbooks need a long PaddleOCR budget (per-page OCR is slow).
POLL_TIMEOUT_S = 1800.0
POLL_INTERVAL_S = 5.0
PARALLEL_UPLOADS = 4

P = "/Users/parshva"  # corpus root prefix


# ---------------------------------------------------------------------------
# Shared agent persona — identical on both agents; only feature config differs.
# ---------------------------------------------------------------------------

AGENT_ROLE = "Technical & Operations Knowledge Assistant"
AGENT_GOAL = (
    "Answer questions across a multi-domain knowledge library spanning machine-"
    "learning research papers, programming and ML textbooks, digital-elevation-"
    "model super-resolution research, the author's academic project, college "
    "internship report templates, and commercial/consumer bank wire-transfer "
    "documents. Route each question to the most relevant knowledge base(s)."
)
AGENT_INSTRUCTIONS = (
    "Ground every factual claim in the retrieved knowledge base content and cite "
    "the source. Pick the knowledge base(s) most relevant to the question. If the "
    "libraries do not cover a question, say what is missing rather than refusing "
    "flatly. Do not invent facts, figures, dates, or numbers."
)
AGENT_PROVIDER = "gpt-4o"
AGENT_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# 11 KB specifications.
# ---------------------------------------------------------------------------

@dataclass
class KBSpec:
    idx: int
    name: str
    description: str
    examples: list[str]
    files: list[str]
    filter_fields: list[str] = field(default_factory=lambda: ["source"])


KB_SPECS: list[KBSpec] = [
    KBSpec(
        idx=1,
        name="kb_transformers_foundation_models",
        description=(
            "Specific deep-learning research PAPERS on three named architectures: "
            "the Transformer / 'Attention Is All You Need', Kolmogorov-Arnold "
            "Networks (KAN), and NVIDIA's Nemotron-4 340B report. Use ONLY for "
            "questions about these specific papers/architectures — self-attention, "
            "positional encoding, KAN, Nemotron specs. "
            "NOT for general machine-learning how-to or training (use the ML "
            "textbooks KB), NOT for NLP techniques (use the NLP KB), NOT for "
            "Python coding (use the Python KB)."
        ),
        examples=[
            "how does the self-attention mechanism work in transformers",
            "what is multi-head attention",
            "what are Kolmogorov-Arnold Networks",
            "Nemotron-4 340B model architecture and training data",
        ],
        files=[
            f"{P}/Papers/attention_is_all_you_need.pdf",
            f"{P}/Papers/KAN.pdf",
            f"{P}/Papers/Nemotron_4_340B_8T.pdf",
        ],
    ),
    KBSpec(
        idx=2,
        name="kb_llm_evaluation",
        description=(
            "LLM evaluation and hallucination-detection research. Contains the "
            "Lynx open-source hallucination evaluation model paper. Use for "
            "questions about evaluating LLM outputs, detecting hallucinations, "
            "and faithfulness/groundedness scoring."
        ),
        examples=[
            "how does the Lynx hallucination evaluation model work",
            "how to detect hallucinations in LLM outputs",
            "what benchmark does Lynx use for faithfulness",
        ],
        files=[
            f"{P}/Papers/Lynx- Open Source Hallucination Evaluatio Model.pdf",
        ],
    ),
    KBSpec(
        idx=3,
        name="kb_dem_superresolution",
        description=(
            "Digital Elevation Model (DEM) super-resolution research papers: GAN-"
            "based DEM super-resolution (D-SRCAGAN, SRGAN for DEM), lunar DEM super-"
            "resolution via sparse representation and local implicit functions, and "
            "DEM enhancement using deep learning. Use for questions about DEM super-"
            "resolution methods, GANs for elevation data, and lunar terrain reconstruction."
        ),
        examples=[
            "how does D-SRCAGAN super-resolve digital elevation models",
            "GAN approaches for lunar DEM super-resolution",
            "sparse representation for lunar DEM reconstruction",
            "deep learning DEM enhancement methods",
        ],
        files=[
            f"{P}/ISRO/Papers/D-SRCAGAN__DEM_Super-resolution_Generative_Adversarial_Network.pdf",
            f"{P}/ISRO/Papers/dem_srgan_eartharxiv.pdf",
            f"{P}/ISRO/Papers/GAN-for-Pixel-Scale-Lunar-DEM.pdf",
            f"{P}/ISRO/Papers/ijgi-10-00501-with-cover.pdf",
            f"{P}/ISRO/Papers/Lunar_DEM_Super-resolution_reconstruction_via_sparse_representation.pdf",
            f"{P}/ISRO/Papers/Super-Resolution_of_Digital_Elevation_Model_with_Local_Implicit_Function_Representation.pdf",
            f"{P}/ISRO/Papers/Super-resolution_reconstruction_of_a_digital_eleva.pdf",
            f"{P}/Papers/2101.04812_DEM ENHANCEMENTUSING_DEEPLEARNING.pdf",
        ],
    ),
    KBSpec(
        idx=4,
        name="kb_authors_dem_project",
        description=(
            "The author's (Parshva) own academic / ISRO project work on enhancing "
            "Digital Elevation Models from high-resolution imagery, plus the project "
            "completion certificate and the final 8th-semester project report. Use "
            "for questions specifically about the author's DEM enhancement project, "
            "their methodology, results, and certification."
        ),
        examples=[
            "what was Parshva's DEM enhancement project about",
            "the author's methodology for enhancing digital elevation models",
            "results of the 8th semester DEM project",
        ],
        files=[
            f"{P}/ISRO/Parshva_Enhancement of Digital  Elevation Model from High.pdf",
            f"{P}/ISRO/Parshva Certificate837.pdf",
            f"{P}/College/Report Final 8th sem.pdf",
        ],
    ),
    KBSpec(
        idx=5,
        name="kb_ml_textbooks",
        description=(
            "General machine-learning and deep-learning TEXTBOOK how-to "
            "(Hands-On Machine Learning, Deep Learning with Python, Designing ML "
            "Systems). Use for broad ML/DL concepts and practice: model training, "
            "evaluation, overfitting, bias-variance, regularization, batch "
            "normalization, transfer learning, CNNs, feature engineering, and "
            "production ML system design. This is the default for any general "
            "'how does ML/deep learning work' question. "
            "NOT for specific architecture papers like the Transformer/KAN/Nemotron "
            "(use the transformers/foundation-models KB), NOT for NLP-specific "
            "techniques (use the NLP KB), NOT for Python language questions."
        ),
        examples=[
            "how to design a production machine learning system",
            "what is a learning curve in machine learning",
            "deep learning training best practices",
            "feature engineering for ML models",
        ],
        files=[
            f"{P}/Books/Hands On Machine Learning 3rd edition.pdf",
            f"{P}/Books/dokumen.pub_designing-machine-learning-systems-an-iterative-process-for-production-ready-applications-1nbsped-1098107969-9781098107963.pdf",
            f"{P}/ISRO/Books/Deep Learning with Python.pdf",
        ],
    ),
    KBSpec(
        idx=6,
        name="kb_nlp",
        description=(
            "A dedicated Natural Language Processing textbook. Use ONLY for "
            "NLP-specific techniques applied to TEXT: tokenization, stemming/"
            "lemmatization, part-of-speech tagging, named-entity recognition, word "
            "embeddings, TF-IDF, bag-of-words, parsing, sentiment analysis, and "
            "text classification. "
            "NOT for general machine learning (use the ML textbooks KB), NOT for "
            "transformer/attention papers (use the transformers KB), NOT for "
            "Python programming (use the Python KB)."
        ),
        examples=[
            "natural language processing tokenization techniques",
            "how does text classification work",
            "NLP language modeling fundamentals",
        ],
        files=[
            f"{P}/Books/natural language processing.pdf",
        ],
    ),
    KBSpec(
        idx=7,
        name="kb_python_programming",
        description=(
            "Python PROGRAMMING books (Learning Python, Web Scraping with Python, "
            "Python for Data Science, Python for Gemini/Bard). Use ONLY for writing "
            "Python CODE: language syntax, data types, list comprehensions, "
            "decorators, generators, exception handling, virtual environments, web "
            "scraping with BeautifulSoup/requests, and calling LLM APIs from Python. "
            "NOT for machine-learning theory/concepts (use the ML textbooks KB), "
            "NOT for NLP techniques (use the NLP KB), NOT for model-architecture "
            "papers (use the transformers KB)."
        ),
        examples=[
            "how to scrape a website with Python",
            "Python list comprehension syntax",
            "using Python with the Gemini API",
            "Python data science workflow",
        ],
        files=[
            f"{P}/Books/Learning Python, 5th Edition.pdf",
            f"{P}/Books/Web Scraping with Python, 2nd Edition.pdf",
            f"{P}/Books/Python for Data Science Certificate.pdf",
            f"{P}/Books/Python for Gemini and Bard.pdf",
        ],
    ),
    KBSpec(
        idx=8,
        name="kb_webrtc_networking",
        description=(
            "'WebRTC for the Curious' — real-time peer-to-peer communication over "
            "the web: signaling, STUN/TURN, ICE, SRTP, data channels. Use for "
            "questions about WebRTC, real-time audio/video, NAT traversal, and "
            "peer connections."
        ),
        examples=[
            "how does WebRTC establish a peer connection",
            "what is ICE and STUN in WebRTC",
            "WebRTC data channels",
        ],
        files=[
            f"{P}/Books/webrtc-for-the-curious.pdf",
        ],
    ),
    KBSpec(
        idx=9,
        name="kb_internship_report_templates",
        description=(
            "College 8th-semester internship report templates and guidelines: the "
            "official report-format guidelines, demo internship reports, title-page "
            "/ certificate / acknowledgement templates, the all-chapters template, "
            "and student daily/weekly diary log formats. Use for questions about how "
            "to structure or format a college internship report and what sections it requires."
        ),
        examples=[
            "what sections must a college internship report include",
            "internship report formatting guidelines",
            "how to fill the student daily diary log",
            "internship report title page format",
        ],
        files=[
            f"{P}/College/Sem-8/reportformatforaninternship/3180701_Internship_Report_Guidelines.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Demo_Internship_Report_1.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Demo_Internship_Report_2.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Feedback from industry expert.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Part_1_TitlePage.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Part_2_Certificate_and_Completion_Certificate.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Part_3_Acknowledgement_Abstract_and_Table_of_Contents.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Part_4_All_chapters.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/STUDENT’S DAILY DIARY LOG.pdf",
            f"{P}/College/Sem-8/reportformatforaninternship/Weekly log.pdf",
        ],
    ),
    KBSpec(
        idx=10,
        name="kb_wire_commercial",
        description=(
            "Commercial bank wire-transfer documents (scanned): wire transfer "
            "submission checklists, manual wire transfer agreements, and wire "
            "requests for commercial customers including Bravo Enterprises Inc and "
            "Bath Planet of Chicago Inc. Use for questions about COMMERCIAL wire "
            "transfer procedures, approvals, agreements, and these specific "
            "commercial customers — NOT consumer wires."
        ),
        examples=[
            "documents required for a commercial wire transfer",
            "Bravo Enterprises wire transfer agreement",
            "Bath Planet of Chicago wire request",
            "commercial wire transfer approval workflow",
        ],
        files=[
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 1 FW Wire Approval Midwest Imaging.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 1 scan_kjain_2026-01-20-10-25-08.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 1 Wire-Checklist.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 2 BravoEnterprisesInc01122026pdf.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 2 BravoEnterprisesIncwireagreementpdf.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 2 FW Bravo enterprises wire transfer .pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 2 scan_mgamez-jimenez_2026-01-12-11-43-24.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 3 Bath Planet Of Chicago Inc 01212026.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 3 One more wire out request.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 3 Wire Agreement.pdf",
            f"{P}/lyzr/ocr-testing/Files/Commercial Sample 3 Wire Request.pdf",
        ],
    ),
    KBSpec(
        idx=11,
        name="kb_wire_consumer",
        description=(
            "Consumer bank wire-transfer documents (scanned): wire approval and "
            "request documents for an individual consumer customer (Vickie Ann "
            "Rossi). Use for questions about CONSUMER wire transfers and this "
            "specific individual customer — NOT commercial wires."
        ),
        examples=[
            "consumer wire transfer approval documents",
            "Vickie Ann Rossi wire transfer",
            "individual customer wire request",
        ],
        files=[
            f"{P}/lyzr/ocr-testing/Files/Consumer Sample 1 FW Wire - approval needed.pdf",
            f"{P}/lyzr/ocr-testing/Files/Consumer Sample 1 scan_agora_2026-01-20-09-30-22.pdf",
            f"{P}/lyzr/ocr-testing/Files/Consumer Sample 1 VICKIE ANN ROSSI 01202026.pdf",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Async upload pool
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    path: str
    asset_id: Optional[str]
    status: Optional[str]
    duration_s: float
    error: Optional[str] = None


async def _upload_one(client, base_url, api_key, kb_id, path, provider) -> UploadResult:
    t0 = time.perf_counter()
    headers = {"x-api-key": api_key}
    parse_config = {"provider": provider, "rag_id": kb_id, "extract_text": True, "label_pages": False}
    try:
        with open(path, "rb") as fh:
            files = {"files": (os.path.basename(path), fh.read(), "application/pdf")}
        data = {"parse_config": json.dumps(parse_config)}
        r = await client.post(f"{base_url}{ASSETS_UPLOAD_PATH}", headers=headers, files=files, data=data)
        r.raise_for_status()
        payload = r.json()
        asset_id = next((res.get("asset_id") for res in (payload.get("results") or []) if res.get("success")), None)
        if not asset_id:
            return UploadResult(path, None, None, time.perf_counter() - t0, error=f"no asset_id: {payload}")
        deadline = time.time() + POLL_TIMEOUT_S
        last = {}
        while time.time() < deadline:
            sr = await client.get(f"{base_url}{PARSE_STATUS_PATH.format(asset_id=asset_id)}", headers=headers)
            if sr.status_code == 404:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue
            sr.raise_for_status()
            last = sr.json()
            st = (last.get("parsing_status") or "").lower()
            if st in {"success", "completed", "done"}:
                return UploadResult(path, asset_id, st, time.perf_counter() - t0)
            if st in {"failed", "error"}:
                return UploadResult(path, asset_id, st, time.perf_counter() - t0, error=f"parse failed: {last}")
            await asyncio.sleep(POLL_INTERVAL_S)
        return UploadResult(path, asset_id, "timeout", time.perf_counter() - t0, error=f"timeout; last={last}")
    except Exception as exc:
        return UploadResult(path, None, None, time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")


async def ingest_kb(base_url, api_key, kb_id, files, provider, parallelism) -> list[UploadResult]:
    # Create the Semaphore INSIDE the coroutine so it binds to the event loop
    # that asyncio.run() creates for THIS call. Creating it once in main() and
    # reusing across multiple asyncio.run() calls binds it to a stale, closed
    # loop and raises "bound to a different event loop".
    sem = asyncio.Semaphore(parallelism)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)) as client:
        async def bound(p):
            async with sem:
                r = await _upload_one(client, base_url, api_key, kb_id, p, provider)
                tag = "ok  " if r.error is None else "FAIL"
                print(f"      [{tag}] {os.path.basename(p)[:60]:<60} ({r.duration_s:5.1f}s, {r.status})"
                      + (f"  {r.error[:80]}" if r.error else ""), flush=True)
                return r
        return await asyncio.gather(*[bound(p) for p in files])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=["dev", "prod", "local"], default="dev")
    ap.add_argument("--parser", choices=["advanced", "standard"], default="advanced",
                    help="advanced=PaddleOCR (default), standard=LiteParse")
    ap.add_argument("--out", default="/tmp/dev_multi_kb_fixture.json")
    ap.add_argument("--only-kbs", default=None, help="Comma-separated 1-indexed KB idxs to build (subset).")
    ap.add_argument("--parallel-uploads", type=int, default=PARALLEL_UPLOADS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("LYZR_API_KEY")
    if not api_key:
        print("Error: LYZR_API_KEY env var is required.", file=sys.stderr)
        return 2

    only = None
    if args.only_kbs:
        only = {int(x) for x in args.only_kbs.split(",")}
    specs = [s for s in KB_SPECS if (only is None or s.idx in only)]

    # Validate files exist up front
    print(f"[plan] {len(specs)} KB(s), parser={args.parser}:")
    missing = []
    for s in specs:
        present = [f for f in s.files if os.path.exists(f)]
        absent = [f for f in s.files if not os.path.exists(f)]
        missing.extend(absent)
        print(f"  {s.idx:>2}. {s.name:<34} {len(present)}/{len(s.files)} files present")
        for f in absent:
            print(f"        MISSING: {f}")
    if missing:
        print(f"\n[plan] {len(missing)} file(s) missing — they will be skipped.")
    if args.dry_run:
        print("\n[dry-run] no KBs/agents created.")
        return 0

    studio = Studio(api_key=api_key, env=args.env)
    base_url = studio._http.base_url  # type: ignore[attr-defined]
    print(f"[setup] env={args.env} base_url={base_url}")

    kb_records = []
    t_all = time.perf_counter()

    for s in specs:
        present = [f for f in s.files if os.path.exists(f)]
        if not present:
            print(f"[kb {s.idx}] {s.name}: no files present, skipping KB creation.")
            continue
        print(f"\n[kb {s.idx}] Creating '{s.name}' ({len(present)} files)...", flush=True)
        kb = studio.create_knowledge_base(
            name=f"{s.name}_{int(time.time())}",
            vector_store="qdrant",
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o",
            description=s.description,
        )
        print(f"[kb {s.idx}]   kb_id={kb.id}  ingesting...", flush=True)
        results = asyncio.run(ingest_kb(base_url, api_key, kb.id, present, args.parser, args.parallel_uploads))
        ok = [r for r in results if r.error is None]
        fail = [r for r in results if r.error is not None]
        print(f"[kb {s.idx}]   {len(ok)}/{len(results)} ingested ok", flush=True)
        kb_records.append({
            "idx": s.idx, "name": s.name, "kb_id": kb.id,
            "description": s.description, "examples": s.examples,
            "filter_fields": s.filter_fields,
            "files_ok": [os.path.basename(r.path) for r in ok],
            "files_fail": [{"f": os.path.basename(r.path), "err": (r.error or "")[:120]} for r in fail],
            "_kb_obj": kb,  # transient, stripped before json dump
        })

    if not kb_records:
        print("[setup] No KBs created — aborting agent creation.", file=sys.stderr)
        return 1

    # Build feature configs across all created KBs.
    agentic_cfgs, oneshot_cfgs = [], []
    for rec in kb_records:
        kb = rec["_kb_obj"]
        agentic_cfgs.append(kb.to_agentic_config(top_k=10, retrieval_type="basic"))
        oneshot_cfgs.append(kb.to_oneshot_config(
            top_k=20, retrieval_type="basic",
            filter_fields=rec["filter_fields"], examples=rec["examples"],
        ))

    common = dict(provider=AGENT_PROVIDER, role=AGENT_ROLE, goal=AGENT_GOAL,
                  instructions=AGENT_INSTRUCTIONS, temperature=AGENT_TEMPERATURE)
    ts = int(time.time())

    print(f"\n[setup] Creating agentic_rag agent with {len(agentic_cfgs)} KBs...", flush=True)
    agent_a = studio.create_agent(
        name=f"multikb_agentic_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "agentic_rag": agentic_cfgs}, "priority": 0}],
        **common,
    )
    print(f"[setup]   agentic_agent_id={agent_a.id}", flush=True)

    print(f"[setup] Creating oneshot_rag agent with {len(oneshot_cfgs)} KBs...", flush=True)
    agent_b = studio.create_agent(
        name=f"multikb_oneshot_rag_{ts}",
        features=[{"type": "KNOWLEDGE_BASE",
                   "config": {"lyzr_rag": {}, "oneshot_rag": oneshot_cfgs,
                              "planner_model": "gpt-4o-mini", "merge_top_k": 6},
                   "priority": 0}],
        **common,
    )
    print(f"[setup]   oneshot_agent_id={agent_b.id}", flush=True)

    fixture = {
        "env": args.env,
        "parser": args.parser,
        "ts": ts,
        "agentic_agent_id": agent_a.id,
        "oneshot_agent_id": agent_b.id,
        "kbs": [{k: v for k, v in rec.items() if k != "_kb_obj"} for rec in kb_records],
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, default=str)

    total_ok = sum(len(r["files_ok"]) for r in kb_records)
    total_fail = sum(len(r["files_fail"]) for r in kb_records)
    print(f"\n[setup] DONE in {time.perf_counter()-t_all:.0f}s — {len(kb_records)} KBs, "
          f"{total_ok} files ingested, {total_fail} failed.")
    print(f"[setup] Fixture: {args.out}")
    print(f"[setup] Next: python benchmark_multi_kb.py --fixture {args.out} --probe-once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
