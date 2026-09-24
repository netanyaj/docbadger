#!/usr/bin/env python3
"""
Linking-stage eval: v1 linker vs v1 + graph ranker (linker A) on a real PR.

The existing harness (run_eval.py) scores the Verifier/Corrector/Validator
on hand-built (old_code, new_code, doc_section) cases. It never exercises
*linking*: which doc sections get sent to the Verifier in the first place.
This script does, on a real repository and PR range:

  1. Runs DocBadger's deterministic stages (diff -> filter -> heuristic links).
  2. Builds v1's candidate list and the graph-ranked list.
  3. Runs the REAL Verifier once on the union of both lists (cached per
     pair), so both arms are judged by the same verdicts.
  4. Reports, per arm: LLM calls needed, stale verdicts surfaced, and what
     each arm would surface under the production call budget (default 50).

The Verifier is the judge here, which is a proxy: a "stale" verdict is not
ground truth. Stale verdicts are written out for hand review.

Two phases, so the graph work and the LLM work can run on different machines:

    # 1. Build both arms (needs git + `parsing_rs`, no LLM, no network):
    python scripts/eval_linking.py export --repo /path/to/repo \
        --base <sha> --head <sha> --pairs pairs.json

    # 2. Judge with the real Verifier (needs only requirements.txt + a key):
    LLM_API_KEY=... python scripts/eval_linking.py judge --pairs pairs.json \
        --out results.json [--budget 50]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_analyzer import get_modified_functions  # noqa: E402  (stdlib + git only)
from change_filter import filter_meaningful  # noqa: E402
from code_parser import get_all_code_chunks  # noqa: E402
from doc_parser import get_all_doc_sections  # noqa: E402
from heuristic_linker import build_heuristic_links_with_source  # noqa: E402
from graph_ranker import build_call_graph, rank_candidates  # noqa: E402
from verifier import judge_staleness  # noqa: E402


def export(args):
    os.chdir(args.repo)
    functions = filter_meaningful(get_modified_functions(args.base, args.head))
    sections = get_all_doc_sections(args.docs_root)
    links = build_heuristic_links_with_source(get_all_code_chunks("."), sections)
    v1 = [[f.qualified_id, sid, src] for f in functions
          for sid, src in sorted(links.get(f.qualified_id, {}).items())]
    ranked = rank_candidates([f.qualified_id for f in functions], links, sections,
                             build_call_graph("."))
    graph_arm = [[p.chunk_id, p.section_id, p.source] for p in ranked]
    used = {c for c, _, _ in v1 + graph_arm}
    used_sections = {s for _, s, _ in v1 + graph_arm}
    payload = {
        "repo": args.repo, "base": args.base, "head": args.head,
        "functions": {f.qualified_id: {"old_code": f.old_code, "new_code": f.new_code}
                      for f in functions if f.qualified_id in used},
        "sections": {sid: sections[sid].text for sid in used_sections},
        "arms": {"v1": v1, "graph": graph_arm},
    }
    with open(args.pairs, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"v1 pairs={len(v1)}  graph pairs={len(graph_arm)}  "
          f"union={len({(c, s) for c, s, _ in v1 + graph_arm})}  -> {args.pairs}")


def judge(args):
    data = json.load(open(args.pairs))
    fns, secs, arms = data["functions"], data["sections"], data["arms"]
    union = sorted({(c, s) for arm in arms.values() for c, s, _ in arm})

    def one(pair):
        cid, sid = pair
        v = judge_staleness(fns[cid]["old_code"], fns[cid]["new_code"], secs[sid], args.model)
        u = v.get("usage")
        return pair, {"stale": v["stale"], "diagnosis": v["diagnosis"],
                      "tokens": (getattr(u, "prompt_tokens", 0) or 0) + (getattr(u, "completion_tokens", 0) or 0)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = dict(pool.map(one, union))

    def summarize(arm):
        judged = [(c, s, src, verdicts[(c, s)]) for c, s, src in arm]
        in_budget = judged[: args.budget]
        return {
            "pairs": len(arm),
            "stale_total": sum(1 for j in judged if j[3]["stale"]),
            "stale_within_budget": sum(1 for j in in_budget if j[3]["stale"]),
            "truncated": len(arm) > args.budget,
            "tokens_total": sum(j[3]["tokens"] for j in judged),
            "stale_pairs": [{"chunk": c, "section": s, "link": src, "diagnosis": v["diagnosis"]}
                            for c, s, src, v in judged if v["stale"]],
        }

    report = {k: data[k] for k in ("repo", "base", "head")}
    report.update({"budget": args.budget, "model": args.model,
                   "errors": sum(1 for v in verdicts.values() if v["stale"] is None)})
    report.update({name: summarize(arm) for name, arm in arms.items()})
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for name in arms:
        r = report[name]
        print(f"{name:5}: pairs={r['pairs']:3}  stale={r['stale_total']:2}  "
              f"stale within budget {args.budget}={r['stale_within_budget']:2}  tokens={r['tokens_total']}")
    print(f"verifier errors: {report['errors']}  ->  {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--repo", required=True)
    e.add_argument("--base", required=True)
    e.add_argument("--head", required=True)
    e.add_argument("--pairs", required=True)
    e.add_argument("--docs-root", default=".")
    j = sub.add_parser("judge")
    j.add_argument("--pairs", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--budget", type=int, default=50)
    j.add_argument("--model", default=os.environ.get("LLM_MODEL", "openai/gpt-4o"))
    j.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    export(args) if args.cmd == "export" else judge(args)


if __name__ == "__main__":
    main()
