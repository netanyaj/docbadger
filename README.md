# DocBadger

A GitHub Action that finds documentation made stale by a pull request and flags it on the PR, with a proposed fix.

It only proposes. Nothing is committed to your docs without a human.

## How it works

```
diff → filter meaningful changes → link changed code to doc sections → Verifier (LLM) → Corrector → Validator → PR comment
```

- **Linking** decides which doc sections a change could affect. It uses exact names first, then leaf names, then an embedding fallback.
- **Verifier** marks a section stale only if the change *contradicts* it. A doc that is merely incomplete is not stale.
- **Corrector + Validator** draft a fix and check it against the new code before it is shown.
- **Guardrails:** a hard cap on LLM calls per run (default 50), fail-open by default, and a cost estimate in every summary.

## Usage

```yaml
- uses: netanyaj/docbadger@main
  with:
    llm_api_key: ${{ secrets.OPENROUTER_API_KEY }}
    model: openai/gpt-4o        # optional
    docs_path: docs             # optional
    max_llm_calls_per_run: 50   # optional
```

See `action.yml` for all inputs and outputs.

---

## DocBadger × Potpie (branch `feat/potpie-context`)

An experiment: can a code graph make DocBadger's linking smarter? I tested it on [Potpie](https://github.com/potpie-ai/potpie), an open-source context graph for AI coding agents, and used Potpie's own repo as the test bed.

### Why

On large PRs, v1 linking finds many possible matches but few real ones. On Potpie PR [#1057](https://github.com/potpie-ai/potpie/pull/1057) it produced 96 (function, doc section) pairs against a 50-call budget, and 84 of them were loose name matches. The run hit its cap on noise. v1 also can't see two kinds of drift:

1. **Deleted code.** v1 skips deleted files, so a doc that names a removed module or span stays wrong.
2. **Indirect changes.** A doc describes function `C`, and the PR changes `F`, which `C` calls. The doc never names `F`, so v1 never checks it.

### What this branch adds

| Stage | File | LLM calls | What it does |
|---|---|---|---|
| Reference integrity | `src/reference_integrity.py` | 0 | Flags removed file paths, symbols and string literals (e.g. span names) that a doc still names. Skips SHA-pinned sections, ADRs and test files. |
| Graph ranker (linker A) | `src/graph_ranker.py` | 0 | Builds a call graph with Potpie's open-source parser (`parsing_rs`). It keeps exact links, keeps a name-only link only if the graph backs it up, and adds links to docs that describe a *caller* of the changed function. It then spends the LLM budget in score order. |
| Linking eval | `scripts/eval_linking.py` | Verifier only | Compares v1 linking with v1 + graph ranker on a real PR range. `export` needs git and `parsing_rs`. `judge` needs an API key. |

Both stages are optional. If `parsing_rs` isn't installed, the ranker falls back to v1 behavior.

**Status:** both stages run as a library and in the eval script. They are **not yet wired into the Action's pipeline** (`src/main.py`).

### Results

**Reference integrity on PR #1057:** 4 stale references (the `daemon.*` spans in `observability.md`) and 0 false positives, with zero LLM calls. The first version had 13 false positives out of 22 findings. The SHA-pinning, ADR and test-file rules brought that to 0.

**Graph ranker:**

| PR | v1 pairs | Graph pairs | LLM calls saved | Precision (v1 → graph) | Notes |
|---|---|---|---|---|---|
| [#1057](https://github.com/potpie-ai/potpie/pull/1057) (tuning PR) | 96 | 48 | 50% | 19% → 35% | Same 3 stale sections found. v1 hit the budget cap before reaching all of them. |
| [#1032](https://github.com/potpie-ai/potpie/pull/1032), [#976](https://github.com/potpie-ai/potpie/pull/976), [#985](https://github.com/potpie-ai/potpie/pull/985) (held out) | 112 | 77 | 31% | n/a (0 stale in both arms) | The graph arm alone found an undocumented `entity-label-drift` report through caller links. |

Raw verdicts are in `eval/linking/`.

### Caveats

- **Judge.** Verdicts come from Claude applying the Verifier's rules blind to which arm each pair came from, not from the production model (GPT-4o). OpenRouter was blocked from the environment where the eval ran.
- **#1057 is not held-out.** One ranker fix (using names the change added or removed) was made after seeing v1's verdicts on that PR.
- **Held-out recall is unmeasured.** None of the three held-out PRs made a doc contradict the code, so there was nothing stale to find or miss.
- **Name-based edges.** The parser's call edges are name-based. Common names like `.set` are ambiguous, so caller links only use names defined once in the repo.

### What it found in Potpie's docs

Four docs-only fixes on Potpie `main`: a wrong plugin install command, removed daemon spans still listed, an outdated `provision()` signature, and a missing quality report. Each is being raised as a PR from a fork of Potpie.

### Run it

```bash
pip install -r requirements.txt
# parsing_rs is not in the PyPI `potpie` package: build it from the Potpie repo (maturin)
python scripts/eval_linking.py export --repo ../potpie --base <sha> --head <sha> --pairs pairs.json
LLM_API_KEY=... python scripts/eval_linking.py judge --pairs pairs.json --out results.json --budget 50
pytest tests/test_reference_integrity.py tests/test_graph_ranker.py
```
