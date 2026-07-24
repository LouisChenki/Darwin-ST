# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Darwin-ST is an **autonomous spatio-temporal forecasting research system**: it self-optimizes traffic-forecasting models (PeMS04/08, METR-LA, PEMS-BAY) toward SOTA via evolutionary NAS + HPO, and — its core research bet — uses an LLM to **invent new operators by cross-domain analogy** (e.g. transplanting masked-autoencoding from CV into ST). The whole system lives in the `src/darwin_st/` package: a self-contained Python system that calls an LLM as a low-frequency Tier-2 tool (the Python harness drives; the LLM does not).

The governing design lives in `docs/` — read these before substantial work: `ANALYSIS.md` (original diagnosis), `BLUEPRINT.md` (P0–P4 roadmap), `P2_ALGORITHM_DESIGN.md` (NAS/HPO/MAP-Elites decisions), `TIER2_DESIGN_DECISIONS.md` + `TIER2_RESEARCH_FINDINGS.md` (the LLM-creation layer), `SERVER_VALIDATION.md` (what's actually been run on real hardware).

## Commands

```bash
uv sync                              # create .venv, install deps (CPU/MPS torch on macOS, cu128 on Linux)
uv run pytest tests/ -q              # full suite (~600 tests, device-agnostic, runs on CPU)
uv run pytest tests/test_metrics.py -v          # one file
uv run pytest tests/test_metrics.py::test_zeros_are_masked_out   # one test
```

There is no lint/typecheck configured. Tests are the correctness contract — every module has a `tests/test_<module>.py`. `pythonpath=["src"]` is set in `pyproject.toml`, so imports are `from darwin_st...` without installing.

## The "device-agnostic logic, verify locally" rule

The entire P0–P2 stack (metrics, protocol, adjacency, genotype, builder, evolution, archive, orchestrator) is **device-independent** — it produces identical results on CPU and GPU. Correctness is validated locally with pytest on tiny tensors; only real training and real LLM/Aider calls need the server. When adding logic, keep it CPU-testable and inject the GPU/LLM-touching parts (`eval_fn`, `LLMClient`, `code_backend`) so tests use mocks. This is why almost everything takes an injected function rather than hardcoding torch.cuda or an API client.

## Architecture: the two-tier autonomy loop

The system is a **deterministic Python harness (Tier-1, high-frequency) that delegates "creativity" to an LLM (Tier-2, low-frequency, fires only on stagnation)**. This split is the spine — encode coordination in code, never rely on an LLM to "remember not to stop."

```
src/darwin_st/
  data/       P0 — trustworthy evaluation foundation. metrics.py (masked MAE/RMSE/MAPE,
              the ONLY scorer), protocol.py (per-dataset split/scaler profiles, leak-free),
              adjacency.py (Gaussian-kernel / dcrnn-csv graphs), prepare.py (download→window→split).
  memory/     P1 — SQLite experiment store. store.py (record_trial, best_so_far, query_graveyard
              by signature = hard-block of known-failed configs, nearest_experiments) + reflect.py.
  search/     P2 inner — operators.py (SPATIAL_OPS/TEMPORAL_OPS registries),
              embeddings.py (STID-style identity embeddings — the single biggest accuracy lever),
              genotype.py (discrete mutable architecture + protected innovation zone),
              builder.py (genotype → nn.Module, [B,T,N,C] convention, node dim N never flattened),
              evolution.py (aging/regularized evolution — removes OLDEST not worst).
  optim/      P2 engine — hpo.py (Optuna TPE + SuccessiveHalvingPruner=ASHA, one study per arch),
              scheduler.py (GPUScheduler, one arch per GPU), archive.py (MAP-Elites 108-cell grid),
              orchestrator.py (the run loop: ask→evaluate→memory→archive→program-driven SOTA stop),
              train.py (real eval_fn: builder+hpo+training+masked eval; make_eval_fn(dataset)).
  knowledge/  P3 — cross-domain mechanism KG. ontology.py (Mechanism = abstract_function +
              preconditions + causal_behavior + origin_domain), seeds.py (16 seed cards),
              graph_store.py (InMemory | Neo4j backends), embedding.py (Hash | SentenceTransformer),
              retrieval.py (find_cross_domain_analogy: MAC vector recall + FAC precondition-graph
              + complementary set-cover; embedding_text uses ONLY abstract_function+preconditions).
  creation/   Tier-2 — fusion.py (ZeroInitResidualFusion: y=base+α·branch, α=0, the strongest
              anti-degeneration guard), validation.py (operator gates: shape/grad/NaN/non-trivial,
              the reward-hacking defense), synthesizer.py (plan-then-code, synthesize_many = N
              hypotheses at once), aider_backend.py (writes operators via Aider in a git sandbox),
              registry.py (injects synthesized ops into SPATIAL_OPS — search space "grows"),
              creation_loop.py (diagnose bottleneck → retrieve → synthesize_many → inject →
              seed genotypes for evolution).
```

**The full creation closed loop** (validated on server, see SERVER_VALIDATION.md §P2.5-g): orchestrator stagnates → `CreationLoop.maybe_create` diagnoses a bottleneck → `find_cross_domain_analogy` returns a *complementary set* of cross-domain mechanisms → `synthesize_many` generates N fusion hypotheses, Aider writes each in a sandbox, the validation harness gates them → surviving operators are injected into `SPATIAL_OPS` and seed genotypes enter evolution → **real masked-MAE training is the final arbiter** (most fusions die, good ones survive). Synthesized operators register as `synth_`-prefixed spatial ops.

## Non-obvious invariants (breaking these silently corrupts results)

- **Masked metrics only.** `data/metrics.py` is the single source of truth; predictions must be inverse-transformed to real scale before scoring. Never compare against unmasked MAE or normalized-scale numbers.
- **Per-dataset protocol profiles.** PeMS04/08 use 6/2/2 split + 12-step-average MAE; METR-LA/PEMS-BAY use 7/1/2 + per-horizon. `protocol.py` dispatches; hardcoding one corrupts the others. Scaler fits on **train only**.
- **Node dimension N is sacred.** All tensors are `[B,T,N,C]`; operators must never flatten/mean away N (see `docs/P2_ALGORITHM_DESIGN.md` 架构铁律). The validation harness enforces this on synthesized ops.
- **Aging evolution removes the OLDEST member, not the worst** — this is the noise-regularization core, not a bug.
- **Baselines are corrected literature numbers** in `baseline_registry.py` (PeMS04 SOTA ≈17.8). The program-driven stop compares `best_so_far` against these — don't "fix" them to old wrong values.
- **`darwin-st/scripts/*.py` are thin re-export shells** over `src/darwin_st/`. Edit the `src/` source, not the shells.

## Server / real-run facts (China-network AutoDL box, 4×RTX5090)

Memory files track the live connection; the recurring gotchas:
- Background commands must use `nohup bash -lc "..."` — without `-l`, conda's PATH isn't loaded and `python` is not found. Long ops (training, git fetch, aider) must be detached + polled, never awaited in one SSH call.
- GitHub is rate-limited: set `GITHUB_MIRROR=https://gh-proxy.com/` (downloads) and `git remote set-url origin https://gh-proxy.com/https://github.com/...` (fetch).
- Cache must point at the data disk: `DARWIN_ST_CACHE=/root/autodl-tmp/...` (system disk is only 30GB).
- DeepSeek/HF: model is `deepseek-v4-pro` (or `-flash`); set `HF_ENDPOINT=https://hf-mirror.com` for model downloads. API keys live in a server file outside the repo, never committed.
- Pipe-buffering hides progress: run entry scripts with `python -u`.

## Git

Work happens on `feat/p0-evaluation-foundation` (not the default `prediction` branch). Commit per milestone; **push and merging to a main branch are "outbound" actions — confirm with the user first.** Commit messages end with a `Generated with dmxapi` line. Before each commit, list the staged files and dry-check for secrets/`.venv`/`__pycache__`/`.DS_Store`.

## Entry-point scripts (`scripts/`)

`run_autoresearch.py` (P2 optimization only), `run_creation_loop.py` (full Tier-2 creation loop), `synthesize_operator.py` (single cross-domain synthesis), `build_knowledge_graph.py` (ingest seeds → Neo4j + retrieval check), `baseline_smoke.py` (one-genotype training smoke test). All are env-var configured.
