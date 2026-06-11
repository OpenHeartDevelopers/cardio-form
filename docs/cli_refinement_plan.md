# CLI Refinement Plan

**Status:** proposed (awaiting execution)
**Owner:** Jose Alonso Solis-Lemus
**Created:** 2026-06-11
**Related:** [`docs/python-3_11-upgrade-plan.md`](./python-3_11-upgrade-plan.md) — the
Phase 1 CLI integration tests are a prerequisite for the Click migration below.

## 1. Motivation

The `cardio_form/cli` package has accumulated structural debt. Three asks
motivate this plan:

1. Consolidate the per-command orchestrators (proposed `process_handler.py`).
2. Make heavy imports lazy so `cardioform --help` does not load PyTorch.
3. Migrate from `argparse` to `click`.

This document records the audit, the decision on each ask, and a phased plan.

## 2. Audit — current state

`entrypoint.py` builds one `argparse` parser with subparsers and a `run(args)`
switchboard that dispatches to `run_*_job(...)` functions in 6 sibling files
(`segment.py`, `reconstruct_3d.py`, `reconstruct_la.py`, `full_pipeline.py`,
`merge_labels.py`, `filter_labels.py`, `relabel.py`). Each sibling contains
**three** things:

1. `run_*_job(...)` — the thin orchestrator (worth keeping).
2. `main(args)` — an arg→kwarg wrapper that **`entrypoint` never calls** (it
   calls `run_*_job` directly) → effectively dead code.
3. an `if __name__ == "__main__":` block with its **own full `argparse`**.

### 2.1 The real problem is duplication, not cross-imports

The siblings do **not** import each other; the only "cross import" is
`entrypoint` importing all of them, which is normal for a dispatcher. What rots
is that argument definitions and the arg→kwarg mapping exist in **three places**:
`entrypoint.run()`, each module's `main()`, and each module's `__main__` parser.

This duplication has already **drifted**, which proves the cost is real:

| Command | `entrypoint` parser | Module `__main__` parser |
| --- | --- | --- |
| `segment` | `--device` default `cpu` | `--device` default `auto` |
| `reconstruct_la` | no `--model-version` | has `--model-version` |

So `python -m cardio_form.cli.segment` behaves differently from
`cardioform segment`. That is a live inconsistency, not a hypothetical.

### 2.2 The `--help` loads torch chain

`cardioform --help` currently imports all of PyTorch. The chain:

```
entrypoint
  → imports each cli submodule
    → `from cardio_form.pipeline import CardioForm`
      → pipeline.py does `import torch` at module top
```

The sneakiest offender: `entrypoint.py:3` does
`from cardio_form.pipeline import CHOICES_VIEW_TYPE` — importing a 3-element
string list that merely *lives in* the torch-importing module. That single line
forces torch on every invocation, including `--help`.

The label handlers (`filter` / `merge` / `relabel`) are already torch-free
(`geometry` pulls scipy / SimpleITK / nibabel, not torch).

## 3. Decisions on the three asks

### Ask 1 — consolidate into a handler module: ACCEPTED (refined)

Seven commands is well past the Rule of Three, so a shared module is justified.
Consolidating the `run_*_job` orchestrators lets us delete every `main()` and
`__main__` block, removing the drift in §2.1.

**Refinement:** the *actual logic* lives in `pipeline.py` / `geometry.py`; what
moves is only orchestration glue (load → call → save, logging, error→exit).
Name the module `handlers.py` to reflect that it is the orchestration seam, not
where logic lives. This ask only pays off if combined with Ask 2 — otherwise
top-level torch imports in the merged module reintroduce the eager-load problem.

### Ask 2 — lazy imports: ACCEPTED (strongly)

Done together with Ask 1. See §4 Phase A for placement.

### Ask 3 — migrate to Click: ACCEPTED, but DEFERRED

Click is nicer (decorators, native sub-groups, no manual subparser wiring) and
is a pure-Python dependency that fits the "pyproject owns conda-independent
deps" model. Two reasons to defer it rather than bundle it now:

- **argparse is not the disease; duplication is.** Asks 1 + 2 fix the structure
  and the torch-on-help problem without touching the framework. Switching
  frameworks first leaves the handler consolidation still to do.
- **Sequencing.** A Click rewrite changes the entire user-facing CLI surface —
  exactly what the upgrade plan's Phase 1 integration tests are meant to pin
  down *before* changes. Rewriting the surface with no tests changes the
  contract and the safety net simultaneously. Land the Phase 1 CLI tests first;
  then Click becomes a low-risk, test-backed refactor.

## 4. Phased plan

### Phase A — Consolidate handlers + lazy imports (do now)

Asks 1 and 2 are inseparable; do them as one change. Keep `argparse`.

1. **Create `cardio_form/cli/handlers.py`** holding the 7 consolidated
   `run_*_job` orchestrators (moved verbatim from the siblings; behaviour
   unchanged).
2. **Lazy heavy imports:** inside each torch-backed handler
   (`segment`, `reconstruct`, `reconstruct_la`, `full_pipeline`), import
   `torch` / `from cardio_form.pipeline import CardioForm` **function-locally**,
   not at module top. The label handlers stay torch-free as-is.
3. **Move `CHOICES_VIEW_TYPE`** out of `pipeline.py` into a lightweight module
   (`cardio_form/config.py` or a new `cardio_form/constants.py`) so the parser
   can reference it without importing torch. Update `pipeline.py` and
   `entrypoint.py` to import it from the new home.
4. **Slim `entrypoint.py`** to: build the argparse parser → dispatch to
   `handlers`. Remove the per-submodule imports of `run_*_job`; import from
   `handlers` instead.
5. **Delete the `main()` + `__main__` blocks** from the 6 sibling files
   (resolves the §2.1 drift). See decision D1 for whether the sibling files are
   removed entirely or kept as thin re-export shims.

**Exit criteria**
- `cardioform --help` and every `cardioform <cmd> --help` run **without
  importing torch** (verify with `-X importtime` or a guard test).
- All commands produce identical behaviour and output files to before.
- Argument defaults/choices are defined **once**.

### Phase B — Click migration (after upgrade-plan Phase 1 CLI tests exist)

Only begins once the CLI integration tests from the upgrade plan are green.

1. Replace the argparse parser in `entrypoint.py` with a Click group +
   sub-commands; each sub-command is a thin wrapper calling the existing
   `handlers.run_*_job` (handlers are unchanged → the refactor is contained).
2. Add `click` to `pyproject.toml` runtime dependencies (pure-Python,
   conda-independent — fits the existing dependency split).
3. Preserve the public CLI contract (command names, flags, aliases) so the
   Phase 1 tests pass unchanged; any intentional surface change is called out
   and the tests updated deliberately.

**Exit criteria**
- Phase 1 CLI integration tests pass against the Click implementation with no
  unexplained contract changes.
- `--help` still does not load torch (Click groups import lazily the same way).

## 5. Open decisions (need maintainer input before execution)

- **D1 — per-module entrypoints.** Keep `python -m cardio_form.cli.<cmd>`
  working (re-point each sibling at `handlers`), or commit fully to the single
  `cardioform` console script and delete the sibling files? Recommendation:
  delete — the single console script makes them redundant.
- **D2 — Click timing.** Confirm Click is deferred behind the upgrade-plan
  Phase 1 CLI tests (recommended), or bundle it into Phase A despite the
  no-net-safety risk.
