# Python 3.11 Upgrade & Test-Hardening Plan

**Status:** proposed (awaiting execution)
**Owner:** Jose Alonso Solis-Lemus
**Created:** 2026-06-11

## 1. Motivation

CardioForm's environment is pinned to **Python 3.9**, which reached end-of-life
in October 2025. That single pin forces a cascade of brittle secondary pins:

| Pin | Why it exists today | Consequence |
| --- | --- | --- |
| `python=3.9` | original development environment | EOL — no further security fixes |
| `numpy==1.24.4` | last NumPy that resolves cleanly on py3.9 + torch 2.5.1 | itself EOL |
| `python-gdcm==3.2.2` | comment: 3.2.6 is sdist-only, no `cp39` wheel | the pin exists **only** because of py3.9 |
| `pytorch=2.5.1` + `pytorch-cuda=11.8` (exact) | reproducibility on the Linux GPU box | makes `environment.yaml` unsolvable on macOS |

The `gdcm` pin is the clearest symptom: it works around a problem that exists
*solely* because Python is held at 3.9. Bumping to **Python 3.11** is expected
to dissolve several of these pins outright.

The risk is silent breakage: there are **no tests** in the repository today, so
a version bump could change behaviour without anyone noticing. This plan
therefore builds the safety net **before** touching any versions.

### Constraints captured from the maintainer

- **GPU/CUDA must stay** — it is used when CardioForm runs on a Linux server.
  The GPU environment file is not being removed; macOS simply uses the CPU file.
- The maintainer is open to the 3.11 bump **but only with unit and integration
  tests in place first** to prove nothing broke.

## 2. Known landmines (found during survey)

These are concrete breakages a newer dependency stack will surface. They are the
reason tests come first.

1. **`scipy.ndimage.morphology` removed.** `src/cardio_form/geometry.py` imports
   `from scipy.ndimage.morphology import binary_erosion`. That namespace was
   deprecated and **removed in modern SciPy**; a 3.11 environment will pull a
   SciPy new enough to break this import on load. Correct form:
   `from scipy.ndimage import binary_erosion`.
2. **`torch.autograd.Variable` deprecated.** Used in `reconstruct_3d.py` and
   `reconstruct_la_3d.py`. Still functional but should be modernised to plain
   tensors during this work.
3. **NumPy 2.0 incompatibility.** torch 2.5.1 and nnU-Net v2 are **not** safe on
   NumPy 2.x. The upgrade targets the **NumPy 1.26** line (`>=1.26,<2.0`), not
   a blind unpin.
4. **`python-gdcm` pin obsolete on 3.11.** Real `cp311` wheels exist for newer
   `python-gdcm`; the pin and its comment should be removed.

## 3. Codebase testability map

The library splits into two tiers. This split drives the entire test strategy:
the light tier is fully testable on a Mac with no GPU and no model weights.

| Tier | Modules | Heavy deps |
| --- | --- | --- |
| **Light** (no torch / no nnU-Net) | `labels`, `config`, `io`, `utils`, `output_managers`, `models`, `geometry` (887 lines), all `cli/*` | numpy, nibabel, scipy, SimpleITK, argparse |
| **Heavy** (torch / nnU-Net + weights) | `pipeline`, `segment_2d`, `reconstruct_3d`, `reconstruct_la_3d` | torch, nnUNetv2, model assets |

---

## Phase 0 — Unblock macOS & housekeeping

Small, no-risk changes that let the maintainer install today and stop the next
person hitting the same wall. No version changes.

### Steps

1. **Fix `setup.sh` activation mismatch.** `setup.sh` hardcodes
   `conda activate cardioform`, but the CPU file creates `cardioform-cpu`.
   Make the activation name parameterised or document both names so CPU users
   are not left activating a non-existent environment.
2. **Drop the dead `nvidia` channel from `environment-cpu.yaml`.** It is ignored
   on macOS/CPU and only adds solver noise.
3. **Document the platform split in `README.md`:**
   - `environment.yaml` → **Linux + GPU** (CUDA 11.8).
   - `environment-cpu.yaml` → **macOS / CPU-only Linux** (uses MPS-capable or
     CPU torch build).
   - Explicitly state that `environment.yaml` **cannot** solve on macOS because
     `pytorch-cuda` has no macOS build — this is the error the maintainer hit.

### Exit criteria

- `conda env create -f environment-cpu.yaml` succeeds on Apple Silicon.
- README clearly tells a new user which file to use on which platform.

---

## Phase 1 — Test harness + baseline on Python 3.9

Build the safety net **while still on 3.9**, so the suite captures *current*
(known-good) behaviour. This green baseline is what Phase 2 is validated against.

### 1a. Tooling

1. Add a `dev` extra to `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   dev = ["pytest>=7", "pytest-cov"]
   ```
2. Add `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and register a
   custom marker `heavy` (torch/nnU-Net/model-weight tests, skipped by default).
3. Create `tests/` with `tests/unit/`, `tests/integration/`, and a `conftest.py`
   holding shared synthetic fixtures (small numpy volumes, tiny NIfTI files via
   nibabel, a temp-dir factory). **No patient data** enters the repo.

### 1b. Unit tests — light tier (characterization)

The goal is to lock in present behaviour, not to assert correctness from first
principles. Cover, in priority order:

1. **`geometry.py`** (highest value — 887 lines, pure logic):
   - `remap_labels` / label merge / filter operations on small synthetic arrays.
   - the `compute_*` plane/contour functions, fed hand-built arrays.
   - erosion-dependent paths (these exercise the `scipy` import that Phase 2
     will fix — a deliberate canary).
2. **`io.py`** — round-trip NIfTI load/save on a temp file; affine/header
   preservation.
3. **`labels.py`** — the `pycemrg.data.LabelManager` shim re-exports correctly;
   schema load from `labels.yaml`.
4. **`config.py`, `models.py`, `output_managers.py`, `utils.py`** — config
   parsing, suffix-map construction, logging configuration smoke tests.

### 1c. Integration tests — CLI tier

1. Drive each `cardioform` subcommand through its argparse entrypoint via
   `subprocess` (or by calling `main()` with crafted `argv`), asserting:
   - `--help` exits 0 for every subcommand (cheap import-health check across
     the whole CLI surface).
   - the I/O-only subcommands (`labels relabel`, `merge_labels`,
     `filter_labels`) produce expected output files from synthetic NIfTI input.
2. **Heavy tier:** one `@pytest.mark.heavy` smoke test for `segment_2d` that is
   **skipped by default** (no weights in CI). It documents how to run locally
   with a real model so the heavy path is not entirely untested.

### 1d. Continuous integration

1. Add `.github/workflows/tests.yml` running the **light** suite (everything not
   marked `heavy`) on push / PR, on Python 3.9 (matching the current env), using
   `pip install -e .[dev] --no-deps` plus the minimal scientific deps.
2. This workflow's first green run is the **baseline of record**.

### Exit criteria

- Light unit + CLI integration tests pass locally and in CI on Python 3.9.
- `pytest -m "not heavy"` is green; `heavy` tests are collected but skipped.

---

## Phase 2 — Python 3.11 bump (on a branch)

Only begins once Phase 1 is green. Done on a dedicated branch so `main` stays
installable throughout.

### Steps

1. **Bump both env files to `python=3.11`** (`environment.yaml` keeps
   `pytorch-cuda=11.8`; `environment-cpu.yaml` stays CUDA-free).
2. **Unpin what 3.9 forced:**
   - Remove `python-gdcm==3.2.2` pin (and its comment) → use a 3.11-compatible
     release.
   - Move NumPy to `>=1.26,<2.0` (stay below 2.0 for torch/nnU-Net safety).
3. **Recreate the env and run the Phase 1 suite.** Fix breakages in this order:
   - `geometry.py`: `from scipy.ndimage import binary_erosion`.
   - `reconstruct_*`: replace `torch.autograd.Variable` with plain tensors.
   - any further failures surfaced by the tests.
4. **Re-run the heavy smoke test locally** (with real weights) on 3.11 to
   confirm the torch/nnU-Net path still loads and predicts.
5. **Update CI** matrix to Python 3.11 (optionally keep 3.9 transiently for one
   release to ease migration).
6. **Validate the Docker images** (`Dockerfile.cpu`, `Dockerfile.gpu`) build on
   the 3.11 env — these are the production artifacts.

### Exit criteria

- Full light suite green on 3.11; heavy smoke test passes locally with weights.
- Both Docker images build and the GPU image still runs on the Linux server.

---

## Phase 3 — Relax the pin strategy (durability)

Reduce future brittleness once 3.11 is proven.

### Steps

1. In the human-edited env files, convert exact `==` pins to **compatible
   ranges** (`>=,<`) for everything except the deliberate hard axis
   (`pytorch-cuda` on the GPU file).
2. Capture exact reproducibility in a **separate generated lockfile**
   (`conda-lock` or `conda env export --no-builds`) rather than baking exact
   versions into the source-of-truth files. Humans edit ranges; the lockfile
   records the resolved truth.
3. Document the workflow in README: edit ranges → regenerate lock → commit both.

### Exit criteria

- Env files express intent (ranges); a committed lockfile pins reality.
- A documented one-command path regenerates the lockfile.

---

## Sequencing & recommendation

- **Phase 0 + Phase 1 together** is the recommended first deliverable: it
  unblocks macOS immediately *and* stands up the safety net.
- Heavy-tier tests stay **skip-by-default** so real coverage lands on the light
  tier without needing patient data or weights in CI.
- Phase 2 starts only against a green Phase 1 baseline; Phase 3 follows once 3.11
  is proven in Docker and on the GPU server.
