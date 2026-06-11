# CardioForm Development Roadmap

This document outlines the planned tasks for improving, refactoring, and finalizing the CardioForm library. The tasks are organized by priority, from immediate fixes to major architectural upgrades.

### Priority 0: Environment Health — Python 3.11 Upgrade & Test Hardening

> Full step-by-step plan: [`docs/python-3_11-upgrade-plan.md`](./python-3_11-upgrade-plan.md).
> Driver: the env is pinned to **Python 3.9 (EOL Oct 2025)**, which forces a cascade
> of brittle secondary pins (`numpy==1.24.4`, `python-gdcm==3.2.2`). The GPU/CUDA path
> stays (Linux server); macOS uses the CPU file. Bump to 3.11 **only behind a test net**.

-   [ ] **Phase 0 — Unblock macOS & housekeeping** (no version changes)
    -   **Goal:** Let macOS users install today; stop the next person hitting the same wall.
    -   **Plan:** Fix `setup.sh` activation-name mismatch (`cardioform` vs `cardioform-cpu`);
        drop the dead `nvidia` channel from `environment-cpu.yaml`; document the
        Linux-GPU / macOS-CPU split in `README.md` (incl. why `environment.yaml`
        cannot solve on macOS). See plan §Phase 0.
-   [ ] **Phase 1 — Test harness + baseline on Python 3.9**
    -   **Goal:** Build the safety net while still on 3.9 to capture known-good behaviour.
    -   **Plan:** Add `pytest` `dev` extra + `heavy` marker; unit (characterization) tests
        for the light tier (`geometry`, `io`, `labels`, `config`, `models`,
        `output_managers`, `utils`); CLI integration tests via argparse/subprocess;
        skip-by-default heavy smoke test for `segment_2d`; CI workflow for the light
        suite. See plan §Phase 1.
-   [ ] **Phase 2 — Python 3.11 bump (on a branch)**
    -   **Goal:** Bump to 3.11 and dissolve the forced pins, validated against the Phase 1 baseline.
    -   **Plan:** `python=3.11` in both env files; unpin `python-gdcm`; NumPy → `>=1.26,<2.0`;
        fix known landmines (`scipy.ndimage.morphology` import in `geometry.py`,
        `torch.autograd.Variable` in `reconstruct_*`); rerun suite; validate both Docker
        images + GPU server. See plan §Phase 2.
-   [ ] **Phase 3 — Relax the pin strategy**
    -   **Goal:** Reduce future brittleness once 3.11 is proven.
    -   **Plan:** Convert exact `==` pins to compatible ranges; capture reproducibility in a
        separate generated lockfile rather than the source-of-truth env files. See plan §Phase 3.

### Priority 1: Restore Core Functionality

-   [ ] **Fix the GPU Docker Image Build Failure**
    -   **Goal:** Restore the automated building and publishing of the GPU-enabled Docker image.
    -   **Problem:** The `v0.2.0` release failed because the GitHub Actions runner ran out of disk space during the `conda` environment creation for the GPU image.
    -   **Plan:**
        1.  Modify `docker-publish.yml` to include the `easimon/maximize-build-space-action` step to free up runner disk space.
        2.  Re-enable the "Build and push GPU Docker image" step in the workflow.
        3.  Trigger a test build by creating a new release (e.g., `v0.3.0-alpha`) or re-running a previous workflow to verify the fix.

### Priority 2: New Features & Immediate Refactoring

> **v0.3 status:** both Priority 2 items DONE. `cardio_form/io.py` added; `relabel`
> shipped as `cardioform labels relabel` (orchestrator `cli/relabel.py`); all
> geometry I/O moved out into `io.py` (label ops + the recon plane/contour loaders,
> now `compute_*` taking arrays). Note: orchestrators live in `src/cardio_form/cli/`,
> not `scripts/` (P3 src-layout move).

-   [x] **Implement the `relabel` Feature** (v0.3)
    -   **Goal:** Provide users with a CLI tool to remap, merge, or delete segmentation labels from a NIfTI file.
    -   **Plan:**
        1.  Create a new `cardio_form/io.py` module to centralize NIfTI loading and saving logic.
        2.  Add a pure, I/O-free `remap_labels(data: np.ndarray, mapping: dict)` function to `cardio_form/geometry.py`.
        3.  Create an orchestrator script, `scripts/utils/run_relabel.py`, to handle CLI arguments and orchestrate the I/O and geometry calls.
        4.  Integrate the new script into `docker/entrypoint.py` under a `relabel` mode.

-   [x] **Refactor `geometry.py` to Separate I/O from Logic** (v0.3)
    -   **Goal:** Reduce technical debt and improve testability by ensuring core logic functions do not perform file I/O.
    -   **Plan:**
        1.  Apply the pattern established by the `relabel` feature to all other functions in `geometry.py` that currently handle file paths (e.g., `filter_labels`, `merge_labels`, `load_sax_plane_geometry`).
        2.  Move all file loading/saving operations out of `cardio_form/geometry.py` and into the corresponding orchestrator scripts in the `scripts/` directory, using the new `cardio_form/io.py` module.

### Priority 3: Major Architectural Rework

-   [x] **Refactor Project to use a `src` Layout** (v0.3 — done as a full installable package: `pyproject.toml`, `pip install -e .`, console script `cardioform`, no PYTHONPATH/sys.path hacks. Dockerfiles + setup.sh updated.)
    -   **Goal:** Align the project structure with modern Python packaging standards for improved clarity and prevention of common import issues.
    -   **Plan:**
        1.  Create a `src/` directory in the project root.
        2.  Move the entire `cardio_form/` package directory into `src/`.
        3.  Update all relevant paths in configuration and script files. This includes:
            -   `ENV PYTHONPATH=/app/src` in both `Dockerfile.cpu` and `Dockerfile.gpu`.
            -   Any paths in GitHub Actions workflows (`.github/workflows/`).
            -   Local development environment setup instructions.

-   [~] **Migrate to `pycemrg` Shared Library** (formerly "cemrg-core-utils"; PARTIAL)
    -   **Goal:** Centralize common code, reduce maintenance overhead, and align with dependent projects like `MyoScint`.
    -   **Done (v0.3):** `pycemrg` added as a dependency (declared in `pyproject.toml`; conda owns the
        scientific stack, installed via `pip install -e . --no-deps`). `LabelManager` migrated:
        `cardio_form/labels.py` is now a thin shim re-exporting `pycemrg.data.LabelManager`
        (schema + API are drop-in compatible).
    -   **Deferred to v0.3.1:**
        1.  `ModelManager` -> `pycemrg.assets.AssetManager` (needs `models.yaml` schema reconciliation
            for `unzipped_target_path`; touches the model download/unzip hot path).
        2.  `OutputManager` -> `pycemrg.files.OutputManager` (API differs: suffix registry vs raw suffix,
            `str` vs `Path`; needs a thin wrapper to preserve the canonical suffix map).
        3.  `utils.configure_logging` -> `pycemrg.core.logs.setup_logging` (pervasive call-site change).



