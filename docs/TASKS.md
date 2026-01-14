# CardioForm Development Roadmap

This document outlines the planned tasks for improving, refactoring, and finalizing the CardioForm library. The tasks are organized by priority, from immediate fixes to major architectural upgrades.

### Priority 1: Restore Core Functionality

-   [ ] **Fix the GPU Docker Image Build Failure**
    -   **Goal:** Restore the automated building and publishing of the GPU-enabled Docker image.
    -   **Problem:** The `v0.2.0` release failed because the GitHub Actions runner ran out of disk space during the `conda` environment creation for the GPU image.
    -   **Plan:**
        1.  Modify `docker-publish.yml` to include the `easimon/maximize-build-space-action` step to free up runner disk space.
        2.  Re-enable the "Build and push GPU Docker image" step in the workflow.
        3.  Trigger a test build by creating a new release (e.g., `v0.3.0-alpha`) or re-running a previous workflow to verify the fix.

### Priority 2: New Features & Immediate Refactoring

-   [ ] **Implement the `relabel` Feature**
    -   **Goal:** Provide users with a CLI tool to remap, merge, or delete segmentation labels from a NIfTI file.
    -   **Plan:**
        1.  Create a new `cardio_form/io.py` module to centralize NIfTI loading and saving logic.
        2.  Add a pure, I/O-free `remap_labels(data: np.ndarray, mapping: dict)` function to `cardio_form/geometry.py`.
        3.  Create an orchestrator script, `scripts/utils/run_relabel.py`, to handle CLI arguments and orchestrate the I/O and geometry calls.
        4.  Integrate the new script into `docker/entrypoint.py` under a `relabel` mode.

-   [ ] **Refactor `geometry.py` to Separate I/O from Logic**
    -   **Goal:** Reduce technical debt and improve testability by ensuring core logic functions do not perform file I/O.
    -   **Plan:**
        1.  Apply the pattern established by the `relabel` feature to all other functions in `geometry.py` that currently handle file paths (e.g., `filter_labels`, `merge_labels`, `load_sax_plane_geometry`).
        2.  Move all file loading/saving operations out of `cardio_form/geometry.py` and into the corresponding orchestrator scripts in the `scripts/` directory, using the new `cardio_form/io.py` module.

### Priority 3: Major Architectural Rework

-   [ ] **Refactor Project to use a `src` Layout**
    -   **Goal:** Align the project structure with modern Python packaging standards for improved clarity and prevention of common import issues.
    -   **Plan:**
        1.  Create a `src/` directory in the project root.
        2.  Move the entire `cardio_form/` package directory into `src/`.
        3.  Update all relevant paths in configuration and script files. This includes:
            -   `ENV PYTHONPATH=/app/src` in both `Dockerfile.cpu` and `Dockerfile.gpu`.
            -   Any paths in GitHub Actions workflows (`.github/workflows/`).
            -   Local development environment setup instructions.

-   [ ] **Migrate to `cemrg-core-utils` Shared Library**
    -   **Goal:** Centralize common code, reduce maintenance overhead, and align with dependent projects like `MyoScint`.
    -   **Plan:**
        1.  Add `cemrg-core-utils` as a dependency in `environment.yaml` and `environment-cpu.yaml`.
        2.  Remove the local implementations of `ModelManager`, `LabelManager`, and `OutputManager` from the `cardio_form` codebase.
        3.  Refactor all internal code to import and use these managers from the new shared library. This will primarily affect `pipeline.py`, `segment_2d.py`, and the entrypoint.



