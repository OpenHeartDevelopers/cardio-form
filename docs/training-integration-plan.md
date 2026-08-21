# Training Integration Plan (DRAFT)

**Status:** draft, pending reconciliation with the upstream ML codebases.

This plan is written from the *inference* code in this repo. CardioForm is a
refactor of the ML engineers' original scripts, and their server-side work has
moved on since that refactor. Every contract below was reverse-engineered from
the code that loads and runs the checkpoints, not from the code that produced
them. Sections marked **[NEEDS UPSTREAM]** cannot be settled until their tree is
packaged, brought over, and read.

Read this as a checklist to hold against their code, not as an agreed design.

---

## 1. What exists here today

The repo is inference-only. Grepping `src/cardio_form/` for `optimizer`, `loss`,
`backward()`, `Dataset`, `DataLoader`, and `epoch` returns **nothing**. The two
3D nets are defined as `nn.Module` subclasses but are only ever loaded from a
checkpoint. There is no training loop, no data loader, no augmentation, no
metric, and no test suite to protect any of it.

Four models are in the manifest:

| Manifest key | Kind | Defined in | Weights |
|---|---|---|---|
| `segment_sax` | nnUNetv2 2D | external (nnUNet) | `segment_sax.zip` |
| `segment_lax_2ch` | nnUNetv2 2D | external (nnUNet) | `segment_lax_ch2.zip` |
| `segment_lax_4ch` | nnUNetv2 2D | external (nnUNet) | `segment_lax_ch4.zip` |
| `reconstruction_3d` | custom 3D U-Net | `reconstruct_3d.py:18` | `reconstruct_3d_epoch150_params.zip` |
| `la_reconstruction_3d` | custom 3D U-Net | `reconstruct_la_3d.py` | `la_3d_reconstruct_epoch_219_params.zip` |

The three `segment_*` entries are **not our architecture**. Retraining them means
driving nnUNetv2's own training CLI and repackaging its output; there is no model
code of ours to touch. The two reconstruction nets are ours and are the real
subject of this plan.

### 1.1 Contracts recovered from the inference path

These are the constraints any retrained checkpoint must satisfy to load and run
in this repo unchanged.

**Checkpoint format (both 3D nets).** `torch.load(...)` then
`unet.load_state_dict(checkpoint['model_state_dict'])`
(`reconstruct_3d.py:123-124`, `reconstruct_la_3d.py:319-320`). The checkpoint is
a dict with at least the key `model_state_dict`. Anything else in it is ignored.

**Whole-heart net — `ReconstructUNet3D`** (`reconstruct_3d.py:18-104`):
- Constructed as `in_channel=1, out_channel=9` (`reconstruct_3d.py:121`). Nine
  output channels = background + the eight labels in `labels.yaml`.
- Three `MaxPool3d(2)` stages, so every spatial input dimension must be divisible
  by 8.
- Input is the sparse volume from `geometry.vol_grid_gen`, then
  `np.transpose(vol_sp, [1, 0, 2])`, then `[np.newaxis, np.newaxis, ...]`, then
  **multiplied by 30** (`reconstruct_3d.py:234-235`).
- Output is `np.argmax(prd, axis=1)[0, ...]` transposed back by `[1, 0, 2]`
  (`reconstruct_3d.py:254-255`).

**LA net — `ReconstructLaUNet3d`** (`reconstruct_la_3d.py`):
- Constructed as `in_channel=1, out_channel=6` (`reconstruct_la_3d.py:317`).
- Input grid is **fixed at 128x128x128**, centre `[64, 64, 64]`
  (`image_load_nifti_ori`, in `reconstruct_la_3d.py`).
- Input values are remapped before inference: `vol_in == 1 -> 1` and
  `vol_in == 2 -> 5`, summed, then **multiplied by 50**
  (`reconstruct_la_3d.py:334-340`).
- `np.flip(..., axis=1)` is applied to the input and again to the argmax output.

**The magic scale factors (30 and 50) are training-time normalisation constants
baked into the inference path.** They differ between the two nets. If the
upstream training code normalises differently now, a retrained checkpoint will
load fine and silently produce garbage. This is the single highest-risk item in
this plan.

**[NEEDS UPSTREAM]** Confirm 30 and 50 against their training preprocessing. If
they are derived rather than constant, the derivation must come across too.

### 1.2 The `ModelManager` naming contract

`get_model_path` (`models.py:65`) resolves a manifest entry to a local file, and
its unzip logic imposes a strict, undocumented naming rule
(`models.py:100-110`):

- A zip named `<name>.zip` is always extracted to `~/.cache/cardio_form/<name>/`.
- If the manifest key starts with `segment_`, it then expects
  `<name>/fold_all/checkpoint_final.pth`.
- Otherwise it expects `<name>/<name>.pth` — the `.pth` basename must equal the
  zip basename.

So `reconstruct_3d_epoch150_params.zip` must contain
`reconstruct_3d_epoch150_params.pth`. Because the epoch number is embedded in the
filename, **every retrain produces a new zip name and a new `.pth` name that must
match each other.** Any publish tooling has to enforce this or downloads will
succeed and then fail to resolve.

Note also that `models.py:174` ends with
`default_model_manager = ModelManager()` — a module-level singleton built at
import time. It contradicts the no-singletons rule in the developer manifest and
it makes importing `cardio_form.models` fail if `models.yaml` is missing. Flagged
here because the training work will import this module; not actioned.

---

## 2. Reconciliation step (do this first)

Nothing below can be finalised before their tree is here. Proposed sequence:

1. **Package their server code.** `git bundle` if it is a repo (preserves
   history, which is what tells us what changed since the refactor); a dated
   tarball of the working tree if it is not.
2. **Land it outside this repo**, as a sibling checkout, not a vendored
   subdirectory. It is a reference, not a dependency.
3. **Diff their model definitions against ours.** `ReconstructUNet3D` and
   `ReconstructLaUNet3d` were copied during the refactor. If their architecture
   drifted — channel widths, block structure, normalisation layers — our
   `load_state_dict` will raise on key or shape mismatch, and we will need either
   a versioned architecture or a state-dict migration.
4. **Extract the training preprocessing** and compare it against §1.1.
5. **Answer the open questions in §3**, then revise this document from draft to
   agreed plan.

---

## 3. Open questions requiring their codebase **[NEEDS UPSTREAM]**

Each of these changes the shape of the work. None can be answered from this repo.

**Q1 — Architecture drift.** Do their current `ReconstructUNet3D` /
`ReconstructLaUNet3d` still match ours exactly? If not, do we version the
architecture (so old checkpoints keep loading) or hard-cut?

**Q2 — Normalisation constants.** Are the `* 30` and `* 50` factors still
correct? Are they constants, or per-case derived values that happen to look
constant in our copy?

**Q3 — Training data layout.** What is on disk, in what directory structure, with
what naming? Is there a manifest/split file, or is the split implicit? This
determines whether the `Dataset` takes a manifest or a convention.

**Q4 — Label space at training time.** The whole-heart net emits 9 channels,
consistent with `labels.yaml`. The LA net emits **6**, and its input remap
(`1 -> 1`, `2 -> 5`) implies a *different* label convention on the LAX inputs
than `labels.yaml` uses (where 1 is `LV_myo`, 2 is `LV_bp`). What are the LA
net's six classes, and what label space do its training inputs use?

**Q5 — Loss, optimiser, schedule.** What loss (Dice, cross-entropy, compound)?
Optimiser and LR schedule? These have no trace in our code at all.

**Q6 — Augmentation.** Any? Applied where — dataset or collate?

**Q7 — Checkpoint contents.** Does their checkpoint carry more than
`model_state_dict` (optimiser state, epoch, metrics)? Needed for resume support
and for deciding what we strip before publishing.

**Q8 — Validation and metrics.** How is a retrain judged good enough to promote?
Without this there is no gate on publishing new weights.

**Q9 — Hardware assumptions.** Does training assume multi-GPU, a specific CUDA
version, or a memory footprint incompatible with our pinned Torch?

**Q10 — nnUNet retraining.** Do they retrain the `segment_*` models at all, or
are those frozen? If they do, we need their `nnUNetv2_plan_and_preprocess` /
`nnUNetv2_train` invocation and the dataset JSON.

**Q11 — New functionality.** Your brief mentions "add a functionality". Is there
a specific new model or head in flight upstream? If so it should be designed into
the subpackage layout now rather than bolted on.

---

## 4. Proposed architecture (revise after §2)

### 4.1 New subpackage `src/cardio_form/training/`

Following the repo's orchestration/logic split: the training *logic* is stateless
and takes contracts; the CLI orchestrator owns file I/O, environment, and paths.

```
src/cardio_form/training/
├── __init__.py
├── contracts.py     # TrainingConfig, DataSplit, TrainingResult dataclasses
├── datasets.py      # Dataset classes; array-in/array-out preprocessing
├── preprocess.py    # PURE: the sparse-volume + scaling steps, shared with inference
├── loops.py         # train_reconstruction_3d(...), train_la_reconstruction_3d(...)
├── metrics.py       # validation metrics used for the promotion gate
└── nnunet.py        # thin wrapper driving nnUNetv2's own training CLI
```

`contracts.py` carries one dataclass per concern rather than long argument lists:

- `TrainingConfig` — model key, epochs, batch size, LR, optimiser name, device,
  output dir, seed, resume-from path.
- `DataSplit` — explicit train/val/test case lists. Explicit, not globbed, so a
  run is reproducible and a split is reviewable.
- `TrainingResult` — final checkpoint path, best epoch, metric history.

**The important structural point:** `preprocess.py` should hold the sparse-volume
construction and the scaling constants *once*, and both the training path and the
existing inference path should call it. Today those steps live inline in
`reconstruct_3d.py:234-235` and `reconstruct_la_3d.py:334-340`. Duplicating them
into the training code is exactly how the train/inference skew in §1.1 becomes
permanent. This is a refactor of existing inference code and should be called out
as such when scoping — it is not free.

### 4.2 CLI

A new orchestrator `src/cardio_form/cli/train.py` wired into the
`entrypoint.py` switchboard, matching the existing argparse style:

```bash
cardioform train reconstruction_3d --data-dir ... --split ... -o ... -p ... [--epochs N] [--device cuda]
cardioform train la_reconstruction_3d ...
cardioform train segment_sax ...        # delegates to nnUNetv2
```

Model keys deliberately match the `models.yaml` manifest keys so training,
publishing, and loading all speak one vocabulary.

---

## 5. Weights publishing flow

Proposed `helper_scripts/publish_weights.py`, run by hand after a training run
passes its gate. Steps:

1. Take a checkpoint path and a target version string (e.g. `v0.2.0`).
2. Strip the checkpoint to just `model_state_dict` unless resume support is
   wanted in the published artefact (see Q7).
3. Package into the exact layout `ModelManager` expects (§1.2): for the
   reconstruction nets, a zip `<name>.zip` containing `<name>.pth` with matching
   basenames; for `segment_*`, `fold_all/checkpoint_final.pth`.
4. Write the zip into `release_assets/`.
5. Compute SHA256.
6. Upload to a GitHub Release via `gh release upload`.
7. Emit the `models.yaml` stanza (key, version, URL, hash).

**Decision needed from you:** should step 7 *edit* `models.yaml` in place, or
print the stanza for manual paste? Printing is safer — it keeps a human in the
loop on which version becomes `default:` — but it is a manual step every time.

**Verification** is `helper_scripts/verify_models.py`, which already exists and
exercises `ModelManager`. A publish is not complete until a clean cache resolves
the new version end to end.

### 5.1 `models.yaml` versioning convention

Currently every model sits at `v0.1.0` with a `local_dev` sibling, and there is
no stated rule for promotion. Proposed and needing your sign-off:

- Versions are per-model, not global. Retraining `la_reconstruction_3d` alone
  bumps only that key.
- A new version is added as a new entry under `versions:`; **`default:` is a
  separate, deliberate commit.** Publishing and promoting are two acts.
- `local_dev` always points at `weights/` and is never promoted to `default`.
- Old versions are never deleted from the manifest, so a published pipeline can
  be reproduced.

---

## 6. Phasing

**Phase 0 — Reconciliation.** §2. Ends with this document revised and §3
answered. No code.

**Phase 1 — Shared preprocessing.** Extract the sparse-volume and scaling steps
into `training/preprocess.py` and route the *existing inference path* through it,
with no behaviour change. This is the riskiest refactor in the plan because there
is no test suite; it wants characterisation tests first (see §7).

**Phase 2 — Reconstruction training.** `contracts.py`, `datasets.py`, `loops.py`
for the two 3D nets. Success criterion: retraining from the existing data
reproduces the published checkpoints' metrics within tolerance.

**Phase 3 — CLI.** `cli/train.py` plus switchboard wiring.

**Phase 4 — Publishing.** `publish_weights.py` and the §5.1 convention, written
into `CLAUDE.md`.

**Phase 5 — nnUNet path.** `training/nnunet.py`, only if Q10 says they retrain
those models.

---

## 7. Named, not proposed

Work this plan touches but does not authorise:

- **Test coverage.** Phase 1 refactors untested inference code. Priority 0 Phase 1
  in `TASKS.md` is the characterisation-test harness and is the natural
  precondition. It is not in this plan's scope.
- **`AssetManager` migration.** `ModelManager` is local to this repo, not
  `pycemrg`'s. `TASKS.md` defers the move to `pycemrg.assets.AssetManager` to
  v0.3.1 pending a `models.yaml` schema reconciliation. Building publish tooling
  against the current local `ModelManager` means that tooling will need revisiting
  after the migration.
- **The `default_model_manager` singleton** (`models.py:174`).
- **GPU CI.** Still blocked on runner disk exhaustion; training will not run in CI
  regardless.
- **Hardcoded filename in `segment_2d.py:78`**, which builds its output name
  inline instead of via `OutputManager`, against the rule in `CLAUDE.md`.

---

## 8. Principal risks

1. **Silent train/inference skew** via the scaling constants (§1.1). Mitigated by
   Phase 1, which is why Phase 1 comes before Phase 2.
2. **Architecture drift** (Q1) making old checkpoints unloadable, discovered late.
3. **No promotion gate** (Q8) — without agreed metrics, "retrained" and
   "better" are unrelated claims.
4. **Naming-contract breakage** (§1.2) — a mismatched zip/`.pth` basename fails
   only at model-resolution time, on a user's machine, after a download.
