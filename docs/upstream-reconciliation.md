# Upstream Reconciliation — CardioForm vs. the two server codebases

**Status:** draft for review. Nothing implemented. This document ends at a proposed scope.

**Date:** 2026-08-21. **Author:** coordinator synthesis of two independent read-only audits.

Evidence:
- `report_A_Whole_Heart_MRI_Segmentation.md` — audit of tree A (42 files, 9 Python).
- `report_B_Whole_heart_pipline.md` — audit of tree B (22,482 files, 88 Python).
- Direct reading of `/home/jsolisle/dev/python/cardio-form` by the coordinator.

Each agent saw exactly one server tree and never saw CardioForm. All three-way
comparisons below are the coordinator's, from reading the third side directly.

---

## 1. Verdict

**Tree B is the active research tree. Tree A is a curated export of B, produced on
2026-08-05. Neither is a divergent fork.**

The evidence is direct, not inferential. `create_clean_whole_heart_repo.sh:4-5` in
tree B hard-codes:

```
SOURCE="/data/Abdul/Whole_heart_pipline"
DEST="/data/Abdul/Whole_Heart_MRI_Segmentation"
```

The script's body copies only the four inference scripts, the two LA mapping
scripts, and the model weights, and strips the mesh, QC and longitudinal tooling.
That is exactly the content of tree A. The script's mtime is 2026-08-05, matching
tree A's `test_*.log` files (2026-08-05, 08:49-09:11), which record the runtime
path `/data/Abdul/Whole_Heart_MRI_Segmentation/...`.

This resolves the mtime paradox in the brief. Tree A holds the single newest
Python mtime because it was *written* most recently, by the export. Tree B holds
the newer bulk because that is where work continues.

**Consequence for CardioForm:** tree A is the closer relative — it is the same
inference subset CardioForm packages, exported with a refreshed environment and
verified end-to-end (its six test logs show a complete successful run on
`MASTER001`). Tree B is where anything *new* must be sourced from. Use A as the
reference implementation, B as the parts bin.

One caveat worth recording, corrected 2026-08-28: tree A is newer in
**packaging**, and the export **did** strip newer models. The five production
weights in A are copies of B's, identical in size and mtime (2025-08-12), and A
also carries four `*_longi_compatible*.pth` files (2026-04-14) that are present
and identical in B. But B holds three further retrained cohort families that the
export dropped entirely — see §5.

---

## 2. Architecture compatibility

**Both checkpoints load into CardioForm's classes unchanged. No migration needed.**

The two server trees define one class, `UNet3d`, duplicated verbatim into two
files (`whole_heart_3d/utils.py:12-98` and `LA_3d_seg/utils.py:211-297` in tree A;
`:12-98` and `:210-296` in tree B). CardioForm splits it into two named classes,
`ReconstructUNet3D` (`reconstruct_3d.py:18`) and `ReconstructLaUNet3d`
(`reconstruct_la_3d.py:15`), whose bodies are also identical to each other.

`load_state_dict` matches on parameter name and shape, not class name. The
attribute names are the same on both sides:

`conv_encode1`, `conv_maxpool1`, `conv_encode2`, `conv_maxpool2`, `conv_encode3`,
`conv_maxpool3`, `bottleneck`, `conv_decode3`, `conv_decode2`, `final_layer`
(CardioForm `reconstruct_3d.py:59-79`; tree A `utils.py:53-73`; tree B same).

Structure matches at every checked point: three `MaxPool3d(kernel_size=2)`,
widths 16/32/64/128/256, `crop_and_concat` helper, `contracting_block` /
`expansive_block` / `final_block`.

Constructed channel counts agree exactly:

| Net | Server trees | CardioForm |
|---|---|---|
| Whole heart | `UNet3d(in_channel=1, out_channel=9)` (A `utils.py:520`, B `utils.py:520`) | `ReconstructUNet3D(in_channel=1, out_channel=9)` (`reconstruct_3d.py:121`) |
| LA | `UNet3d(in_channel=1, out_channel=6)` (A `utils.py:304`, B `utils.py:303`) | `ReconstructLaUNet3d(in_channel=1, out_channel=6)` (`reconstruct_la_3d.py:316`) |

Checkpoint key is `checkpoint['model_state_dict']` on both sides (A
`utils.py:521-522`, B `utils.py:521-522`, CardioForm `reconstruct_3d.py:123` and
`reconstruct_la_3d.py:319`).

**Answer to plan Q1: no drift. Do not version the architecture.**

---

## 3. The label-space map

This is the deliverable that fixes the live LA bug.

### 3.1 What each stage emits or expects

| Stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Proof |
|---|---|---|---|---|---|---|---|---|---|---|
| nnUNet 2D **SAX** output | bg | LV | Myo | RV | — | — | — | — | — | `Dataset314_SAXUKBB/.../dataset.json:5-10` |
| nnUNet 2D **LAX 2CH** output | bg | LV | Myo | LA | — | — | — | — | — | `Dataset282_LAX2CH/.../dataset.json:5-10` |
| nnUNet 2D **LAX 4CH** output | bg | LV | Myo | RV | LA | RA | — | — | — | `Dataset283_LAX4CH/.../dataset.json:5-12` |
| Whole-heart 3D **input** (sparse vol) | bg | LV bp | Myo | RV bp | *unused* | LA bp | RA bp | *unused* | *unused* | `px2vx_*` tables, below |
| Whole-heart 3D **output** | bg | LV myo | LV blood | RV | **Aorta?** | LA | RA | **Aorta/PA?** | PA | `qc_whole_heart_3d.py:32-42` (tree B) — reverse-engineered |
| CardioForm `labels.yaml` | bg | LV_myo | LV_bp | RV_bp | **MYO_septum?** | LA_bp | RA_bp | **Ao?** | PA | `labels.yaml` |
| LA 3D **input** (after relabel) | bg + Myo | LA | LV | RA | RV | — | — | — | — | `LA_*_label_mapping.py` |
| LA 3D **tensor** (after internal remap) | bg | LA | — | — | — | LV | — | — | — | `reconstruct_la_3d.py:336,338` |
| LA 3D **output** | ? | ? | ? | ? | ? | ? | — | — | — | **no legend exists in either tree** |

### 3.2 The whole-heart path — already correct in CardioForm

`vol_grid_gen` applies three lookup tables (index = source label, value = target):

| Table | Value | CardioForm | Tree A |
|---|---|---|---|
| `px2vx_sax` | `[0, 1, 2, 3]` | `geometry.py:551` | `utils.py:370` |
| `px2vx_2ch` | `[0, 1, 2, 5]` | `geometry.py:471` | `utils.py:330` |
| `px2vx_4ch` | `[0, 1, 2, 3, 5, 6]` | `geometry.py:509` | `utils.py:348` |

Identical values. The back-projection inverses are also present and identical
(`geometry.py:624, 663, 705`). **The whole-heart path needs no change.**

Applying these to the nnUNet outputs gives the input space in the table above:
2CH `LA(3) -> 5`, 4CH `LA(4) -> 5` and `RA(5) -> 6`, SAX identity.

### 3.3 The LA path — the live bug, located

**`cli/reconstruct_la.py` passes `ch2_file` and `ch4_file` straight through to
`run_la_reconstruction` (lines 21-22, 39-40). No relabel happens anywhere.**

The LA net expects `1 = LA, 2 = LV, 4 = RV` on its LAX inputs. It reads those
three values in two separate places:

- `reconstruct_la_3d.py:336, 338` — `vol_in == 1 -> 1` and `vol_in == 2 -> 5`,
  which is the only content that reaches the tensor.
- `reconstruct_la_3d.py:154, 161, 167, 170, 173` — `lax_coords` reads
  `v == 1` (LA), `v == 2` (LV) and `v == 4` (RV) to place the 128x128x128 grid.

The second is easy to miss and matters as much as the first: **a wrong label 4
misplaces the grid before the network ever runs.** The brief's fingerprint
("1 = LA, 2 = LV") is correct but incomplete on this point.

Feeding raw nnUNet 4CH output instead produces: LA read as LV(1), LV read as
Myo(2), RV read as LA(4). Every anatomical reference is wrong, the grid centre is
wrong, and the run still completes without error.

**The fix, both dictionaries, identical in both trees, digit for digit:**

| View | Mapping | Result | Proof (A / B) |
|---|---|---|---|
| 2CH | `{1: 2, 2: 0, 3: 1}` | LA=1, LV=2 | `LA_2CH_label_mapping.py:29` in both |
| 4CH | `{1: 2, 2: 0, 3: 4, 4: 1, 5: 3}` | LA=1, LV=2, RA=3, RV=4 | `LA_4CH_label_mapping.py:30` in both |

Tree B has not revised them. There is one correct mapping, not two competing ones.

`geometry.remap_labels(data, mapping)` already exists in CardioForm
(`geometry.py:863`). The missing piece is orchestrator wiring, not new logic.

### 3.4 Two label conflicts that remain open

**(a) `labels.yaml` describes the whole-heart net's OUTPUT space only.** On the
2D outputs and the sparse volume, `1 = LV blood pool` and `2 = Myocardium` — the
*opposite* of `labels.yaml`, where `1 = LV_myo` and `2 = LV_bp`. Anyone running
`cardioform labels filter|merge|relabel` against a 2D segmentation or a sparse
volume is reading the wrong names. `labels.yaml` carries no statement of which
stage it applies to.

**(b) Channels 4 and 7 are contested, and neither source is authoritative.**

| Channel | `labels.yaml` | Tree B QC legend |
|---|---|---|
| 4 | `MYO_septum` ("problematic, often merged or deleted") | `Aorta` |
| 7 | `Ao` | `Aorta/PA` |
| 8 | `PA` | `PA` (agree) |

The QC legend's own comment (`qc_whole_heart_3d.py:31`) says it comes "from px2vx
mappings in utils.py **+ visual inspection**" — a human guess, not training
metadata. Channels 4, 7 and 8 are never assigned by any `px2vx` table, so no
input-side evidence constrains them either. **Two independent guesses that
disagree. This needs the ML engineers, not more grepping.**

**(c) The LA net's six output classes are undocumented in both trees.** No
`dataset.json`, no legend, no comment. Only the *input* encoding is proven.

---

## 4. Constants ledger

| Constant | CardioForm | Tree A | Tree B | Status |
|---|---|---|---|---|
| Whole-heart input scale | `* 30` (`reconstruct_3d.py:235`) | `* 30` (`utils.py:542`) | `* 30` (`utils.py:542`) | **identical, literal** |
| LA input scale | `* 50` (`reconstruct_la_3d.py:341`) | `* 50` (`utils.py:324`) | `* 50` (`utils.py:323`) | **identical, literal** |
| Whole-heart transpose | `[1, 0, 2]` in and out (`:234, :256`) | `:541, :549` | `:541, :549` | identical |
| LA flip | `np.flip(axis=1)` in and out (`:334, :351`) | `:317, :328` | `:316, :327` | identical |
| Whole-heart grid | — | 160^3, 10% margin | 160^3, 10% margin | identical |
| LA grid | 128^3, centre `[64,64,64]` (`:250-252`) | 128^3, centre `[64,64,64]` | 128^3, centre `[64,64,64]` | identical |

Both agents confirmed independently that `* 30` and `* 50` are **literals, not
per-case derived values**. What is multiplied is the label volume itself, not an
image intensity — an integer-scaling trick, so there is no per-case derivation to
port.

**Answer to plan Q2: the constants are correct and constant. The single highest
risk named in the training plan is retired.**

---

## 5. Portable assets

| Asset | Where | Value | Cost |
|---|---|---|---|
| **LA relabel dictionaries** | Both trees, `LA_*_label_mapping.py` | Fixes the live bug | **Small.** `remap_labels` exists; wire it into `cli/reconstruct_la.py`. |
| **Retrained cohort weights (Rosie, Vanderbilt, VU Pre/Post)** | B only, **not exported to A**. `2D_segmentation/model_weights/` and `.../results/`: `Dataset315_RosieSAX` (2 folds), `Dataset316_RosieLAX2CH` (2), `Dataset317_RosieLAX4CH` (5), `Dataset400_UVSAX`, `Dataset401_UVLAX2CH`, `Dataset402_UVLAX4CH` (fold_all), `Dataset404_UVSAX_PrePost` (2), `Dataset405_UVLAX2CH_PrePost` (5), `Dataset406_UVLAX4CH_PrePost` (5). Apr-May 2026, `nnUNetTrainer250epochs` or stock trainer. | Genuinely retrained 2D SAX/LAX models on three cohorts CardioForm has never seen. The strongest portable asset found. | **Medium-high.** Repackaging to `segment_*.zip` + `fold_all/checkpoint_final.pth` per `ModelManager` (plan §1.2); most are per-fold, not `fold_all`, so a fold choice or ensemble decision is needed first. Metrics **do** exist (found 2026-08-28): 29 `validation/summary.json` files. Rosie foreground Dice 0.917-0.938 on n=11-12; VU Pre/Post 0.812-0.900 on n=36-37; VU 400-402 report 0.982-0.991 but are `fold_all` runs, so not held out. **No UKBB baseline metrics exist**, so no cohort can be shown better than production. Rosie SAX and 2CH have only fold_0 validated and no `fold_all` model. |
| LongiSeg SAX weights | B `model_weights_longi/Dataset330_LongiSAX/`, Apr 2026, ~246 MB x3 trainer variants | Newer 2D SAX model | **Medium-high, with a risk.** Unproven whether LongiSeg-trained weights load under plain `nnUNetv2` — they may require the external `LongiSeg` package. Verify before committing. |
| `*_longi_compatible*.pth` | Both trees, `model_weights/SAX/` and `model_weights/LAX/`, 2026-04-14 | **Low.** Each is 10,636 bytes smaller than its Aug-2025 production counterpart (20,051 for `ukbb_sax_longi_compatible.pth`). A constant delta across three different models indicates a stripped metadata block, not retraining. These are the UKBB baselines reformatted as LongiSeg *inputs*, not better models. | None. Do not port. |
| nnUNet retraining recipe | B `prepare_longitudinal_datasets.py:409-434` | Answers plan Q10 in full | **Small** to record; large to run. |
| QC tooling | B `qc_whole_heart_3d.py` and others | A 9-panel per-subject PDF; heuristic flags | **Medium.** Self-contained, thresholds hand-set (`< 5000` voxels at `:168`). |
| Evaluation metrics | B `evaluate_longiseg_vs_baseline.py` | Dice, HD95, volume + temporal consistency, Wilcoxon | **Medium.** Nearest thing to a promotion gate that exists. |
| Mesh pipeline | B `cme-dt-devran-main`, `mesh_pipeline_scripts/` | BiV mesh, volume/EF, CRT response | **Large.** Shells out to an external `meshtool`/CARP binary (`meshtool_func.py:20-27`) and pins `torch==2.8`, conflicting with CardioForm's stack. |
| Longitudinal pipeline | B, Datasets 330-332 | Longitudinal segmentation | **Large.** External `LongiSeg` dependency. |

**Not portable — already present.** Tree A's report lists back-projection
(`vol_grid_bp`) as capability CardioForm lacks. That is wrong; the agent never
read CardioForm. `vol_grid_bp` is present at `geometry.py:580` and wired at
`reconstruct_3d.py:287` behind a `compute_bp=True` flag. No work needed.

**Not portable — does not exist.** Neither tree contains training code for the two
3D reconstruction nets. No loop, no loss, no optimiser, no `torch.save`. Both
agents searched independently and found nothing. **The training code for the nets
CardioForm actually owns is not in either tree.** Any v0.4.0 plan that assumes it
can be ported is built on a false premise.

**Unexplained, in both trees.** `LA_3d_seg/model_due/exp_plus_/epoch_219_params.pth`
(77,008,873 bytes) sits beside the file the code loads,
`exp_plus/epoch_219_params.pth` (25,687,017 bytes). Same claimed epoch, ~3x the
size, referenced by nothing. A 3x ratio is what a full training checkpoint
(weights + optimiser moments) costs over a stripped weights-only file. If that
guess holds, the orphan is the unstripped sibling and would answer plan Q7. Not
verified — see "Outside the plan".

---

## 6. Answers to `docs/training-integration-plan.md` §3

| # | Question | Status | Answer |
|---|---|---|---|
| Q1 | Architecture drift | **Answered** | No drift. Identical structure and attribute names; only the class name differs, which `load_state_dict` ignores. Do not version. |
| Q2 | Normalisation constants | **Answered** | `* 30` and `* 50` unchanged in both trees, literal, not derived. |
| Q3 | Training data layout | **Partial** | For nnUNet: standard `nnUNet_raw`/`nnUNet_preprocessed` convention, Dataset IDs 320-322 and 330-332, explicit split from `split_longi_datasets.py` (460 train / 116 val for Dataset330). For the two 3D nets: **unknown, no training code exists.** |
| Q4 | Label space at training time | **Partial** | All input spaces proven (§3). The LA net's six output classes are undocumented in both trees. Whole-heart output rests on a human-reverse-engineered QC legend that disagrees with `labels.yaml` on channels 4 and 7. |
| Q5 | Loss, optimiser, schedule | **Partial** | nnUNet 2D: SGD, `momentum 0.99`, `nesterov True`, `weight_decay 3e-05`, `initial_lr 0.0001` (SAX, LAX2CH) and `0.001` (LAX4CH), `num_epochs 1000`, poly decay. LongiSeg run: 250 epochs, lr `0.01` decaying to `7e-05`. For the two 3D nets: **nothing, no training code.** |
| Q6 | Augmentation | **Open** | nnUNet defaults for the 2D models. Nothing recoverable for the 3D nets. |
| Q7 | Checkpoint contents | **Open** | No `torch.save` call anywhere in either tree. Only the read side proves the `model_state_dict` key. The `exp_plus_` orphan is a lead (§5). |
| Q8 | Validation and promotion gate | **Answered, negatively** | No model-level gate in either tree. B's `evaluate_longiseg_vs_baseline.py` gives Dice/HD95/volume + temporal consistency and a Wilcoxon test, but as decision support, not a block. B's other gates are case-level QC, not model-level. |
| Q9 | Hardware assumptions | **Answered** | Single GPU throughout; zero `DataParallel`/`DistributedDataParallel` in either tree. CUDA 11.8. Pins conflict: tree A `torch==2.5.1`, `nnunetv2==2.8.1`, numpy `2.2.6`, Python 3.10; tree B instructions Python 3.9 + numpy `1.24.4`; B's `cme-dt-devran-main` `torch==2.8`. CardioForm is Python 3.9 / numpy 1.24.4. |
| Q10 | nnUNet retraining | **Answered** | Yes, they retrain. Exact invocations at `prepare_longitudinal_datasets.py:409-434`, warm-started from `Dataset314_SAXUKBB/.../checkpoint_final.pth`, trainer `nnUNetTrainer250epochs`. Retrained SAX weights exist (`model_weights_longi/`, Apr 2026). |
| Q11 | New functionality in flight | **Answered** | LongiSeg longitudinal segmentation, BiV-ME mesh fitting, volume/EF/CRT analysis. **None is a new head or model on our two nets** — no subpackage layout change is forced. |

Seven answered, three partial, one open.

**Note on Q9, worth acting on separately:** tree A already runs the stack that
`TASKS.md` Priority 0 is trying to reach — Python 3.10, `torch 2.5.1`,
`nnunetv2 2.8.1`, numpy 2.x — and its six test logs show a clean end-to-end run.
That is working evidence that the numpy 1.24.4 pin can be dissolved. It does not
prove CardioForm survives the bump, but it removes the main unknown.

---

## 7. Proposed v0.4.0 scope

Ordered. The cut line is marked.

**1. Fix the LA label-space bug.** Wire `geometry.remap_labels` into
`cli/reconstruct_la.py` with the two dictionaries from §3.3, applied to the 2CH
and 4CH inputs before `run_la_reconstruction`. Add a guard that refuses input
whose label set does not match either the raw nnUNet space or the mapped space,
so a wrong input fails loudly instead of silently. *This is the reason for the
investigation and the only item that must ship.*

**2. Make the label spaces explicit in `labels.yaml`.** Today the file states one
space and silently means the whole-heart output. Give it named spaces — the three
nnUNet 2D outputs, the sparse-volume input, the whole-heart output — so the
`labels` CLI can be told which one it is operating on. Mark channels 4 and 7
`unverified` rather than asserting `MYO_septum` and `Ao`.

**3. Record the constants ledger and the architecture verdict in the repo.**
Fold §2 and §4 into `docs/training-integration-plan.md`, moving it from draft to
agreed on those two points and closing Q1 and Q2.

--- **cut line for v0.4.0** ---

**Deferred, with reasons:**

- **Training code for the 3D nets.** Cannot be ported — it does not exist in
  either tree (§5). It must be requested from the ML engineers or written from
  scratch. Plan Phases 2-4 are blocked on this, not on our effort.
- **Shared `training/preprocess.py`** (plan Phase 1). Sound, but it refactors
  untested inference code, and `TASKS.md` Priority 0 Phase 1 (the
  characterisation-test harness) is its natural precondition.
- **Retrained SAX weights.** Real value, but gated on proving LongiSeg-trained
  weights load under plain `nnUNetv2` (§5).
- **QC tooling, mesh pipeline, longitudinal pipeline.** Each is a feature in its
  own right with external binary or package dependencies. None belongs in a
  bug-fix release.

**Rationale for the cut.** Item 1 is a correctness fix on a shipping code path.
Items 2 and 3 are documentation of facts now established, and they are what stops
this investigation being repeated. Everything below the line either needs
information we do not have, or needs the test suite that `TASKS.md` already ranks
as Priority 0.

---

## 8. Open for the ML engineers

Three questions no amount of reading their code can answer:

1. **What are the LA net's six output classes?** Undocumented in both trees.
2. **What are whole-heart output channels 4 and 7?** `labels.yaml` says
   `MYO_septum` and `Ao`; their QC script says `Aorta` and `Aorta/PA`. Both are
   guesses.
3. **Where is the training code for the two 3D reconstruction nets?** Neither
   tree contains it. Also: does `exp_plus_/epoch_219_params.pth` hold optimiser
   state, and is it the same run as `exp_plus/epoch_219_params.pth`?
