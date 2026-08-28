# Audit: Whole_heart_pipline (tree B)

ROOT = `/home/jsolisle/installs/from_server/Whole_heart_pipline`. Read-only audit; no code executed, no checkpoints loaded, no hashes computed.

> Persisted by the coordinator. The subagent harness blocked the agent's own Write call; content is verbatim from the agent's report, with HTML entities decoded.

## Top 5 findings

1. **The LA relabel step exists and is byte-for-byte identical to the fingerprint.** `2D_segmentation/codes/LA_mapping_code/LA_2CH_label_mapping.py:29` has `mapping = {1: 2, 2: 0, 3: 1}` and `LA_4CH_label_mapping.py:30` has `mapping = {1: 2, 2: 0, 3: 4, 4: 1, 5: 3}`. Not revised — an exact match. If CardioForm truly omits this step, this tree is the missing reference implementation, ready to port as-is.
2. **Both 3D U-Nets match the CardioForm fingerprint exactly on every checkable axis** — architecture, channel counts, scale factors (`* 30`, `* 50`), flips, checkpoint dict key (`checkpoint['model_state_dict']`). Only the class name differs (`UNet3d` here vs. `ReconstructUNet3D`/`ReconstructLaUNet3d`); internal submodule names are unchanged, so `load_state_dict` compatibility looks preserved.
3. **This tree is the acknowledged source of the sibling tree.** `create_clean_whole_heart_repo.sh:4-5` hard-codes `SOURCE="/data/Abdul/Whole_heart_pipline"` and `DEST="/data/Abdul/Whole_Heart_MRI_Segmentation"`, and the script's body copies only the four inference scripts, the LA mapping scripts, and the model weights — stripping out mesh/QC/longitudinal tooling. This establishes provenance without needing to open the sibling tree.
4. **This tree contains substantial capability CardioForm does not have**: a full LongiSeg longitudinal retraining pipeline (Datasets 330/331/332, logged training runs), a BiV-ME mesh-fitting pipeline (`cme-dt-devran-main`, dependent on an external `meshtool`/CARP binary), volumetric/EF analysis scripts, and QC tooling with hand-set pass/fail thresholds.
5. **No checkpoint newer than `epoch150`/`epoch_219` was found for the two 3D reconstruction nets** — those exact filenames are still what's loaded. However, `LA_3d_seg/model_due/exp_plus_/epoch_219_params.pth` (77,008,873 bytes) is an unexplained second file, ~3x the size of the one actually loaded by `model_load()` (`exp_plus/epoch_219_params.pth`, 25,687,017 bytes) — worth investigating before assuming there is only one LA checkpoint lineage.

---

## Q1 — Inventory

The canonical pipeline is documented in `whole_heart_pipline_instructions.txt` (5,132 bytes, top level) and matches the code exactly:

1. **2D segmentation** (nnUNetv2, per-view models):
   - SAX: `2D_segmentation/codes/SAX_code/predictions_SAX.py` — loads `Dataset314_SAXUKBB/nnUNetTrainer__nnUNetPlans__3d_fullres/` (`predictions_SAX.py:55`), `checkpoint_final.pth`, folds `('all',)`.
   - LAX 2CH: `2D_segmentation/codes/LAX_code/predictions_LAX_2ch.py` — loads `Dataset282_LAX2CH/nnUNetTrainer__nnUNetPlans__2d` (`predictions_LAX_2ch.py:37-38`).
   - LAX 4CH: `2D_segmentation/codes/LAX_code/predictions_LAX_4ch.py` — loads `Dataset283_LAX4CH/nnUNetTrainer__nnUNetPlans__2d` (`predictions_LAX_4ch.py:36-37`).
2. **3D whole-heart reconstruction**: `3D_segmentation/whole_heart_3d/whs_4ch_main.py`, calling `eval_bbk_devran` in `3D_segmentation/whole_heart_3d/utils.py`. Model dir arg required (`whs_4ch_main.py:191`); checkpoint file fixed as `epoch_150_params.pth` (`utils.py:511`).
3. **LA label mapping**: `2D_segmentation/codes/LA_mapping_code/LA_4CH_label_mapping.py` then `LA_2CH_label_mapping.py`, per the instructions file's Step 3.
4. **3D LA reconstruction**: `3D_segmentation/LA_3d_seg/LA_whole_heart_main.py`, calling `model_load()`/`mr_lax_inference()` in `3D_segmentation/LA_3d_seg/utils.py`.

The instructions file's directory diagram, environment setup (`conda create -n whole_heart python=3.9`, `pytorch-cuda=11.8`), and exact CLI invocations match the scripts as they exist on disk (`whole_heart_pipline_instructions.txt:1-100`).

Beyond this canonical 4-step pipeline, the tree holds four more distinct sub-pipelines not in the instructions file:

- **Longitudinal retraining pipeline** (top level): `prepare_longitudinal_datasets.py` builds nnUNet Datasets 320-322 and LongiSeg Datasets 330-332 from RMTL/Tred_HF/Rosie sources (`prepare_longitudinal_datasets.py:4-13`), prints the exact `nnUNetv2_plan_and_preprocess`/`nnUNetv2_train`/`LongiSeg_train` commands to run (`prepare_longitudinal_datasets.py:403-434`), `split_longi_datasets.py` creates the train/val split, and `evaluate_longiseg_vs_baseline.py` compares LongiSeg vs. baseline nnUNet with Dice/HD95/volume-consistency metrics. `train_330_baseline.log` and `train_330_fold0.log` are the actual training run logs.
- **VU (Vanderbilt) clinical CRT pipeline**: `2D_segmentation/extract_vu_dataset_nnunet.py` and `run_inference_vu_test.py` build/run inference on the VU Pre/Post CRT cohort (Datasets 400-406); `run_whole_heart_3d.py` (top level) batch-runs the whole-heart 3D net over `VU_corrected/`; `qc_whole_heart_3d.py` produces a 9-panel QC PDF per subject.
- **Mesh pipeline**: `mesh_pipeline_scripts/` (documented in its own `README.md`) plus the `VU_mesh_pipeline*` and `VU_mesh_results*` directories drive biventricular mesh generation via `cme-dt-devran-main` (a separate BiV-ME/contour-fitting codebase with its own `README.md` and `MANUAL_FIX_GUIDE.md`).
- **Manual-correction loop**: `check_update/` holds per-subject `contours/gp_points_file.txt` / `gp_frame_info_file.txt` corrections, driven by `manual_correction_guide.csv` (top level), feeding back into `VU_corrected/` and the mesh pipeline.

How they connect: 2D segmentation output feeds both the whole-heart 3D net directly, and (via the LA label-mapping step) the LA 3D net. VU_corrected (presumably QC/manually-corrected 2D output) feeds `run_whole_heart_3d.py` -> `VU_3d_segmentation/` -> `qc_whole_heart_3d.py` -> the mesh pipeline (`mesh_pipeline_scripts/`, `VU_mesh_pipeline_retrained_only/`) -> volume/EF analysis (`compute_mesh_volumes.py`, `clean_ef_analysis.py`, `plot_volume_analysis.py`) -> CRT response comparison (`VU_mesh_pipeline_retrained_only/crt_response_analysis.csv`). The longitudinal retraining pipeline is a parallel, separate track that only touches the 2D SAX/LAX nnUNet models (Datasets 330-332), not the 3D reconstruction nets.

## Q2 — Model architectures

Two 3D U-Net class definitions exist, both named `UNet3d`, defined independently (duplicated, not shared) in:
- `3D_segmentation/whole_heart_3d/utils.py:12-98`
- `3D_segmentation/LA_3d_seg/utils.py:210-296`

Both are structurally identical: `contracting_block`/`expansive_block`/`final_block` methods (`utils.py:13`, `:24`, `:37` in the whole-heart file; `:211`, `:222`, `:235` in the LA file), three `torch.nn.MaxPool3d(kernel_size=2)` calls (whole-heart `utils.py:54,56,58`; LA `utils.py:252,254,256`), a bottleneck, and a `crop_and_concat` helper (whole-heart `utils.py:75`; LA `utils.py:273`).

Constructed channel counts at the call site:
- Whole-heart: `UNet3d(in_channel=1, out_channel=9)` — `3D_segmentation/whole_heart_3d/utils.py:520`.
- LA: `UNet3d(in_channel=1, out_channel=6)` — `3D_segmentation/LA_3d_seg/utils.py:303`.

**Comparison to fingerprint: identical** in every structural respect (block names, 3 pooling stages, `crop_and_concat`, in/out channel counts) except the class name itself — this tree's `UNet3d` vs. CardioForm's `ReconstructUNet3D`/`ReconstructLaUNet3d`. Because `state_dict` keys derive from attribute names (`conv_encode1`, `conv_maxpool1`, `bottleneck`, `conv_decode3`, `final_layer`, etc.), not the class name, and those attribute names are unchanged, the checkpoints in this tree should still load into CardioForm's classes via `load_state_dict` — assuming CardioForm's attribute names match (not verified by the agent, since that repo was not opened).

## Q3 — Normalisation constants

Both scale factors are present, unchanged, and are literal constants (not computed per case):
- Whole-heart: `test_x = img_i_[np.newaxis, np.newaxis, ...] * 30` — `3D_segmentation/whole_heart_3d/utils.py:542`.
- LA: `test_x[0, 0, ...] = vol_in_ * 50` — `3D_segmentation/LA_3d_seg/utils.py:323`.

No case-dependent normalisation (e.g. intensity percentiles) feeds either of these two lines — the value multiplied is the label/probability volume itself (`vol_sp` / `vol_in_`), a fixed integer-scaling trick, not an image-intensity normalisation.

## Q4 — Label spaces (priority)

**2D SAX** (`Dataset314_SAXUKBB`, `2D_segmentation/model_weights/SAX/Dataset314_SAXUKBB/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json`):
```
"labels": {"background": 0, "LV": 1, "Myo": 2, "RV": 3}
```
Confirmed independently in `prepare_longitudinal_datasets.py:47`.

**2D LAX 2CH** (`Dataset282_LAX2CH/.../dataset.json`):
```
"labels": {"background": 0, "LV": 1, "Myo": 2, "LA": 3}
```

**2D LAX 4CH** (`Dataset283_LAX4CH/.../dataset.json`):
```
"labels": {"background": 0, "LV": 1, "Myo": 2, "RV": 3, "LA": 4, "RA": 5}
```

**Relabel step (2D -> LA-3D-net input space)**, exactly matching the fingerprint dictionaries, digit for digit:
- `LA_2CH_label_mapping.py:29`: `mapping = {1: 2, 2: 0, 3: 1}` — LV(1)->2, Myo(2)->0(background), LA(3)->1.
- `LA_4CH_label_mapping.py:30`: `mapping = {1: 2, 2: 0, 3: 4, 4: 1, 5: 3}` — LV(1)->2, Myo(2)->0, RV(3)->4, LA(4)->1, RA(5)->3.

Both are **exact matches** to the fingerprint dictionaries — not revised.

The mapped output (LA=1, LV=2) is then consumed by the LA 3D net's own internal remap: `3D_segmentation/LA_3d_seg/utils.py:318` (`vol_in_la[np.where(vol_in == 1)] = 1`) and `:320` (`vol_in_lv[np.where(vol_in == 2)] = 5`) — confirming the "1 = LA, 2 = LV" input convention.

**Whole-heart 3D net output** (argmax over 9 channels, `utils.py:548`): no `dataset.json` or training-side legend exists in the tree. The only legend found is in a downstream QC script, `qc_whole_heart_3d.py:32-42`, whose own comment (`:31`) says it is "from px2vx mappings in utils.py **+ visual inspection**" — reverse-engineered by a human, not proven training metadata:
```
0: Background, 1: LV myo, 2: LV blood, 3: RV, 4: Aorta, 5: LA, 6: RA, 7: Aorta/PA, 8: PA
```

**LA 3D net output** (argmax over 6 channels, `utils.py:327`): **no legend of any kind found anywhere in the tree** — these integers (0-5) are unlabelled here.

## Q5 — Training code

**2D nnUNet models**: no training loop, `Dataset` class, loss, or optimiser code is vendored — training is delegated entirely to the external `nnunetv2`/`LongiSeg` CLI packages:
- `prepare_longitudinal_datasets.py:409-434` documents:
  ```
  nnUNetv2_plan_and_preprocess -d 320 321 322 --verify_dataset_integrity -np 8
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 320 3d_fullres $fold -pretrained_weights .../Dataset314_SAXUKBB/.../checkpoint_final.pth -tr nnUNetTrainer250epochs
  LongiSeg_plan_and_preprocess -d 330 331 332 --verify_dataset_integrity
  CUDA_VISIBLE_DEVICES=1 LongiSeg_train 330 3d_fullres $fold -pretrained_weights .../Dataset314_SAXUKBB/.../checkpoint_final.pth
  ```
- `cme-dt-devran-main/src/segmentation/train.py:83-92,131` calls `nnUNetv2_plan_and_preprocess -d <id> -c 2d -pl nnUNetPlannerResEncL` then `nnUNetv2_train <dataset> 2d <fold> -p nnUNetResEncUNetLPlans` for 5 folds by default (`:66-68`).

**3D reconstruction nets**: no training code at all — only inference and the network class definitions. The checkpoint-writing code is not in this tree.

**Training log evidence** (`train_330_baseline.log`, `train_330_fold0.log` — LongiSeg on `Dataset330_LongiSAX`):
- Config: `3d_fullres`, `LongiSegPreprocessor`, `batch_size: 2`, `patch_size: [224, 160, 64]`, architecture `PlainConvUNet`, `n_stages: 6`, `features_per_stage: [32, 64, 128, 256, 320, 320]` (`train_330_baseline.log:24`).
- Split: "This split has 460 training and 116 validation cases." (`:16`).
- `train_330_baseline.log`: `Epoch 0`, `Current learning rate: 0.01`; tail shows `Epoch 258`, `Current learning rate: 0.00764`, `Pseudo dice [0.97, 0.9178, 0.9348]`.
- `train_330_fold0.log`: ends `Epoch 249`, `Current learning rate: 7e-05`, `Pseudo dice [0.975, 0.9345, 0.9474]`, then `Training done.` — consistent with a 250-epoch polynomial LR decay schedule.
- Internal timestamps: 2026-04-29.

## Q6 — Checkpoints and weights

**3D reconstruction net checkpoints:**

| Path | Size | mtime | Note |
|---|---|---|---|
| `3D_segmentation/whole_heart_3d/model/epoch_150_params.pth` | 77,039,978 bytes | Aug 12 2025 | Loaded by `eval_bbk_devran` (`utils.py:511`). |
| `3D_segmentation/LA_3d_seg/model_due/exp_plus/epoch_219_params.pth` | 25,687,017 bytes | Aug 12 2025 | Loaded by `model_load()` (`utils.py:306`). |
| `3D_segmentation/LA_3d_seg/model_due/exp_plus_/epoch_219_params.pth` | 77,008,873 bytes | Aug 12 2025 | **Not referenced by any script found.** Same claimed epoch, ~3x the size actually loaded — flag for investigation. |

No file suggests an epoch newer than 150/219 for these two 3D nets.

**No `.zip` weight files exist anywhere in this tree** — the 2D nnUNet models are standard nnUNet results-folder checkpoints, not CardioForm's `segment_*.zip` packaging.

**2D nnUNet production checkpoints:**
- `Dataset314_SAXUKBB/.../fold_all/checkpoint_final.pth` — 240,701,485 bytes, Aug 12 2025.
- `Dataset282_LAX2CH/.../fold_all/checkpoint_final.pth` — 165,563,053 bytes, Aug 12 2025.
- `Dataset283_LAX4CH/.../fold_all/checkpoint_final.pth` — 165,705,709 bytes, Aug 12 2025.

**`model_weights_longi/`** exists, holds later (Apr 2026) retrained LongiSeg checkpoints for `Dataset330_LongiSAX` only, across three trainer variants (`LongiSegTrainer250epochs`, `nnUNetTrainerNoLongi`, `LongiSegTrainer`, ~246.5 MB each). Strictly newer than the Aug 2025 baseline, but only for the **2D SAX** model, not the 3D reconstruction nets.

**Extra loose checkpoints**: `ukbb_sax_longi_compatible.pth`/`_v2.pth`, `ukbb_lax2ch_longi_compatible.pth`, `ukbb_lax4ch_longi_compatible.pth` (all Apr 14 2026) — filenames suggest UKBB-baseline weights reformatted for LongiSeg-trainer compatibility, not independently trained models.

**Checkpoint dict keys**: no `torch.save({...})` call exists anywhere in this tree (searched all 88 `.py` files). Only the read side is visible: `checkpoint['model_state_dict']` (`3D_segmentation/whole_heart_3d/utils.py:521-522`, identically `3D_segmentation/LA_3d_seg/utils.py:308-309`).

## Q7 — Preprocessing contract

**Whole-heart 3D net**: `vol_grid_gen()` (`utils.py:265-381`) builds a **160x160x160** sparse grid by reprojecting SAX/2CH/4CH label point-clouds onto axes derived from LAX plane normals, 10% margin (`:307`). Before inference: `np.transpose(vol_sp, [1, 0, 2])` (`:541`) then `* 30` (`:542`). After: `np.transpose(lab, [1, 0, 2])` (`:549`).

**LA 3D net**: `image_load_nifti_ori()` (`utils.py:145-167`) builds a **128x128x128** grid centred at `[64,64,64]` (`:157`) via `NearestNDInterpolator`. Before inference: `np.flip(..., axis=1)` (`:316`), remap, `* 50` (`:323`). After: `np.flip(np.argmax(pred, axis=1).squeeze(), axis=1)` (`:327`).

**4D CINE / ED-ES extraction**: lives in `cme-dt-devran-main/src/utils/pick_ed_es.py` / `pick_ed_es_accelerated.py` and `src/segmentation/split_merge_4d.py` — not part of the core 4-step pipeline.

## Q8 — What is new here

- **LongiSeg longitudinal retraining pipeline** — depends on external `LongiSeg` package (`git clone https://github.com/MIC-DKFZ/LongiSeg.git`, `prepare_longitudinal_datasets.py:421`).
- **BiV-ME mesh generation pipeline** — `cme-dt-devran-main/src/volumetric/meshtool_func.py:20-27` shells out to an external `meshtool` binary (CARP/openCARP `carp_txt` format) via `os.system(cmd)`. Orchestrated by `mesh_pipeline_scripts/` and `VU_mesh_pipeline*`.
- **QC tooling** — `qc_whole_heart_3d.py` (9-panel PDF, heuristic flags e.g. `n_vox.get(2,0) < 5000` at `:168`), `mesh_pipeline_scripts/qc_check_subject.py` (>=5 good SAX slices required, `:48`), `VU_mesh_pipeline_retrained_only/check_and_copy_final_cases.py` (file-presence + voxel-count gate, `:23-38,78-81`).
- **Volume/EF analysis** — `compute_all_volumes.py`, `compute_mesh_volumes.py`, `clean_ef_analysis.py`, `plot_volume_analysis.py`.
- **Manual-correction workflow** — `check_update/` + `manual_correction_guide.csv`.
- **VU clinical CRT cohort processing** — result dirs `VU_3d_segmentation/`, `VU_corrected/`, `VU_mesh_results_final/` (109 entries), `VU_mesh_pipeline_retrained_only/crt_response_analysis.csv`.
- **`create_clean_whole_heart_repo.sh`** — packaging/export script producing the sibling tree (see Q9).

## Q9 — Provenance

No `.git` directory exists anywhere (searched at multiple depths and recursively for nested `.git`). No `__version__`/`CHANGELOG` files found.

- Original weights cluster: `2D_segmentation/model_weights/{SAX,LAX}/Dataset31x/28x/*` and both 3D reconstruction checkpoints — mtime **Aug 12 2025**.
- `3D_segmentation/whole_heart_3d/utils.py`: `"""Created on Thu Sep 26 15:56:47 2024 / @author: aq22"""` (`:3,5`).
- `3D_segmentation/LA_3d_seg/LA_whole_heart_main.py`: `"""Created on Thu Dec 5 17:53:29 2024 / @author: Abdul Qayyum"""` (`:3,5`).
- `predictions_LAX_2ch.py`: `"""Created on Tue Jan 28 17:10:31 2025 / @author: Abdul Qayyum"""` (`:3-4`).
- Training logs internal timestamps `2026-04-29`; most `VU_*` results and `model_weights_longi/` dated Apr-May 2026.
- `create_clean_whole_heart_repo.sh` — mtime Aug 5 2026, the most recent file mtime at ROOT; the ROOT directory itself is also mtime Aug 5 2026.
- All "@author" tags name `Abdul Qayyum` / `aq22`.

Per instructions, no comparison to the sibling tree was performed — only the one-directional evidence that this tree is the export source (`create_clean_whole_heart_repo.sh:4-5`).

## Q10 — Hardware assumptions

- No `DataParallel`/`DistributedDataParallel` anywhere (zero hits, searched explicitly).
- `CUDA_VISIBLE_DEVICES`, always single-GPU: `prepare_longitudinal_datasets.py:413` (`=0`), `:431` (`=1`); `run_inference_vu_test.py:151`; `VU_mesh_pipeline_retrained_only/find_es_from_frames.sh:6` (`="1"`); `run_nnunet_es_missing.sh:9` (`=""`); instructions file uses `=1` for the 3D steps.
- CUDA/PyTorch pin: `whole_heart_pipline_instructions.txt` — `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y`, plus `numpy==1.24.4` compatibility pin.
- `cme-dt-devran-main/requirements.txt:1` pins `torch==2.8` — a **different, newer** pin than the top-level instructions; this is the only requirements/environment file anywhere in the tree.
- Fixed footprints: 160^3 (whole-heart) / 128^3 (LA) hard-coded patch grids. Production nnUNet plans: `Dataset314_SAXUKBB` 3d_fullres `batch_size: 9`, `patch_size: [10, 224, 192]`; `Dataset282_LAX2CH` 2d `batch_size: 47`; `Dataset283_LAX4CH` 2d `batch_size: 49`. LongiSAX training used `batch_size: 2`, `patch_size: [224, 160, 64]`.
- `torch.cuda.empty_cache()` called once per whole-heart inference (`utils.py:547`).

## Q11 — Validation and promotion gate

No hard-coded Dice/HD95 threshold gates the two 3D reconstruction nets from "trained" to "deployed" — none found.

What exists:
- nnUNet's own best-checkpoint tracking (`checkpoint_best.pth` alongside `checkpoint_final.pth` in every results folder; EMA-pseudo-Dice logic visible in training logs), but no script here reads and enforces a minimum from it.
- `evaluate_longiseg_vs_baseline.py` — genuine held-out comparison (Dice, HD95, volume-consistency, temporal-Dice-consistency, `:34-76`) plus Wilcoxon signed-rank test to declare a per-structure winner (`:230-241`) — decision support, not an automated blocking gate.
- Case-level QC gates (not model-level): `qc_check_subject.py:48` requires >=5 good SAX slices; `check_and_copy_final_cases.py:23-38,78-81` requires all 6 files present, >=100/>=10 voxel minimums, full {1,2,3} SAX label set, before promoting to `final_cases/`; `qc_whole_heart_3d.py:164-169` flags (does not block) missing-label or <5000-voxel subjects.

---

## Summary

**(a) LA label-mapping step**: **Yes**, present at `2D_segmentation/codes/LA_mapping_code/LA_2CH_label_mapping.py:29` and `LA_4CH_label_mapping.py:30`. Dictionaries match the fingerprint **exactly**: `{1: 2, 2: 0, 3: 1}` and `{1: 2, 2: 0, 3: 4, 4: 1, 5: 3}` — not revised, digit for digit identical.

**(b) `* 30` / `* 50` scale factors**: **Yes**, both present and unchanged — `utils.py:542` (`* 30`, whole-heart) and `utils.py:323` (`* 50`, LA), both literal constants.

**(c) Weights newer than epoch150/epoch_219**: **No**, for the two 3D reconstruction nets — the loaded checkpoints are still exactly `epoch_150_params.pth` and `epoch_219_params.pth` (both Aug 12 2025). One anomaly worth flagging: `LA_3d_seg/model_due/exp_plus_/epoch_219_params.pth` (77,008,873 bytes) is an unreferenced second file, ~3x the size of the one actually used — not clearly "newer," but unexplained and worth a closer look. Separately, the **2D SAX nnUNet model** does have newer retrained weights (`model_weights_longi/`, Apr 2026) — but that's a different model than the 3D reconstruction nets CardioForm wraps.
