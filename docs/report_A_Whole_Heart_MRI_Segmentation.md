# Audit: Whole_Heart_MRI_Segmentation (tree A)

ROOT = `/home/jsolisle/installs/from_server/Whole_Heart_MRI_Segmentation`. Read-only audit; no code executed, no checkpoints loaded, no hashes computed. Tree confirmed at exactly 42 files / 9 Python files.

> Persisted by the coordinator. The subagent harness blocked the agent's own Write call; content is verbatim from the agent's report, with HTML entities decoded.

## Top 5 findings

1. This tree **does contain** the LA label-mapping step CardioForm is missing, and both dictionaries match the fingerprint digit for digit: `{1: 2, 2: 0, 3: 1}` (`2D_segmentation/codes/LA_mapping_code/LA_2CH_label_mapping.py:29`) and `{1: 2, 2: 0, 3: 4, 4: 1, 5: 3}` (`2D_segmentation/codes/LA_mapping_code/LA_4CH_label_mapping.py:30`).
2. Both 3D U-Net classes are named `UNet3d` (not `ReconstructUNet3D` / `ReconstructLaUNet3d` as CardioForm names them) but are byte-identical to each other in body, and structurally match the fingerprints at every checked point: 3x `MaxPool3d(2)`, `contracting_block`/`expansive_block`/`final_block`/`crop_and_concat`, constructed `in_channel=1, out_channel=9` for whole-heart (`3D_segmentation/whole_heart_3d/utils.py:520`) and `in_channel=1, out_channel=6` for LA (`3D_segmentation/LA_3d_seg/utils.py:304`).
3. The `* 30` (`3D_segmentation/whole_heart_3d/utils.py:542`) and `* 50` (`3D_segmentation/LA_3d_seg/utils.py:324`) scale-factor literals are present and unchanged, as are the `[1, 0, 2]` transpose and the pre/post `np.flip(..., axis=1)` calls.
4. There is **no training code anywhere** in this tree — no loss function, optimizer, LR schedule, `Dataset`/`DataLoader` class, or `nnUNetv2_train` invocation. All hyperparameters recoverable here come from nnUNet-generated `debug.json`/`plans.json` metadata sidecars, not from scripts in this repo.
5. Four checkpoint files — `ukbb_sax_longi_compatible.pth`, `ukbb_sax_longi_compatible_v2.pth`, `ukbb_lax2ch_longi_compatible.pth`, `ukbb_lax4ch_longi_compatible.pth` — are dated 2026-04-14, roughly 8 months newer than the `epoch150`/`epoch_219` baseline and the nnUNet `checkpoint_final.pth` files (all 2025-08-12), and are **not referenced by any script** in this tree (`grep` for their names/`"longi"` in `*.py` returns nothing).

---

## Q1 — Inventory

The pipeline is a strict 6-step, single-direction chain, run once from repo root per `test_*.log` timestamps (all 2026-08-05, in this exact order):

1. `2D_segmentation/codes/SAX_code/predictions_SAX.py` — nnUNetv2 2D->3D-fullres SAX segmentation (08:49:45)
2. `2D_segmentation/codes/LAX_code/predictions_LAX_2ch.py` — nnUNetv2 2D 2CH segmentation (08:51:09)
3. `2D_segmentation/codes/LAX_code/predictions_LAX_4ch.py` — nnUNetv2 2D 4CH segmentation (08:52:04)
4. `3D_segmentation/whole_heart_3d/whs_4ch_main.py` — custom 3D whole-heart reconstruction, consumes raw (unmapped) SAX/2CH/4CH outputs directly (08:57:07)
5. `2D_segmentation/codes/LA_mapping_code/LA_4CH_label_mapping.py` then `LA_2CH_label_mapping.py` — relabels the 4CH/2CH nnUNet outputs into LA-net input space (08:58:15, 08:58:39)
6. `3D_segmentation/LA_3d_seg/LA_whole_heart_main.py` — custom 3D LA reconstruction, consumes the Step-5 relabeled files (09:11:27)

Both `README_whs.md` (steps 1-6, `README_whs.md:129-254`) and `ORIGINAL_PIPELINE_INSTRUCTIONS.txt` (4 grouped steps, `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:73-120`) describe this same chain, and the CLI flags they document match the `argparse` definitions in the code exactly (e.g. `predictions_SAX.py:27-29`, `whs_4ch_main.py:186-191`, `LA_2CH_label_mapping.py:16-17`, `LA_whole_heart_main.py:82-84`). The `test_*.log` files confirm the code runs end-to-end on the documented `MASTER001` example and finishes ("Prediction process finished!", `test_sax.log`; "Processing completed successfully.", `test_la_3d.log`).

Discrepancies between the two instruction documents (not between docs and code):
- `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:9` specifies `python=3.9`; `README_whs.md:53` specifies `python=3.10`.
- `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:16-17` says `pip install numpy==1.24.4` "if issue"; the shipped `requirements.txt:51` pins a numpy 2.2.6 wheel — the two documents disagree with each other about which numpy version is expected.
- `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:5` names the original host path `/data/Abdul/Whole_heart_pipline/`; the actual runtime paths recorded in the test logs are `/data/Abdul/Whole_Heart_MRI_Segmentation/...` (e.g. `test_la_3d.log:1`) — i.e. the tree was renamed/copied between when the instructions file was written and when the test logs were produced.

`whs_4ch_main.py` (lines 1-255) contains two large commented-out earlier revisions of itself above the live code, showing the script evolved from hardcoded `/data/Abdul/Whole_heart_pipline/...` paths, to an `argparse` version without `--model_dir`, to the current version with `--model_dir` added (comment `# new parameter`, `whs_4ch_main.py:253`).

## Q2 — Model architectures

Exactly two 3D U-Net class definitions exist in this tree, both named `UNet3d(nn.Module)`, and a byte-for-byte diff of their class bodies is empty (identical code in two files):

- `3D_segmentation/whole_heart_3d/utils.py:12-98`
- `3D_segmentation/LA_3d_seg/utils.py:211-297`

Both define `contracting_block` (`:13-22`/`:212-221`), `expansive_block` (`:24-35`/`:223-234`), `final_block` (`:37-48`/`:236-247`), and `crop_and_concat` (`:75-79`/`:274-278`), with three `torch.nn.MaxPool3d(kernel_size=2)` (`:54,56,58`/`:253,255,257`).

Constructed instances:
- Whole-heart: `UNet3d(in_channel=1, out_channel=9)`, `whole_heart_3d/utils.py:520`.
- LA: `UNet3d(in_channel=1, out_channel=6)`, `LA_3d_seg/utils.py:304`.

Comparison to fingerprint: **structurally identical, class name drifted**. Every structural element named in the fingerprint is present unchanged (block names, pool count, `crop_and_concat`, in/out channel counts). The only difference is the class name — CardioForm's `ReconstructUNet3D`/`ReconstructLaUNet3d` vs this tree's `UNet3d` (reused in two files). Since `load_state_dict` matches on parameter-name/shape, not class name, this rename does not by itself break checkpoint loading — the layer names are identical (`conv_encode1`, `conv_maxpool1`, `bottleneck`, `conv_decode3`, `conv_decode2`, `final_layer`, `utils.py:53-73`/`:252-272`).

The nnUNetv2 2D/3D models (SAX, LAX 2CH, LAX 4CH) use the library's own `PlainConvUNet` architecture (`"UNet_class_name": "PlainConvUNet"`, e.g. `2D_segmentation/model_weights/SAX/Dataset314_SAXUKBB/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json:170`), which is not defined anywhere in this repo — it lives in the external `nnunetv2` package.

## Q3 — Normalisation constants

Both present, unchanged, literal (not computed per case):
- Whole-heart: `test_x = img_i_[np.newaxis, np.newaxis, ...] * 30`, `3D_segmentation/whole_heart_3d/utils.py:542`.
- LA: `test_x[0, 0, ...] = vol_in_ * 50`, `3D_segmentation/LA_3d_seg/utils.py:324`.

## Q4 — Label spaces (priority)

**2D SAX** emits per `2D_segmentation/model_weights/SAX/Dataset314_SAXUKBB/.../dataset.json:5-10`: `background=0, LV=1, Myo=2, RV=3`. Runtime confirms: `test_sax.log` "labels: [0 1 2 3]".

**2D LAX 2CH** emits per `.../Dataset282_LAX2CH/.../dataset.json:5-10`: `background=0, LV=1, Myo=2, LA=3`. Runtime: `test_2ch.log` "pred_array_labels_before: [0 1 2 3]".

**2D LAX 4CH** emits per `.../Dataset283_LAX4CH/.../dataset.json:5-12`: `background=0, LV=1, Myo=2, RV=3, LA=4, RA=5`. Runtime: `test_4ch.log` "pred_array_labels_before: [0 1 2 3 4 5]".

**Relabel step (feeds the LA 3D net only, not the whole-heart 3D net):**
- `LA_2CH_label_mapping.py:29`: `mapping = {1: 2, 2: 0, 3: 1}  # LV -> 2, Myo -> 0, LA -> 1` -> output space `0=Myo(+bg), 1=LA, 2=LV`.
- `LA_4CH_label_mapping.py:30`: `mapping = {1: 2, 2: 0, 3: 4, 4: 1, 5: 3}` (no comment in this file) -> output space `0=Myo(+bg), 1=LA, 2=LV, 3=RA, 4=RV`.

Both dictionaries **match the fingerprint exactly**, and both mapping outputs put `1=LA, 2=LV` — exactly what the LA 3D net expects next.

**Whole-heart 3D net** is fed directly from raw (un-mapped) SAX/2CH/4CH outputs via lookup tables in `vol_grid_gen` (`whole_heart_3d/utils.py`): `px2vx_2ch = [0, 1, 2, 5]` (line 330), `px2vx_4ch = [0, 1, 2, 3, 5, 6]` (line 348), `px2vx_sax = [0, 1, 2, 3]` (line 370). This yields a combined 9-class space for `UNet3d(out_channel=9)`: `0=background, 1=LV, 2=Myo, 3=RV, 5=LA, 6=RA`. Indices 4, 7, 8 are never assigned by any table — unlabelled in this codebase (not proven "unused"). Confirmed in reverse by the back-projection tables `px2vx_2ch = [0,1,2,0,0,3,0,0,0]` (line 448) and `px2vx_4ch = [0,1,2,3,0,4,5,0,0]` (line 467).

**LA 3D net input remap** (`LA_3d_seg/utils.py:317-324`): takes the Step-5 mapped files (`1=LA, 2=LV`), then `vol_in_la[vol_in==1]=1; vol_in_lv[vol_in==2]=5; vol_in_ = vol_in_la+vol_in_lv` — i.e. `1=LA` unchanged, `2=LV`->`5` on the network input. Matches the fingerprint exactly.

**LA 3D net output**: `prd_tyi = np.flip(np.argmax(pred, axis=1).squeeze(), axis=1)` (`utils.py:328`) over 6 channels. No code states what integers 0-5 mean in the *output* — only the *input* encoding (`1=LA, 5=LV`) is proven; reporting the output as unlabelled rather than assuming it mirrors the input.

## Q5 — Training code

**None found.** Grep across all 9 `.py` files for training loops, `Dataset`/`DataLoader` classes, losses, optimizers, LR schedulers, `nnUNetv2_train`, `nnUNetv2_plan_and_preprocess` returned zero matches, and no `torch.save` call exists anywhere (checked separately).

Hyperparameters recoverable only from nnUNet-generated `debug.json`/`plans.json` metadata (proves a training run happened elsewhere; no training script exists in this tree):
- SAX (`.../Dataset314_SAXUKBB/.../fold_all/debug.json`): `:38 "num_epochs": "1000"`, `:26 "initial_lr": "0.0001"`, `:53 "weight_decay": "3e-05"`, `:45 "oversample_foreground_percent": "0.33"`, `:20 "fold": "all"`, `:42` optimizer `SGD(momentum: 0.99, nesterov: True, weight_decay: 3e-05, lr: 0.0001, initial_lr: 0.00611)`.
- LAX2CH: same pattern, `initial_lr=0.0001`.
- LAX4CH: `initial_lr=0.001` — differs from SAX/LAX2CH, a real cross-model hyperparameter difference.
- Batch/patch size (`plans.json`): SAX 3d_fullres `batch_size=9, patch_size=[10,224,192]` (`plans.json:148-152`); LAX2CH 2d `batch_size=47, patch_size=[192,224]`; LAX4CH 2d `batch_size=49, patch_size=[224,192]`.
- Training working dir embedded in metadata: `"preprocessed_dataset_folder_base": "/home/aqayyum/xLSTM-UNet-PyTorch/data/nnUNet_preprocessed/Dataset314_SAXUKBB"` (`debug.json:48`), with training-log filenames `training_log_2025_1_28_07_00_50.txt` (SAX), `training_log_2025_1_29_16_58_01.txt` (LAX2CH), `training_log_2025_1_29_17_03_12.txt` (LAX4CH) — the log files themselves are absent from this tree, only their recorded paths remain.

No `nnUNetv2_plan_and_preprocess`/`nnUNetv2_train` shell invocation exists anywhere; only dataset IDs (`Dataset314_SAXUKBB`, `Dataset282_LAX2CH`, `Dataset283_LAX4CH`) and trainer/plan identifiers (`nnUNetTrainer__nnUNetPlans__{2d,3d_fullres}`) are recoverable, from folder names and `predict_from_raw_data` calls (`predictions_SAX.py:55`). No training code for the two custom `UNet3d` nets either.

## Q6 — Checkpoints and weights

(path / size bytes / mtime / filename meaning; no hashing performed)

| Path | Size | mtime | Meaning |
|---|---|---|---|
| `2D_segmentation/model_weights/SAX/Dataset314_SAXUKBB/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth` | 240701485 | 2025-08-12 09:42:22 | nnUNetv2 final ckpt, SAX 3d_fullres fold "all" |
| `.../LAX/Dataset282_LAX2CH/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth` | 165563053 | 2025-08-12 09:41:05 | nnUNetv2 final ckpt, LAX2CH 2d |
| `.../LAX/Dataset283_LAX4CH/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth` | 165705709 | 2025-08-12 09:39:47 | nnUNetv2 final ckpt, LAX4CH 2d |
| `3D_segmentation/whole_heart_3d/model/epoch_150_params.pth` | 77039978 | 2025-08-12 10:04:13 | custom whole-heart net, epoch 150 |
| `3D_segmentation/LA_3d_seg/model_due/exp_plus/epoch_219_params.pth` | 25687017 | 2025-08-12 10:04:26 | custom LA net, epoch 219 — **the file the code actually loads** (`LA_3d_seg/utils.py:307`) |
| `3D_segmentation/LA_3d_seg/model_due/exp_plus_/epoch_219_params.pth` | 77008873 | 2025-08-12 10:04:23 | same claimed epoch, **not referenced by any code** (`grep -rn "exp_plus_" --include="*.py" .` -> empty) |
| `2D_segmentation/model_weights/SAX/ukbb_sax_longi_compatible.pth` | 240681434 | 2026-04-14 12:44:55 | not referenced by any script |
| `2D_segmentation/model_weights/SAX/ukbb_sax_longi_compatible_v2.pth` | 240690849 | 2026-04-14 13:43:32 | not referenced by any script |
| `2D_segmentation/model_weights/LAX/ukbb_lax2ch_longi_compatible.pth` | 165552417 | 2026-04-14 13:44:27 | not referenced by any script |
| `2D_segmentation/model_weights/LAX/ukbb_lax4ch_longi_compatible.pth` | 165695073 | 2026-04-14 13:44:29 | not referenced by any script |

Flags:
- `exp_plus_/epoch_219_params.pth` (77,008,873 bytes) is suspiciously close in size to the whole-heart net's `epoch_150_params.pth` (77,039,978 bytes), and 3x the size of the `exp_plus/epoch_219_params.pth` (25,687,017 bytes) actually loaded by code — an unreferenced, unexplained orphan artifact (not investigated further; opening it is out of scope).
- All four `*_longi_compatible*.pth` files are newer than `epoch150`/`epoch_219` (2026-04-14 vs 2025-08-12) and unwired into any script.

`torch.save(...)` does not appear anywhere in this tree, so no checkpoint-writing call is visible to quote for dict-key provenance. The only evidence of the `model_state_dict` key convention is the read side: `checkpoint = torch.load(model_file); unet.load_state_dict(checkpoint['model_state_dict'])` (`whole_heart_3d/utils.py:521-522`, identically `LA_3d_seg/utils.py:309-310`).

## Q7 — Preprocessing contract

**4D CINE frame extraction**: not present in this tree. Input data already arrives as per-frame NIfTI files (`README_whs.md:90-107`); no code here extracts frames from a 4D CINE volume.

**Resampling**: handled internally by the external nnUNetv2 `nnUNetPredictor`/`DefaultPreprocessor` for the SAX/LAX models; no custom resampling code in this tree.

**Orientation / flips / transposes** (only in the two custom 3D nets):
- `whole_heart_3d/utils.py:541`: `img_i_ = np.transpose(vol_sp, [1, 0, 2])` before the net's forward pass.
- `whole_heart_3d/utils.py:549`: `lab_ = np.transpose(lab, [1, 0, 2])` on the argmax output.
- `LA_3d_seg/utils.py:317`: `vol_in = np.flip(data_in.get_fdata(), axis=1)` before the net's forward pass.
- `LA_3d_seg/utils.py:328`: `prd_tyi = np.flip(np.argmax(pred, axis=1).squeeze(), axis=1)` on the argmax output.
- Numerous other `np.transpose` calls in `whole_heart_3d/utils.py` (lines 254-260, 313-315, 328-329, 346-347, 368-369, 433-435, 445-446, 464-465, 488-489) implement SAX/2CH/4CH plane-geometry construction, not network-input axis reordering.

**Cropping**: `LA_3d_seg/utils.py:171-180` (`points_rm`) drops points outside `[-64, 63]` on each axis before back-projection. `whole_heart_3d/utils.py` has no explicit crop function; out-of-grid voxels are silently dropped in a `try/except` printing `"Out of box voxels."` (e.g. lines 331-339, 349-357, 371-380).

## Q8 — What is new here

- **The LA relabel step itself** — the step CardioForm is missing entirely (see Q4).
- **Back-projection** of 3D predictions onto original 2D planes: `vol_grid_bp` (`whole_heart_3d/utils.py:384-500`, `whs_4ch_main.py:201-203`) and `image_save_nifti_ori` (`LA_3d_seg/utils.py:183-207`, `LA_whole_heart_main.py:68-74`) — QC-adjacent visualization with no CardioForm counterpart.
- **Sparse-volume intermediate output** (`vol_sp`) saved by both 3D pipelines.
- **Newer, unwired candidate checkpoints** (`*_longi_compatible*.pth`) suggesting an in-progress retraining effort not reflected in any wired model in this tree, and not matching CardioForm's checkpoint filenames.

Explicitly checked, zero hits: mesh generation (`grep -rni "mesh\|vtk\|marching_cubes\|trimesh\|pyvista"` — only unrelated `np.meshgrid` matches), named QC tooling (`grep -rni "\bqc\b\|quality.control"` — zero hits, though `.gitignore:49` has a `qc_*` ignore pattern with no corresponding script), evaluation-metric scripts (see Q11), longitudinal-analysis code (`"longi"` matches only checkpoint filenames, never inside `.py`/`.md`/`.txt`/`.json`).

## Q9 — Provenance

No `.git` directory anywhere (only `.gitignore` exists) — a plain file copy, not a git checkout. No `__version__`, `VERSION` file, or `CHANGELOG` anywhere.

Timeline reconstructed from docstrings and filesystem timestamps:
- Code docstrings, mostly `@author: Abdul Qayyum` (one `@author: aq22`): `predictions_SAX.py:3,5` "Created on Thu Dec 5 17:53:29 2024"; `predictions_LAX_2ch.py:3-4` / `predictions_LAX_4ch.py:3-4` "Created on Tue Jan 28 17:10:31 2025"; `whs_4ch_main.py:3,5` "Created on Thu Sep 26 15:56:47 2024" / `@author: aq22`; `LA_whole_heart_main.py:3,5` "Created on Thu Dec 5 17:53:29 2024".
- Training metadata embeds username `aqayyum` (e.g. `/home/aqayyum/xLSTM-UNet-PyTorch/data/nnUNet_results/...`, `debug.json:43-44`), consistent with the docstring author tag; working-dir name `xLSTM-UNet-PyTorch` implies training happened from a repo of that name, though the saved architecture is plain `PlainConvUNet` (`plans.json:170`) — no proof xLSTM was used in these specific checkpoints.
- Training-run timestamps embedded in `debug.json`: SAX `training_log_2025_1_28_07_00_50.txt`, LAX2CH `training_log_2025_1_29_16_58_01.txt`, LAX4CH `training_log_2025_1_29_17_03_12.txt` (log files themselves absent).
- `checkpoint_final.pth` x3, `epoch_150_params.pth`, both `epoch_219_params.pth`: mtime 2025-08-12 (all within a ~25-minute window, 09:39-10:04).
- `ORIGINAL_PIPELINE_INSTRUCTIONS.txt`: mtime 2025-10-08 08:32; references original host path `/data/Abdul/Whole_heart_pipline/` (line 5).
- Four `*_longi_compatible*.pth`: mtime 2026-04-14 (12:44-13:44).
- `.gitignore` mtime 2026-08-05 08:25; `requirements.txt` mtime 2026-08-05 09:15; `README_whs.md` mtime 2026-08-05 09:22; all six `test_*.log` mtime 2026-08-05 (08:49-09:11), embedding path `/data/Abdul/Whole_Heart_MRI_Segmentation/...` (`test_la_3d.log:1`) — by log time the tree had been renamed from `Whole_heart_pipline` to `Whole_Heart_MRI_Segmentation`.

No comparison made to the sibling tree — not read, per instructions.

## Q10 — Hardware assumptions

No multi-GPU code: `grep -rn "DataParallel\|CUDA_VISIBLE_DEVICES" --include="*.py" .` returns zero matches (the `CUDA_VISIBLE_DEVICES=0/1` pins appear only in shell examples in `README_whs.md:132` and `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:95,117`, not in code). Single-GPU device placement only: `torch.device('cuda')` (`predictions_SAX.py:47`, `predictions_LAX_2ch.py:55`, `predictions_LAX_4ch.py:54`, `whole_heart_3d/utils.py:518`), `torch.device("cuda:0")` (`LA_3d_seg/utils.py:302`).

Version pins:
- `requirements.txt:74-76`: `torch==2.5.1`, `torchaudio==2.5.1`, `torchvision==0.20.1`.
- `requirements.txt:49`: `nnunetv2==2.8.1`.
- `requirements.txt:51`: numpy pinned via wheel to `2.2.6`.
- `README_whs.md:60-61`: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y`.
- `README_whs.md:283-288`: `Ubuntu 22.04`, `Python 3.10`, `CUDA 11.8`, `PyTorch 2.x`, `nnUNet v2`.
- `ORIGINAL_PIPELINE_INSTRUCTIONS.txt:9,12`: `python=3.9`; `pytorch-cuda=11.8`.

Memory-footprint constants: SAX 3d_fullres `batch_size=9, patch_size=[10,224,192]` (`plans.json:148-152`); LAX2CH 2d `batch_size=47, patch_size=[192,224]`; LAX4CH 2d `batch_size=49, patch_size=[224,192]`. Fixed volume sizes in the custom 3D nets: whole-heart 160^3 (`whole_heart_3d/utils.py:309-311,317`), LA 128^3 (`LA_3d_seg/utils.py:160,190`).

## Q11 — Validation and promotion gate

**No validation/promotion gate exists in this tree.** `grep -in "dice\|hausdorff\|iou\|metric\|score"` across every log file matched only an unrelated substring inside prose and boilerplate PyTorch warning text — no genuine metric output anywhere. No script computes Dice, Hausdorff, or IoU; no "best checkpoint" logic (only `checkpoint_final.pth` ships, no `checkpoint_best.pth`); no threshold/acceptance code exists. The only "validation" surface is nnUNet's own internal loop implied by `debug.json`'s `num_val_iterations_per_epoch: 50` — default `nnunetv2` package behavior, not code or a gate present in this repository, and its results are not recorded anywhere in this tree.

---

## Summary

(a) Does this tree contain the LA label-mapping dictionaries, and do they match exactly? **Yes** — `LA_2CH_label_mapping.py:29` `{1:2, 2:0, 3:1}` and `LA_4CH_label_mapping.py:30` `{1:2, 2:0, 3:4, 4:1, 5:3}`, both exact matches to the fingerprint.

(b) Are the `* 30` and `* 50` scale factors present and unchanged? **Yes** — `whole_heart_3d/utils.py:542` (`* 30`) and `LA_3d_seg/utils.py:324` (`* 50`), both literal constants, no per-case computation.

(c) Is there any training code at all? **No** — zero training loops, losses, optimizers, LR schedules, `Dataset`/`DataLoader` classes, or `nnUNetv2_train`/`nnUNetv2_plan_and_preprocess` invocations anywhere in the 9 Python files. All hyperparameter evidence comes from nnUNet-generated `debug.json`/`plans.json` metadata sidecars, not from code present in this tree.
