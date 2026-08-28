# CardioForm: 2D Segmentation & 3D Reconstruction
A pipeline for whole-heart segmentation and 3D reconstruction from CINE MRI.

## 🚀 Quick Start

This project is distributed as a Docker container. The only file you need to download from this repository is the `run-docker.sh` wrapper script.

1.  **Download the wrapper script:**
    ```bash
    curl -O https://raw.githubusercontent.com/<your_org>/cardio_form/main/run-docker.sh
    chmod +x run-docker.sh
    ```

2.  **Pull the Docker image:**
    ```bash
    # For CPU execution (recommended default)
    docker pull cemrg/cardio-form:latest
    ```
    *For GPU execution, see advanced usage below.*

3.  **Run the pipeline:**
    The first argument to the script is your data directory. All subsequent paths inside the command must be relative to `/data`.
    ```bash
    # Example: Run the full pipeline on a subject directory
    ./run-docker.sh /path/to/your/subject01 full \
        --input-dir /data \
        --output-dir /data/output \
        --output-prefix "subject01"
    ```
---

# For developers and to run locally 
## Installation

This guide provides instructions for setting up the project in a clean virtual environment using either Conda or Python's built-in `venv`.

### Prerequisites
*   Python 3.9
*   Git
*   (Optional but Recommended) An NVIDIA GPU with CUDA 11.8 drivers installed.

### Step 1: Clone the Repository

First, clone the project from GitHub to your local machine.

```bash
git clone https://github.com/OpenHeartDevelopers/cardio-form.git
cd cardio-form
```

### Step 2: Create and Activate a Virtual Environment

#### Using `conda`

This is the recommended approach if you use the Anaconda distribution.

```bash
# Create a new conda environment with Python 3.11
conda env create -f environment.yaml 

# Activate the environment
conda activate cardioform
```
### Step 3: Install the package with `setup.sh`

Run the script (inside the activated env):
```shell
cd cardio-form
chmod +x setup.sh # if necessary 

./setup.sh
```

This installs the shared `pycemrg` core and CardioForm itself as editable packages
(`pip install -e . --no-deps`), making the `cardioform` command available. It also
(re)generates `CARDIOFORM_ENV_SETUP` (not tracked by git), which simply runs
`conda activate cardioform` for new shells. Under the `src/` layout there is no
`PYTHONPATH` to set.

## Usage

Once installed, use the single `cardioform` command to run the different parts of the
`CardioForm` pipeline. Run `cardioform --help` (or `cardioform <mode> --help`) for all options.

### 1. `cardioform segment`: Segmenting a 2D MRI

This script runs the `nnunetv2` model to segment a single 2D CINE MRI series (SAX, 2CH, or 4CH).

#### Arguments:
*   `--input`: Path to the input NIfTI file.
*   `--output-dir`: Directory where the output file will be saved.
*   `--output-prefix`: A name to prefix the output filename (e.g., `subject-001`).
*   `--view-type`: The type of MRI view. Must be one of `sax`, `lax_2ch`, `lax_4ch`.
*   `--device`: The device to run on (`auto`, `cpu`, `cuda`). Defaults to `auto`.

#### Example:
```bash
cardioform segment \
    --input /path/to/data/CINE_image_SAX_001.nii.gz \
    --output-dir /path/to/outputs/segmentations \
    --output-prefix "subject-001" \
    --view-type sax
```
**Output**: This will create a file named `subject-001_2D_seg_sax.nii.gz` inside the `/path/to/outputs/segmentations/` directory.

### 2. `cardioform reconstruct`: Creating a 3D Model from 2D Segmentations
This script takes the three 2D segmentations (SAX, 2CH, 4CH) and runs the 3D U-Net to reconstruct the final 
whole-heart segmentation. 
This is the perfect tool for re-running the 3D step after manually correcting a 2D segmentation.

#### Arguments:
* `--sax-file, --ch2-file, --ch4-file`: Paths to the three input 2D segmentation NIfTI files.
* `--output-dir`: Directory where the output files will be saved.
* `--output-prefix`: A name to prefix all output filenames.
* `--device`: The device to run on. Defaults to auto.
* `-qc, --quality-control`: Also write the sparse volume and back-projections. Off by default.

#### Example:
```bash
cardioform reconstruct \
    --sax-file /path/to/outputs/segmentations/subject-001_2D_seg_sax.nii.gz \
    --ch2-file /path/to/outputs/segmentations/subject-001_2D_seg_lax_2ch.nii.gz \
    --ch4-file /path/to/outputs/segmentations/subject-001_2D_seg_lax_4ch.nii.gz \
    --output-dir /path/to/outputs/reconstructions \
    --output-prefix "subject-001"
```

**Output:** `subject-001_whole_heart_segmentation.nii.gz` — and nothing else. Add `-qc` /
`--quality-control` to also write the diagnostic artefacts (`..._intermediate_sparse_volume.nii.gz`
and the three `..._intermediate_qc_*_backprojected.nii.gz` files).

### 3. `cardioform full_pipeline`: The End-to-End Solution
This is the main command that automates the entire process. 
It takes a directory of raw CINE images, runs the 2D segmentations for all views, and then automatically runs the 3D reconstruction.

#### Arguments:
* `--input-dir`: Path to a directory containing the raw CINE MRI files (e.g., ..._SAX.nii.gz, ..._CH2.nii.gz, etc.).
* `--output-dir`: Directory where all output files will be saved.
* `--output-prefix`: A name to prefix all output filenames. If not provided, it is automatically inferred from the name of the --input-dir.
* `--device`: The device to run on. Defaults to auto.
#### Example:
```bash
cardioform full_pipeline \
    --input-dir /path/to/data/subject-001/ \
    --output-dir /path/to/outputs/full_run/ \
    --output-prefix "subject-001_final"
```

**Output:** This will create a flat list of all files (intermediate segmentations and final reconstruction) in the 
`/path/to/outputs/full_run/` directory, all prefixed with `subject-001_final`.

### 4. `cardioform reconstruct_la`: Left-heart 3D reconstruction

Runs a separate 3D network over the two long-axis segmentations. Despite the name it
reconstructs the **left side** of the heart, not the atrium alone: it emits an LA body,
two pulmonary-vein regions, and the LV base.

The 2D LAX segmentations use a different label convention from this network, so the
inputs are remapped automatically before inference. Pass the raw output of
`cardioform segment`; the remapped files are written alongside as QC artefacts.

#### Arguments:
* `-ch2, --ch2-file` / `-ch4, --ch4-file`: The two LAX 2D segmentation NIfTI files.
* `-o, --output-dir`: Directory where the output files will be saved.
* `-p, --output-prefix`: A name to prefix all output filenames.
* `--device`: The device to run on. Defaults to `cpu`.
* `-qc, --quality-control`: Also write the sparse volume, both back-projections, and the
  remapped LAX inputs. Off by default; without it the remapped inputs go to a temporary
  directory and are discarded.

#### Example:
```bash
cardioform reconstruct_la \
    -ch2 /path/to/outputs/subject-001_2D_seg_lax_2ch.nii.gz \
    -ch4 /path/to/outputs/subject-001_2D_seg_lax_4ch.nii.gz \
    -o /path/to/outputs -p "subject-001"
```

**Output:** `subject-001_la_3d_segmentation.nii.gz`. With `-qc`, also the sparse volume,
the two back-projection QC files, and the two remapped LAX inputs.

### 5. `cardioform left_complete`: Enhancing the whole heart with the left-side model

Folds the `reconstruct_la` output into a whole-heart segmentation. The left-side volume
is resampled onto the whole-heart grid and written **only where the whole-heart map is
background**, so existing structure is never overwritten and the LA-LV connection is
preserved.

#### Arguments:
* `-la, --la-file`: An existing LA 3D segmentation. Overrides `-ch2`/`-ch4`.
* `-ch2, --ch2-file` / `-ch4, --ch4-file`: LAX segmentations, used to run the LA
  reconstruction first when `-la` is absent.
* `-whs, --whs-file`: The whole-heart segmentation to enhance.
* `-o, --output-dir` / `-p, --output-prefix`: Output location and prefix.
* `--include-la`, `--include-lv`, `--include-veins`: Restrict which structures are
  merged. No flag merges everything; flags given are combined.

#### Example:
```bash
# Skip the poorly-resolved vein classes
cardioform left_complete \
    -la  /path/to/outputs/subject-001_la_3d_segmentation.nii.gz \
    -whs /path/to/outputs/subject-001_whole_heart_segmentation.nii.gz \
    -o /path/to/outputs -p "subject-001" \
    --include-la --include-lv
```

**Output:** `subject-001_left_complete_segmentation.nii.gz`, on the same grid as the
input whole-heart file.

### 6. `cardioform labels`: Editing segmentation labels

Filter, merge, or remap labels in a segmentation NIfTI. Labels may be given by name,
group, or integer.

> **Every stage uses a different label space.** On the 2D segmentation outputs `1` is
> blood pool and `2` is myocardium — the *opposite* of the whole-heart output. Pass
> `--label-space` to name the space your input uses: `whole_heart` (default), `sax`,
> `lax_2ch`, `lax_4ch`, `sparse`, or `left`. The manifests live in
> `src/cardio_form/config_data/`.

```bash
# Keep only the ventricles group and the aorta
cardioform labels filter  -i seg.nii.gz -o filtered.nii.gz -l ventricles Ao

# Merge the ventricles into a single label value (1)
cardioform labels merge   -i seg.nii.gz -o merged.nii.gz   -l ventricles -v 1

# Remap individual labels with OLD:NEW pairs (names or integers)
cardioform labels relabel -i seg.nii.gz -o relabeled.nii.gz -m MYO_septum:LV_myo 7:0

# Operate on a 2D LAX 4CH segmentation, where LA is 4 (not 5)
cardioform labels filter -i seg_lax_4ch.nii.gz -o la_only.nii.gz -l LA --label-space lax_4ch
```