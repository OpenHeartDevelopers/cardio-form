# CardioForm

A pipeline for whole-heart segmentation and 3D reconstruction from CINE MRI.

---

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
### Step 3: Run the `setup.sh` script for easy environment activation 

Run the script 
```shell
cd cardio-form
chmod +x setup.sh # if necessary 

./setup.sh
```

This creates a file (not tracked by git) called `CARDIOFORM_ENV_SETUP`, which lets you quickly 
activate the environment and set the `PYTHONPATH` variable so you can run the different scripts. 

## Usage

Once the environment is activated, you can use the provided scripts to run the different parts of the `CardioForm` pipeline. All scripts are located in the `scripts/` directory.

### 1. `run_segmentation.py`: Segmenting a 2D MRI

This script runs the `nnunetv2` model to segment a single 2D CINE MRI series (SAX, 2CH, or 4CH).

#### Arguments:
*   `--input`: Path to the input NIfTI file.
*   `--output-dir`: Directory where the output file will be saved.
*   `--output-prefix`: A name to prefix the output filename (e.g., `subject-001`).
*   `--view-type`: The type of MRI view. Must be one of `sax`, `lax_2ch`, `lax_4ch`.
*   `--device`: The device to run on (`auto`, `cpu`, `cuda`). Defaults to `auto`.

#### Example:
```bash
python scripts/run_segmentation.py \
    --input /path/to/data/CINE_image_SAX_001.nii.gz \
    --output-dir /path/to/outputs/segmentations \
    --output-prefix "subject-001" \
    --view-type sax
```
**Output**: This will create a file named `subject-001_2D_seg_sax.nii.gz` inside the `/path/to/outputs/segmentations/` directory.

### 2. `run_reconstruction.py`: Creating a 3D Model from 2D Segmentations
This script takes the three 2D segmentations (SAX, 2CH, 4CH) and runs the 3D U-Net to reconstruct the final 
whole-heart segmentation. 
This is the perfect tool for re-running the 3D step after manually correcting a 2D segmentation.

#### Arguments:
* `--sax-file, --ch2-file, --ch4-file`: Paths to the three input 2D segmentation NIfTI files.
* `--output-dir`: Directory where the output files will be saved.
* `--output-prefix`: A name to prefix all output filenames.
* `--device`: The device to run on. Defaults to auto.

#### Example:
```bash
python scripts/run_reconstruction.py \
    --sax-file /path/to/outputs/segmentations/subject-001_2D_seg_sax.nii.gz \
    --ch2-file /path/to/outputs/segmentations/subject-001_2D_seg_lax_2ch.nii.gz \
    --ch4-file /path/to/outputs/segmentations/subject-001_2D_seg_lax_4ch.nii.gz \
    --output-dir /path/to/outputs/reconstructions \
    --output-prefix "subject-001"
```

**Output:** This will create the final `subject-001_whole_heart_segmentation.nii.gz` in the output directory, 
along with several intermediate and quality-control files (e.g., ...`_sparse_volume.nii.gz`, ...`_qc_sax_backprojected.nii.gz`).

### 3. `run_full_pipeline.py`: The End-to-End Solution
This is the main script that automates the entire process. 
It takes a directory of raw CINE images, runs the 2D segmentations for all views, and then automatically runs the 3D reconstruction.

#### Arguments:
* `--input-dir`: Path to a directory containing the raw CINE MRI files (e.g., ..._SAX.nii.gz, ..._CH2.nii.gz, etc.).
* `--output-dir`: Directory where all output files will be saved.
* `--output-prefix`: A name to prefix all output filenames. If not provided, it is automatically inferred from the name of the --input-dir.
* `--device`: The device to run on. Defaults to auto.
#### Example:
```bash
python scripts/run_full_pipeline.py \
    --input-dir /path/to/data/subject-001/ \
    --output-dir /path/to/outputs/full_run/ \
    --output-prefix "subject-001_final"
```

**Output:** This will create a flat list of all files (intermediate segmentations and final reconstruction) in the 
`/path/to/outputs/full_run/` directory, all prefixed with `subject-001_final`.