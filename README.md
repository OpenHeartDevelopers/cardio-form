# CardioForm

A pipeline for whole-heart segmentation and 3D reconstruction from CINE MRI.

---

## Installation

This guide provides instructions for setting up the project in a clean virtual environment using either Conda or Python's built-in `venv`.

### Prerequisites
*   Python 3.11
*   Git
*   (Optional but Recommended) An NVIDIA GPU with CUDA 11.8 drivers installed.

### Step 1: Clone the Repository

First, clone the project from GitHub to your local machine.

```bash
git clone https://github.com/OpenHeartDevelopers/cardio-form.git
cd cardio-form
```

### Step 2: Create and Activate a Virtual Environment

Choose **one** of the following options.

#### Option A: Using `conda`

This is the recommended approach if you use the Anaconda distribution.

```bash
# Create a new conda environment with Python 3.11
conda create -n cardioform python=3.11 -y

# Activate the environment
conda activate cardioform
```

#### Option B: Using `venv` (Python's built-in tool)

This is a lightweight alternative that doesn't require Anaconda.

```bash
# Create a virtual environment named '.venv' in the project directory
python -m venv .venv

# Activate the environment
# On Linux or macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
```

### Step 3: Install PyTorch with CUDA Support

The performance of this pipeline heavily relies on GPU acceleration. 
The following command installs PyTorch compatible with **CUDA 11.8**. 
If you have a different CUDA version or need a CPU-only installation, 
please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install CardioForm and Dependencies

Finally, install the `cardio-form` package and all its dependencies defined in the project configuration. 
This command should be run from the root of the project directory (where `pyproject.toml` is located).

```bash
# This command installs the project and automatically pulls in all other dependencies.
pip install .
```

After these steps, the pipeline is ready to be used.

---