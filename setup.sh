#!/bin/bash
# One-time developer setup for CardioForm.
#
# Run this once inside the activated conda environment:
#     conda activate cardioform
#     ./setup.sh
#
# It installs the shared `pycemrg` core and CardioForm itself as editable
# packages, then (re)generates CARDIOFORM_ENV_SETUP for activating the env in
# new shells. Under the src/ layout there is no PYTHONPATH to set anymore.
set -e

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
SETUP_FILE="$SCRIPT_DIR/CARDIOFORM_ENV_SETUP"
PYCEMRG_LOCAL="$SCRIPT_DIR/../pycemrg"

# pycemrg: prefer a local editable checkout (sibling ../pycemrg) for development,
# otherwise fall back to the pinned PyPI release.
if [ -d "$PYCEMRG_LOCAL" ]; then
    echo "Installing local pycemrg (editable) from $PYCEMRG_LOCAL"
    pip install -e "$PYCEMRG_LOCAL" --no-deps
else
    echo "Local pycemrg checkout not found; installing pinned release from PyPI"
    pip install "pycemrg==0.1.1"
fi

# CardioForm itself. --no-deps: the conda env is the source of truth for the
# pinned scientific stack (numpy==1.24.4, torch/CUDA, nnunetv2).
echo "Installing CardioForm (editable) from $SCRIPT_DIR"
pip install -e "$SCRIPT_DIR" --no-deps

# Regenerate the activation helper (no PYTHONPATH needed under the src/ layout).
cat > "$SETUP_FILE" <<'EOL'
# Activate the conda environment for CardioForm.
conda activate cardioform
EOL

echo "Setup complete. The 'cardioform' command is now available in the env."
echo "In new shells, run 'source CARDIOFORM_ENV_SETUP' to activate the environment."
