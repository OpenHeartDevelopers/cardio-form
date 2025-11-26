#!/usr/bin/env bash
set -euo pipefail

# ===================================================================================
# CardioForm Docker Runner
# ===================================================================================
# This script is a user-friendly wrapper for running the CardioForm Docker container.
# ===================================================================================

# --- Default Configuration ---
IMAGE_NAME="${CARDIOFORM_IMAGE:-cemrg/cardio-form:latest}"
CACHE_DIR="${HOME}/.cache/cardio_form"
GPU_FLAG="" # By default, do not use GPU

# --- Helper Functions ---
show_help() {
    echo "Usage: $(basename "$0") [WRAPPER_OPTIONS] /path/to/data/dir [MODE] [MODE_OPTIONS]"
    echo ""
    echo "A wrapper script to run the CardioForm Docker container."
    echo "It maps your data directory to '/data' inside the container."
    echo ""
    echo "IMPORTANT: All file paths provided to the mode options must be relative to '/data'."
    echo ""
    echo "Wrapper Options:"
    echo "  --gpu          Enable NVIDIA GPU support (requires nvidia-docker)."
    echo "  -h, --help     Show this help message."
    echo ""
    echo "Example (Full Pipeline):"
    echo "  ./run-docker.sh --gpu /host/path/to/subject001 full --output-dir /data --output-prefix subject001"
    echo ""
    echo "Example (Segmenting a single file):"
    echo "  ./run-docker.sh /host/path/to/subject001 segment --input /data/CINE_SAX.nii.gz --output-dir /data/outputs --output-prefix subject001 --view-type sax"
    echo ""
    echo "You can override the Docker image with the CARDIOFORM_IMAGE environment variable."
}

# --- Argument Parsing for THIS script ---
# We need to manually parse our own options (--gpu, --help) before passing the rest to Docker.
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Check for --gpu flag and remove it from the argument list
NEW_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --gpu)
      GPU_FLAG="--gpus all"
      echo "--> GPU support enabled."
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      NEW_ARGS+=("$arg")
      ;;
  esac
done
# Overwrite the original arguments with the filtered list
set -- "${NEW_ARGS[@]}"

# --- Main Logic ---

# The first remaining argument MUST be the data directory
DATA_DIR="$1"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: The first argument must be a valid directory to mount as your data volume."
    echo "You provided: '$DATA_DIR'"
    exit 1
fi
# Remove the data directory from the list of arguments to pass to the container
shift

# Check if there are any mode arguments left
if [ $# -eq 0 ]; then
    echo "Error: No mode specified. Please specify a mode (e.g., 'segment', 'full')."
    show_help
    exit 1
fi

echo "--> Mounting host data directory: '$DATA_DIR' to '/data' in container."

DOCKER_CACHE='/.cache/cardio_form'
# Ensure the local cache directory exists
mkdir -p "${CACHE_DIR}"
echo "--> Mounting host cache directory: '${CACHE_DIR}' to '${DOCKER_CACHE}'."

echo "--> Running Docker image: ${IMAGE_NAME}"

# --- The Core Command ---
docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    ${GPU_FLAG} \
    --volume="${DATA_DIR}:/data" \
    --volume="${CACHE_DIR}:${DOCKER_CACHE}" \
    "${IMAGE_NAME}" \
    "$@" # Pass all REMAINING arguments to the container
