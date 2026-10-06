"""The release-archive contract for CardioForm model weights.

``ModelManager.get_model_path`` (``models.py``) downloads a zip, extracts it to
``<cache_dir>/<zip stem>/`` and then expects one file at a fixed place inside
it. Nothing checks that place at publish time, so a badly built zip downloads
and verifies cleanly, then fails on a user's machine. This module states the
contract once, so publishing tools and CI can check an archive *before* it is
released.

Pure and stdlib-only: no torch, no network, no file writes. Callers own I/O.
"""

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Manifest keys with this prefix are nnUNetv2 model folders (models.py).
NNUNET_KEY_PREFIX = 'segment_'

# Files nnUNet reads from the model folder, and where ModelManager looks for
# the checkpoint inside it (segment_2d.py initialises with use_folds=('all',)).
NNUNET_FOLDER_FILES = ('dataset.json', 'plans.json', 'dataset_fingerprint.json')
NNUNET_CHECKPOINT = 'fold_all/checkpoint_final.pth'


@dataclass(frozen=True)
class ArchiveSpec:
    """One published weights archive, as recorded in models.yaml."""
    manifest_key: str   # top-level key in models.yaml, e.g. 'segment_sax'
    version: str        # entry under 'versions:', e.g. 'v0.3.0'
    url: str            # download URL of the zip
    sha256: str         # hex digest of the zip


def zip_stem(zip_name: str) -> str:
    """'segment_sax_v0.3.0.zip' -> 'segment_sax_v0.3.0' (the cache directory)."""
    return PurePosixPath(zip_name).name.removesuffix('.zip')


def expected_members(manifest_key: str, zip_name: str) -> list:
    """Archive members ModelManager needs, relative to the archive root."""
    if manifest_key.startswith(NNUNET_KEY_PREFIX):
        return [*NNUNET_FOLDER_FILES, NNUNET_CHECKPOINT]
    # Reconstruction nets: the .pth basename must equal the zip basename.
    return [f'{zip_stem(zip_name)}.pth']


def validate_archive(zip_path, manifest_key: str) -> list:
    """Check an archive's layout against the contract.

    Returns a list of problems; an empty list means the archive is valid.
    Extra members (e.g. nnUNet's ``fold_all/debug.json``) are allowed.
    """
    zip_path = Path(zip_path)
    if zip_path.suffix != '.zip':
        return [f"{zip_path.name}: not a .zip file"]
    if not zipfile.is_zipfile(zip_path):
        return [f"{zip_path.name}: not a readable zip archive"]

    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())

    problems = []
    for required in expected_members(manifest_key, zip_path.name):
        if required in members:
            continue
        # The common packing mistake: zipping the folder, not its contents.
        wrapped = [m for m in members if m.endswith('/' + required)]
        if wrapped:
            problems.append(
                f"{zip_path.name}: '{required}' is inside a wrapper folder "
                f"('{wrapped[0]}'); zip the folder's contents, not the folder")
        else:
            problems.append(f"{zip_path.name}: missing '{required}'")
    return problems


def check_zip_name(manifest: dict, manifest_key: str, version: str, zip_name: str) -> list:
    """Check a NEW archive's filename against the manifest.

    Two rules, both for new publications only (the v0.1.0 names predate them):

    - The name must contain the version, so a reader can tell assets apart.
    - The name must not already appear in any entry's URL. ModelManager caches
      by filename alone, so a reused name silently serves the old weights.

    Returns a list of problems; an empty list means the name is acceptable.
    """
    problems = []
    if version not in zip_name:
        problems.append(f"{zip_name}: name does not contain version '{version}'")

    for key, info in manifest.items():
        for existing_version, entry in info.get('versions', {}).items():
            if PurePosixPath(entry.get('url', '')).name == zip_name:
                problems.append(
                    f"{zip_name}: already used by {key} {existing_version}; "
                    f"ModelManager caches by filename, so this name must be unique")

    entry = manifest.get(manifest_key, {})
    if version in entry.get('versions', {}):
        problems.append(f"{manifest_key} {version}: version already in manifest")
    return problems


def sha256_of(path) -> str:
    """SHA256 of a file, read in chunks so a 200 MB archive is not held in RAM."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_entry(spec: ArchiveSpec) -> dict:
    """The value to insert at ``<manifest_key>.versions.<version>``."""
    return {'url': spec.url, 'sha256': spec.sha256}
