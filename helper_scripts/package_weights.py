"""Package retrained nnUNetv2 2D weights into CardioForm release assets.

Standalone packaging tool, like the rest of ``helper_scripts/``. It is not
imported by the package.

Three things stop a retrained checkpoint being published as-is, and this script
fixes all three while repackaging. None of them alters the learned weights.

1. **Trainer class.** The checkpoints record a custom ``trainer_name`` that is
   not part of ``nnunetv2``. ``initialize_from_trained_model_folder`` raises
   ``RuntimeError`` when it cannot import that name. Variants that override only
   ``num_epochs`` inherit ``build_network_architecture`` unchanged, so the stock
   ``nnUNetTrainer`` is inference-equivalent.
2. **numpy version.** Checkpoints pickled under numpy 2.x reference
   ``numpy._core``, which does not exist in the numpy 1.24.4 that CardioForm
   pins. Aliasing the names lets them load once; re-saving means no consumer
   ever needs the alias.
3. **Dataset name.** nnUNet stores the training dataset's name in two places,
   both readable by anyone who downloads the asset. Each is rewritten to the
   neutral ``public_dataset_name`` below.

The source layout is injected through ``--source-map``, not hardcoded, so this
file carries no information about the training cohort.

Usage::

    python helper_scripts/package_weights.py \
        --source-root /path/to/2D_segmentation/model_weights \
        --source-map  /path/to/source_map.json \
        --output-dir  release_assets

``--source-map`` is JSON, keyed by manifest key::

    {
      "segment_sax": {"dataset_dir": "...", "trainer_dir": "...", "fold": "fold_0"}
    }
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

# The stock nnUNet trainer, guaranteed present in every nnunetv2 version.
TARGET_TRAINER_NAME = 'nnUNetTrainer'

# Files nnUNet reads from the model folder alongside the checkpoint.
MODEL_FOLDER_FILES = ('dataset.json', 'plans.json', 'dataset_fingerprint.json')

# ModelManager expects segment_* checkpoints at <unzip_dir>/fold_all/
# (models.py:107), and segment_2d.py initialises with use_folds=('all',).
STAGED_FOLD_DIR = 'fold_all'
STAGED_CHECKPOINT_NAME = 'checkpoint_final.pth'

# Keys required of every entry in the --source-map file.
SOURCE_MAP_FIELDS = ('dataset_dir', 'trainer_dir', 'fold')


@dataclass(frozen=True)
class WeightSpec:
    """What one packaged model becomes. Carries no source-tree information."""
    manifest_key: str          # key in config_data/models.yaml
    public_dataset_name: str   # replaces the training dataset's stored name
    zip_name: str              # output asset filename; unique per version
    note: str                  # provenance, echoed in the summary


@dataclass(frozen=True)
class SourceLocation:
    """Where one model's files are read from. Injected, never hardcoded."""
    dataset_dir: str
    trainer_dir: str
    fold: str


# v0.2.0. Folds chosen by best held-out foreground Dice from each run's own
# validation/summary.json.
PUBLICATION_V0_2_0 = (
    WeightSpec(
        manifest_key='segment_sax',
        public_dataset_name='Dataset315_SAX',
        zip_name='segment_sax_v0.2.0.zip',
        note='foreground Dice 0.9173 (n=12), only validated fold',
    ),
    WeightSpec(
        manifest_key='segment_lax_2ch',
        public_dataset_name='Dataset316_LAX2CH',
        zip_name='segment_lax_2ch_v0.2.0.zip',
        note='foreground Dice 0.9314 (n=12), only validated fold',
    ),
    WeightSpec(
        manifest_key='segment_lax_4ch',
        public_dataset_name='Dataset317_LAX4CH',
        zip_name='segment_lax_4ch_v0.2.0.zip',
        note='foreground Dice 0.9377 (n=12), best of five (0.9249-0.9377)',
    ),
)


def install_numpy_core_alias():
    """Let numpy 1.x unpickle arrays saved by numpy 2.x.

    numpy 2 renamed ``numpy.core`` to ``numpy._core`` and pickles carry the new
    path. Registering the old modules under the new names satisfies the
    unpickler. Must run before any ``torch.load`` of a numpy-2 checkpoint.
    """
    submodules = ('', '.multiarray', '.umath', '.numeric',
                  '._multiarray_umath', '.numerictypes')
    for suffix in submodules:
        module = __import__('numpy.core' + suffix, fromlist=['_'])
        sys.modules['numpy._core' + suffix] = module


def load_source_map(path: Path) -> dict:
    """Read and validate the injected source layout."""
    raw = json.loads(Path(path).read_text())
    locations = {}
    for key, entry in raw.items():
        missing = [f for f in SOURCE_MAP_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"Source map entry '{key}' is missing {missing}")
        locations[key] = SourceLocation(**{f: entry[f] for f in SOURCE_MAP_FIELDS})
    return locations


def sanitise_checkpoint(checkpoint: dict, trainer_name: str, dataset_name: str) -> dict:
    """Return the checkpoint with its trainer class and dataset name replaced.

    Pure: takes a loaded checkpoint dict, returns a new one. Every other key is
    carried through untouched, so the asset keeps the full training state.
    """
    sanitised = dict(checkpoint)
    sanitised['trainer_name'] = trainer_name

    init_args = dict(sanitised.get('init_args', {}))
    if 'plans' in init_args:
        plans = dict(init_args['plans'])
        plans['dataset_name'] = dataset_name
        init_args['plans'] = plans
    sanitised['init_args'] = init_args
    return sanitised


def rewrite_plans_dataset_name(plans_path: Path, dataset_name: str):
    """Replace ``dataset_name`` in a staged plans.json, in place."""
    plans = json.loads(plans_path.read_text())
    plans['dataset_name'] = dataset_name
    plans_path.write_text(json.dumps(plans, indent=4, sort_keys=False))


def sha256_of(path: Path) -> str:
    """SHA256 of a file, read in chunks so a 200 MB asset is not held in RAM."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_staged(staged: Path, forbidden: str) -> int:
    """Count occurrences of a string across the staged text files.

    A non-zero count means the packaged asset would publish it.
    """
    hits = 0
    needle = forbidden.lower()
    for name in MODEL_FOLDER_FILES:
        path = staged / name
        if path.is_file():
            hits += path.read_text(errors='ignore').lower().count(needle)
    return hits


def stage_model(spec: WeightSpec, location: SourceLocation,
                source_root: Path, staging_root: Path) -> Path:
    """Build the directory that becomes the zip, and return it.

    Layout matches the unpacked shape of the existing v0.1.0 assets: the three
    nnUNet JSON files at the root, the checkpoint under fold_all/.
    """
    import torch

    trainer_path = source_root / location.dataset_dir / location.trainer_dir
    if not trainer_path.is_dir():
        raise FileNotFoundError(f"Trainer directory not found: {trainer_path}")

    staged = staging_root / spec.zip_name.replace('.zip', '')
    if staged.exists():
        shutil.rmtree(staged)
    (staged / STAGED_FOLD_DIR).mkdir(parents=True)

    for filename in MODEL_FOLDER_FILES:
        source = trainer_path / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing {filename} in {trainer_path}")
        shutil.copy2(source, staged / filename)

    rewrite_plans_dataset_name(staged / 'plans.json', spec.public_dataset_name)
    print(f"  plans.json dataset_name -> {spec.public_dataset_name}")

    source_checkpoint = trainer_path / location.fold / STAGED_CHECKPOINT_NAME
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint at {source_checkpoint}")

    checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
    configuration = checkpoint.get('init_args', {}).get('configuration')
    print(f"  trainer '{checkpoint.get('trainer_name')}' -> '{TARGET_TRAINER_NAME}', "
          f"configuration '{configuration}', fold '{location.fold}' -> '{STAGED_FOLD_DIR}'")

    sanitised = sanitise_checkpoint(checkpoint, TARGET_TRAINER_NAME, spec.public_dataset_name)
    torch.save(sanitised, staged / STAGED_FOLD_DIR / STAGED_CHECKPOINT_NAME)

    return staged


def zip_staged(staged: Path, output_path: Path) -> Path:
    """Zip the staged directory's CONTENTS, with no wrapping top-level folder.

    ModelManager extracts into cache_dir/<zip name without .zip>, so a wrapper
    directory would bury the checkpoint one level too deep and fail only at
    model-resolution time on a user's machine.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(staged.rglob('*')):
            if item.is_file():
                archive.write(item, item.relative_to(staged))
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Package retrained nnUNetv2 2D weights into CardioForm release assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", required=True,
                        help="Directory holding the nnUNet dataset directories (read-only).")
    parser.add_argument("--source-map", required=True,
                        help="JSON file giving dataset_dir / trainer_dir / fold per manifest key.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the .zip assets into (e.g. release_assets).")
    parser.add_argument("--staging-dir", default=None,
                        help="Directory for the intermediate staged trees. A temporary "
                             "directory is used and removed if not given.")
    parser.add_argument("--audit-string", default=None,
                        help="Fail if this string survives into a staged asset. Use the "
                             "cohort name you are not publishing.")
    parser.add_argument("--manifest-json", default=None,
                        help="Also write the name/sha256/note summary to this JSON file.")
    args = parser.parse_args()

    install_numpy_core_alias()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    locations = load_source_map(Path(args.source_map))

    with tempfile.TemporaryDirectory(prefix='cardioform_pkg_') as scratch:
        staging_root = Path(args.staging_dir).resolve() if args.staging_dir else Path(scratch)
        staging_root.mkdir(parents=True, exist_ok=True)

        summary = []
        for spec in PUBLICATION_V0_2_0:
            if spec.manifest_key not in locations:
                raise KeyError(f"Source map has no entry for '{spec.manifest_key}'")
            print(f"\n--- {spec.manifest_key} ({spec.note}) ---")
            staged = stage_model(spec, locations[spec.manifest_key], source_root, staging_root)

            if args.audit_string:
                hits = audit_staged(staged, args.audit_string)
                if hits:
                    raise RuntimeError(
                        f"'{args.audit_string}' still appears {hits} time(s) in the staged "
                        f"text files for {spec.manifest_key}. Refusing to package."
                    )
                print(f"  audit: '{args.audit_string}' absent from staged JSON")

            asset = zip_staged(staged, output_dir / spec.zip_name)
            digest = sha256_of(asset)
            print(f"  wrote {asset} ({asset.stat().st_size / (1024 * 1024):.1f} MB)")
            print(f"  sha256 {digest}")
            summary.append({
                'manifest_key': spec.manifest_key,
                'zip_name': spec.zip_name,
                'sha256': digest,
                'size_bytes': asset.stat().st_size,
                'note': spec.note,
            })

    print("\n--- Summary ---")
    for entry in summary:
        print(f"{entry['manifest_key']:<18} {entry['zip_name']:<32} {entry['sha256']}")

    if args.manifest_json:
        manifest_path = Path(args.manifest_json).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSummary written to {manifest_path}")


if __name__ == "__main__":
    main()
