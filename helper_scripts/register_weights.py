"""Register the weights in a draft GitHub release into models.yaml.

Standalone orchestrator, run by ``.github/workflows/register-weights.yml``. It
is not imported by the package. The archive rules live in
``cardio_form.packaging``; this script only owns I/O.

The release must hold the ``.zip`` assets plus a ``weights-manifest.json``
asset, a JSON list with one object per archive::

    [{"manifest_key": "segment_sax", "version": "v0.3.0",
      "zip_name": "segment_sax_v0.3.0.zip", "sha256": "...",
      "size_bytes": 123, "note": "foreground Dice 0.93 (n=12)"}]

For each entry the script checks the hash, the archive layout and the name,
then adds the version under ``versions:``. It never changes ``default:``;
promotion is a separate, deliberate commit. Nothing is written unless every
entry passes.

Usage::

    python helper_scripts/register_weights.py \\
        --tag v0.3.0-models --repo OpenHeartDevelopers/cardio-form \\
        --manifest src/cardio_form/config_data/models.yaml \\
        --pr-body-out pr_body.md

``--asset-dir`` skips the download and reads assets already on disk.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from cardio_form.packaging import (
    ArchiveSpec,
    check_zip_name,
    manifest_entry,
    sha256_of,
    validate_archive,
)

WEIGHTS_MANIFEST = 'weights-manifest.json'
WEIGHTS_MANIFEST_FIELDS = ('manifest_key', 'version', 'zip_name', 'sha256', 'size_bytes', 'note')

# Development entry kept last under 'versions:' by convention.
LOCAL_DEV_VERSION = 'local_dev'

RELEASE_URL = 'https://github.com/{repo}/releases/download/{tag}/{zip_name}'


def make_yaml() -> YAML:
    """Round-trip loader that keeps models.yaml's comments, quotes and nulls."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # never fold the long release URLs
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.representer.add_representer(
        type(None),
        lambda representer, _: representer.represent_scalar('tag:yaml.org,2002:null', 'null'))
    return yaml


def download_release(repo: str, tag: str, dest: Path):
    """Download every asset of a (possibly draft) release with the gh CLI."""
    subprocess.run(
        ['gh', 'release', 'download', tag, '--repo', repo, '--dir', str(dest)],
        check=True)


def load_weights_manifest(path: Path) -> list:
    """Read the release's weights-manifest.json and check its fields."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path.name} must be a non-empty JSON list")
    for index, entry in enumerate(entries):
        missing = [f for f in WEIGHTS_MANIFEST_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"{path.name} entry {index} is missing {missing}")
    return entries


def insert_version(manifest, manifest_key: str, version: str, entry: dict):
    """Add a version under 'versions:', before local_dev so it stays last."""
    versions = manifest[manifest_key]['versions']
    keys = list(versions)
    position = keys.index(LOCAL_DEV_VERSION) if LOCAL_DEV_VERSION in keys else len(keys)
    versions.insert(position, version, entry)


def register(entries: list, asset_dir: Path, manifest, repo: str, tag: str) -> list:
    """Check every entry and insert it into the manifest. Returns problems.

    Each entry is checked against the manifest *after* the earlier entries were
    inserted, so two entries in one release cannot share a name or version.
    The caller must not save the manifest if any problem is returned.
    """
    problems = []
    for entry in entries:
        key, version, zip_name = entry['manifest_key'], entry['version'], entry['zip_name']
        zip_path = asset_dir / zip_name

        if key not in manifest:
            problems.append(f"{key}: not in models.yaml; a new model needs a code change first")
            continue
        if not zip_path.is_file():
            problems.append(f"{zip_name}: listed in {WEIGHTS_MANIFEST} but not in the release")
            continue

        actual = sha256_of(zip_path)
        if actual != entry['sha256']:
            problems.append(f"{zip_name}: sha256 {actual} does not match declared {entry['sha256']}")

        entry_problems = validate_archive(zip_path, key) + check_zip_name(manifest, key, version, zip_name)
        problems.extend(entry_problems)
        if entry_problems:
            continue

        spec = ArchiveSpec(
            manifest_key=key,
            version=version,
            url=RELEASE_URL.format(repo=repo, tag=tag, zip_name=zip_name),
            sha256=actual,
        )
        stanza = manifest_entry(spec)
        stanza['url'] = DoubleQuotedScalarString(stanza['url'])  # match existing style
        insert_version(manifest, key, version, stanza)
    return problems


def pr_body(tag: str, entries: list) -> str:
    """Markdown body for the registration pull request."""
    lines = [
        f"Registers the weights in draft release `{tag}` in `models.yaml`.",
        "",
        "| Model | Version | Archive | Size (MB) | sha256 | Note |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| `{e['manifest_key']}` | `{e['version']}` | `{e['zip_name']}` | "
            f"{e['size_bytes'] / (1024 * 1024):.1f} | `{e['sha256'][:12]}…` | {e['note']} |")
    lines += [
        "",
        "`default:` is unchanged. Before merging:",
        f"1. Publish the draft release `{tag}`. Its assets cannot be downloaded while it is a draft.",
        "2. Promote a version to `default:` in a separate commit, if wanted.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Register the weights in a draft GitHub release into models.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.3.0-models.")
    parser.add_argument("--repo", required=True, help="owner/name of the repository holding the release.")
    parser.add_argument("--manifest", required=True, help="Path to models.yaml, edited in place.")
    parser.add_argument("--pr-body-out", default=None, help="Write the pull-request body to this file.")
    parser.add_argument("--asset-dir", default=None,
                        help="Read assets from this directory instead of downloading the release.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    yaml = make_yaml()
    manifest = yaml.load(manifest_path)

    with tempfile.TemporaryDirectory(prefix='cardioform_register_') as scratch:
        asset_dir = Path(args.asset_dir) if args.asset_dir else Path(scratch)
        if not args.asset_dir:
            download_release(args.repo, args.tag, asset_dir)

        entries = load_weights_manifest(asset_dir / WEIGHTS_MANIFEST)
        problems = register(entries, asset_dir, manifest, args.repo, args.tag)

    if problems:
        print(f"Refusing to register {args.tag}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    yaml.dump(manifest, manifest_path)
    for e in entries:
        print(f"registered {e['manifest_key']:<22} {e['version']:<10} {e['zip_name']}")

    if args.pr_body_out:
        Path(args.pr_body_out).write_text(pr_body(args.tag, entries))


if __name__ == "__main__":
    main()
