#!/usr/bin/env python3
"""Validate and build a deterministic submission ZIP plus external SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

try:
    from .validate_submission import REPO_ROOT, ValidationError, validate_submission
except ImportError:
    from validate_submission import REPO_ROOT, ValidationError, validate_submission


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = handle.name
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def package_submission(source: Path | str, output: Path | str) -> dict:
    source, output = Path(source), Path(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValidationError("Output ZIP must be outside the submission source directory")
    if output.suffix.lower() != ".zip":
        raise ValidationError("Output archive must have a .zip extension")
    checked = validate_submission(source)
    buffer = io.BytesIO()
    # ZIP_STORED avoids zlib-version differences. These declarative files are tiny.
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(checked.files.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_STORED
            archive.writestr(entry, data)
    archive_bytes = buffer.getvalue()
    manifest = {
        "schema_version": 1,
        "validation_scope": "local supported-subset checks; not official harness validation",
        "archive": {
            "name": output.name,
            "bytes": len(archive_bytes),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        },
        "unpacked_bytes": sum(map(len, checked.files.values())),
        "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(checked.files.items())],
    }
    _atomic_write(output, archive_bytes)
    _atomic_write(output.with_suffix(".manifest.json"),
                  (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", nargs="?", type=Path, default=REPO_ROOT / "submission")
    parser.add_argument("--output", "-o", type=Path, default=REPO_ROOT / "dist" / "submission.zip")
    args = parser.parse_args()
    try:
        manifest = package_submission(args.submission_dir, args.output)
    except (ValidationError, OSError) as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"Created {args.output} ({manifest['archive']['bytes']} bytes)")
    print(f"SHA-256: {manifest['archive']['sha256']}")
    print(f"Manifest: {args.output.with_suffix('.manifest.json')}")
    print("Local supported-subset checks passed; official harness validation and scoring are still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
