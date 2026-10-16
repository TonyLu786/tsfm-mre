"""Run manifests: configuration, software versions and SHA-256 digests of all outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from tsfm_mre._version import __version__

DEPENDENCIES = ("numpy", "pandas", "scipy", "matplotlib", "PyYAML")


def file_digest(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(cwd: Path) -> dict[str, Any] | None:
    """Commit hash and working-tree state of the repository containing ``cwd``."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "clean": status == ""}


def build_manifest(
    config: dict[str, Any],
    data: dict[str, Any],
    models: dict[str, dict[str, Any]],
    out_dir: Path,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Describe a run; output paths are relative to ``out_dir``."""
    outputs = {
        p.relative_to(out_dir).as_posix(): file_digest(p)
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "tsfm_mre": __version__,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "runtime_seconds": round(runtime_seconds, 2),
        "git": git_revision(Path(__file__).resolve().parent),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "config": config,
        "models": models,
        "data": data,
        "outputs": outputs,
        "command": command_line(sys.argv),
    }


def command_line(argv: list[str]) -> str:
    """The invoking command with local paths reduced to file names."""
    words = []
    for i, arg in enumerate(argv):
        path = Path(arg)
        if i == 0:
            in_package = path.parent.name == "tsfm_mre" and path.name in {"__main__.py", "cli.py"}
            is_script = path.stem == "tsfm-mre"
            words.append("tsfm-mre" if in_package or is_script else path.name)
        else:
            words.append(path.name if path.is_absolute() else arg)
    return " ".join(words)


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write ``manifest.json`` with Unix line endings."""
    path = out_dir / "manifest.json"
    text = json.dumps(manifest, indent=2, default=str) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
