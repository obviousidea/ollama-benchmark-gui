"""
Versioning helpers for Ollama Benchmark.

version.json is the single source of truth:
  major.minor.patch  — semantic version, bumped manually via bump.py
  build              — auto-incremented every time the app starts
  last_run           — ISO timestamp of last launch
"""

import json
from datetime import datetime
from pathlib import Path

_VERSION_FILE = Path(__file__).parent / "version.json"


def _load() -> dict:
    return json.loads(_VERSION_FILE.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    _VERSION_FILE.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def get_version() -> str:
    """Return 'MAJOR.MINOR.PATCH' string."""
    d = _load()
    return f"{d['major']}.{d['minor']}.{d['patch']}"


def get_build() -> int:
    return _load()["build"]


def get_full_label() -> str:
    """Return e.g. 'v1.0.0  build 42  (2026-04-03)'."""
    d = _load()
    date_str = d.get("last_run", "")[:10]
    return f"v{d['major']}.{d['minor']}.{d['patch']}  build {d['build']}  ({date_str})"


def increment_build() -> int:
    """Increment build counter and update last_run. Returns new build number."""
    d = _load()
    d["build"] += 1
    d["last_run"] = datetime.now().isoformat(timespec="seconds")
    _save(d)
    return d["build"]


def bump(part: str) -> str:
    """
    Bump semantic version.
      part: 'major' | 'minor' | 'patch'
    Resets lower parts to 0. Returns new version string.
    """
    if part not in ("major", "minor", "patch"):
        raise ValueError(f"part must be 'major', 'minor' or 'patch', got '{part}'")
    d = _load()
    if part == "major":
        d["major"] += 1
        d["minor"] = 0
        d["patch"] = 0
    elif part == "minor":
        d["minor"] += 1
        d["patch"] = 0
    else:
        d["patch"] += 1
    _save(d)
    return f"{d['major']}.{d['minor']}.{d['patch']}"
