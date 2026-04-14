"""
Versioning helpers for Ollama Benchmark.

version.json is the single source of truth:
  build    — integer, auto-incremented every time the app starts (never resets)
  last_run — ISO timestamp of last launch

Version label format: YYYY.MM.DD.{build}  e.g. 2026.04.14.6
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
    """Return version string: YYYY.MM.DD.{build}"""
    d = _load()
    date_str = datetime.now().strftime("%Y.%m.%d")
    return f"{date_str}.{d['build']}"


def get_build() -> int:
    return _load()["build"]


def get_full_label() -> str:
    """Return version string used in the UI title: YYYY.MM.DD.{build}"""
    return get_version()


def increment_build() -> int:
    """Increment build counter and update last_run. Returns new build number."""
    d = _load()
    d["build"] += 1
    d["last_run"] = datetime.now().isoformat(timespec="seconds")
    _save(d)
    return d["build"]
