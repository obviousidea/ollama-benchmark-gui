"""
start.py — Universal cross-platform launcher for Ollama Benchmark
──────────────────────────────────────────────────────────────────
Works on Windows, macOS (Intel & Apple Silicon), and Linux.
Uses uv for dependency management (auto-installed if missing).

Usage:
    python start.py
    python3 start.py
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.resolve()


def set_ollama_env() -> str:
    arch   = platform.machine().lower()
    system = platform.system()

    os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
    os.environ["OLLAMA_KV_CACHE_TYPE"]   = "q4_0"

    if system == "Darwin":
        os.environ["TK_SILENCE_DEPRECATION"] = "1"
        if arch == "arm64":
            os.environ["OLLAMA_METAL"] = "1"
            os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
            return "macOS · Apple Silicon (Metal)"
        return "macOS · Intel"

    if system == "Windows":
        return "Windows"

    return f"Linux ({arch})"


def ensure_uv() -> str:
    """Return path to uv, installing it if necessary."""
    # Already on PATH?
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return "uv"
    except FileNotFoundError:
        pass

    print("  uv not found — installing...")
    system = platform.system()

    if system == "Windows":
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass",
             "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
            check=True,
        )
        # uv lands in %USERPROFILE%\.local\bin or %USERPROFILE%\.cargo\bin
        for candidate in [
            Path.home() / ".local" / "bin" / "uv.exe",
            Path.home() / ".cargo"  / "bin" / "uv.exe",
        ]:
            if candidate.exists():
                return str(candidate)
    else:
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True, check=True,
        )
        for candidate in [
            Path.home() / ".local" / "bin" / "uv",
            Path.home() / ".cargo" / "bin" / "uv",
        ]:
            if candidate.exists():
                return str(candidate)

    return "uv"  # hope it's on PATH after install


def print_version(uv: str):
    try:
        result = subprocess.run(
            [uv, "run", "python", "-c",
             "from versioning import get_full_label; print('  ' + get_full_label())"],
            capture_output=True, text=True, cwd=str(PROJECT_DIR),
        )
        if result.stdout.strip():
            print(result.stdout.strip())
    except Exception:
        pass


def main():
    os.chdir(str(PROJECT_DIR))

    tag = set_ollama_env()
    uv  = ensure_uv()

    try:
        uv_ver = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        uv_ver = "?"

    print()
    print(f"  Platform : {tag}")
    print(f"  uv       : {uv_ver}")
    print()

    print_version(uv)
    print()

    # Hand off to uv — inherits all env vars set above
    os.execvp(uv, [uv, "run", "app.py"])


if __name__ == "__main__":
    main()
