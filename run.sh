#!/usr/bin/env bash
# ── Ollama Benchmark — macOS / Linux launcher ─────────────────────────────────
cd "$(dirname "$0")"

# ── Ollama performance variables ──────────────────────────────────────────────
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
export TK_SILENCE_DEPRECATION=1

[[ "$(uname -m)" == "arm64" ]] && export OLLAMA_METAL=1

# ── Install uv if missing ─────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "  uv       : $(uv --version)"
echo "  Platform : $(uname -m) / $(uname -s)"
echo ""

uv run python -c "from versioning import get_full_label; print('  ' + get_full_label())"
echo ""

uv run app.py
