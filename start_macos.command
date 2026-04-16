#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#   Ollama Benchmark — by obviousidea
#   Lanceur macOS — double-cliquez depuis le Finder
# ══════════════════════════════════════════════════════════════════

# Se placer dans le dossier du script (indispensable pour double-clic Finder)
cd "$(dirname "$0")"

clear
echo ""
echo " ============================================================"
echo "   Ollama Benchmark — by obviousidea"
echo " ============================================================"
echo ""

# ── Variables de performance Ollama ──────────────────────────────
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
export TK_SILENCE_DEPRECATION=1   # silence les avertissements Tk sur macOS

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    export OLLAMA_METAL=1
    PLATFORM="Apple Silicon (Metal activé)"
else
    PLATFORM="Intel"
fi

echo " Plateforme : macOS · $PLATFORM"
echo ""

# ── Installation de uv si absent ─────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo " [1/3] Installation de uv (gestionnaire de paquets)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    echo " [1/3] uv installé."
else
    echo " [1/3] uv détecté : OK"
fi
echo ""

# ── Vérification et réparation de l'environnement Python ─────────
echo " [2/3] Vérification de l'environnement Python..."
uv sync --quiet
if [[ $? -ne 0 ]]; then
    echo ""
    echo " ERREUR : impossible de préparer l'environnement."
    echo " Vérifiez votre connexion internet puis relancez."
    echo ""
    read -p " Appuyez sur Entrée pour fermer..." _
    exit 1
fi
echo " [2/3] Environnement OK."
echo ""

# ── Version ───────────────────────────────────────────────────────
VERSION=$(uv run python -c "from versioning import get_full_label; print(get_full_label())" 2>/dev/null)
echo " [3/3] Version : $VERSION"
echo ""
echo " ============================================================"
echo " Lancement..."
echo " ============================================================"
echo ""

uv run app.py
