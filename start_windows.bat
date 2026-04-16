@echo off
title Ollama Benchmark — Démarrage
cd /d "%~dp0"

echo.
echo  ============================================================
echo    Ollama Benchmark — by obviousidea
echo  ============================================================
echo.

REM ── Variables de performance Ollama ─────────────────────────────────────────
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q4_0

REM ── Installation de uv si absent ────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo  [1/3] Installation de uv ^(gestionnaire de paquets^)...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    echo  [1/3] uv installe.
) else (
    echo  [1/3] uv detecte : OK
)
echo.

REM ── Vérification et réparation de l'environnement Python ────────────────────
echo  [2/3] Verification de l'environnement Python...
uv sync --quiet
if errorlevel 1 (
    echo.
    echo  ERREUR : impossible de preparer l'environnement.
    echo  Verifiez votre connexion internet puis relancez.
    echo.
    pause
    exit /b 1
)
echo  [2/3] Environnement OK.
echo.

REM ── Affichage de la version ──────────────────────────────────────────────────
for /f "tokens=*" %%i in ('uv run python -c "from versioning import get_full_label; print(get_full_label())"') do (
    echo  [3/3] Version : %%i
)
echo.
echo  ============================================================
echo  Lancement...
echo  ============================================================
echo.

uv run app.py
pause
