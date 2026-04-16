@echo off
cd /d "%~dp0"

REM ── Ollama Benchmark — Windows launcher ────────────────────────────────────
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q4_0

REM ── Install uv if missing ───────────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo   Installing uv...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    REM Reload PATH so uv is found immediately
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

echo.
for /f "tokens=*" %%i in ('uv --version') do echo   uv       : %%i
echo.

REM ── Sync env (recrée le venv si cassé, installe les deps manquantes) ────────
uv sync --quiet

uv run python -c "from versioning import get_full_label; print('  ' + get_full_label())"
echo.

uv run app.py
pause
