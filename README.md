# ⚡ Ollama Benchmark

A desktop GUI to benchmark local LLM models running on [Ollama](https://ollama.com) — compare speed, quality and VRAM usage across models and machines on your network.

Built with Python + CustomTkinter. No server, no cloud, runs fully offline.

---

## Features

### Benchmark tests
- **Text tests** — cold start, hot (cached), long-form generation — fully customizable via `tests.json`
- **Vision test** — image description for multimodal models
- **Web test** — real-time data fetch (live gold price) + LLM synthesis
- **Invoice extraction** — structured JSON parsing from a PDF, with arithmetic accuracy check

### Results & reporting
- **Ranked results** — models sorted by tok/s with speed, conciseness and duration bars
- **Per-test detail** — tokens, tok/s, duration for each individual question
- **Ollama version** — displayed and logged per host
- **Markdown reports** — auto-exported after each benchmark session

### VRAM monitor
- Live GPU memory bar (NVIDIA on Windows, unified memory on Apple Silicon)
- **Per-app breakdown** — browsers (Chrome, Edge, Firefox…), Electron apps (Teams, Discord, VS Code, Zoom, Slack…), creative tools (OBS, Photoshop, After Effects, DaVinci Resolve, NVIDIA Broadcast…)
- Estimated reclaimable VRAM displayed per app
- CPU offload detection — warns when a model overflows VRAM

### Infrastructure
- **LAN scan** — auto-discover all Ollama instances on your local network
- **Customizable prompts** — edit `tests.json` to add, remove or rewrite any test
- **Hot reload** — reload `tests.json` without restarting the app
- **Model parameters** — configure `num_ctx`, `num_predict`, GPU layers per run
- **Session logs** — timestamped `.log` + `.md` saved in `logs/`

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (auto-installed by the launcher if missing)
- [Ollama](https://ollama.com) running locally or on the network

---

## Quick start

**Windows**
```bat
run.bat
```

**macOS / Linux**
```bash
chmod +x run.sh
./run.sh
```

**Universal (any platform)**
```bash
python start.py
```

Dependencies are installed automatically on first run via `uv`.

---

## Manual install

```bash
pip install uv
uv run app.py
```

---

## Customizing tests

Edit `tests.json` to change any prompt or add new text tests:

```json
{
  "text": [
    { "id": "cold", "label": "Cold start", "prompt": "What is the difference between efficiency and efficacy?" },
    { "id": "hot",  "label": "Hot",        "prompt": "In one sentence, explain what Occam's razor is." },
    { "id": "long", "label": "Long",       "prompt": "Explain step by step how an HTTP request works..." },
    { "id": "code", "label": "Coding",     "prompt": "Write a Python function that checks if a number is prime." }
  ],
  "vision": { "prompt": "Describe this image in detail." },
  "web":    { "prompt": "What is the current price of gold per gram in USD and EUR?" }
}
```

Click **↺ Reload tests** in the sidebar to apply changes without restarting.

---

## Project structure

```
app.py              — UI (CustomTkinter)
core.py             — benchmark engine, network scan, VRAM monitor
versioning.py       — semantic versioning helpers
bump.py             — CLI tool to bump major/minor/patch
start.py            — cross-platform launcher
run.bat             — Windows launcher (sets Ollama env vars)
run.sh              — macOS/Linux launcher
tests.json          — editable benchmark prompts
sample_invoice.pdf  — sample PDF for invoice extraction test
goldorak.jpg        — sample image for vision test
pyproject.toml      — dependencies
```

---

## Bump version

```bash
python bump.py patch   # 1.1.0 → 1.1.1
python bump.py minor   # 1.1.0 → 1.2.0
python bump.py major   # 1.1.0 → 2.0.0
```

---

## Notes

- VRAM monitoring requires **NVIDIA drivers** on Windows (`nvidia-smi`)
- On macOS Apple Silicon, unified memory is monitored via `vm_stat`
- Linux VRAM monitoring is not yet implemented
- `selections.json` (model selections per host) is auto-created and gitignored
- `logs/` directory is auto-created and gitignored

---

## License

MIT — see [LICENSE](LICENSE)

Made by [obviousidea](https://obviousidea.com)
