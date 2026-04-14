"""
Ollama Benchmark — core logic
Network scanning, Ollama API client, benchmark runner (text + vision + web + invoice).
"""

import base64
import json
import re
import socket
import subprocess
import sys
import platform as _platform
import requests
import concurrent.futures
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from datetime import datetime

OLLAMA_PORT = 11434
SCAN_TIMEOUT = 0.5
REQUEST_TIMEOUT = 300

# ── Default prompts (fallback when tests.json is missing / malformed) ──────────
_DEFAULT_TEXT_QUESTIONS: List[Tuple[str, str]] = [
    ("cold", "What is the difference between efficiency and efficacy?"),
    ("hot",  "In one sentence, explain what Occam's razor is."),
    ("long", "Explain step by step how an HTTP request works: from typing a URL in the "
             "browser to the page being displayed. Cover DNS resolution, TCP handshake, "
             "HTTP headers, server processing, and rendering. Be thorough."),
]
_DEFAULT_VISION_QUESTION  = "Describe this image in detail."
_DEFAULT_WEB_QUESTION     = "What is the current price of gold per gram in USD and EUR?"
_DEFAULT_INVOICE_PROMPT   = """\
You are an accounting data extraction assistant.
Analyze the text of this invoice and return ONLY a valid JSON object.
No text before, no text after, no markdown block.

Expected JSON structure (respect the exact keys):
{{
  "fournisseur": "vendor company name",
  "client": "client company name",
  "date": "DD/MM/YYYY",
  "numero_facture": "invoice number",
  "lignes": [
    {{"description": "...", "quantite": 1, "prix_unitaire": 0.00, "montant": 0.00}}
  ],
  "sous_total_ht": 0.00,
  "tva_taux": 0.00,
  "tva_montant": 0.00,
  "total_ttc": 0.00,
  "mode_reglement": "...",
  "regime_special": "special VAT regime text if applicable, otherwise null"
}}

Invoice text:
{text}
"""

# ── File paths ─────────────────────────────────────────────────────────────────
DEFAULT_REFERENCE_IMAGE   = Path(__file__).parent / "goldorak.jpg"
DEFAULT_REFERENCE_INVOICE = Path(__file__).parent / "sample_invoice.pdf"
_TESTS_FILE               = Path(__file__).parent / "tests.json"
PDF_EXTENSIONS            = {".pdf"}
IMAGE_EXTENSIONS          = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Tests config ─────────────────────────────────────────────────────────────

@dataclass
class TestsConfig:
    """Loaded benchmark prompts — editable via tests.json."""
    text: List[Tuple[str, str]]   # [(id, prompt), ...]
    vision_prompt: str
    web_prompt: str
    invoice_prompt: str
    source: str = "defaults"      # "tests.json" | "defaults"

    @property
    def text_ids(self) -> tuple:
        return tuple(t[0] for t in self.text)


def load_tests(path: Optional[Path] = None) -> TestsConfig:
    """Load tests.json; fall back to built-in defaults on any error."""
    target = path or _TESTS_FILE
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        text = [
            (str(t["id"]), str(t["prompt"]))
            for t in raw.get("text", [])
            if "id" in t and "prompt" in t
        ]
        if not text:
            raise ValueError("No text tests defined in tests.json")
        return TestsConfig(
            text=text,
            vision_prompt=raw.get("vision", {}).get("prompt", _DEFAULT_VISION_QUESTION),
            web_prompt=raw.get("web", {}).get("prompt", _DEFAULT_WEB_QUESTION),
            invoice_prompt=raw.get("invoice", {}).get("prompt", _DEFAULT_INVOICE_PROMPT),
            source=str(target.name),
        )
    except Exception:
        return TestsConfig(
            text=_DEFAULT_TEXT_QUESTIONS,
            vision_prompt=_DEFAULT_VISION_QUESTION,
            web_prompt=_DEFAULT_WEB_QUESTION,
            invoice_prompt=_DEFAULT_INVOICE_PROMPT,
            source="defaults",
        )


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    model: str
    host: str
    test_type: str          # "froid" | "chaud" | "vision" | "web"
    question: str
    response_text: str
    total_duration_ms: float
    load_duration_ms: float
    prompt_eval_count: int
    eval_count: int
    tokens_per_second: float
    context_injected: str = ""   # données injectées pour le test web
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelSummary:
    model: str
    host: str
    results: List[BenchmarkResult] = field(default_factory=list)
    error: Optional[str] = None
    size_bytes: int = 0       # taille totale du modèle chargé
    size_vram_bytes: int = 0  # portion effectivement en VRAM GPU

    @property
    def cpu_offload_bytes(self) -> int:
        return max(0, self.size_bytes - self.size_vram_bytes)

    @property
    def is_offloading(self) -> bool:
        return self.size_bytes > 0 and self.size_vram_bytes < self.size_bytes

    @property
    def vram_fill_pct(self) -> float:
        if self.size_bytes == 0:
            return 0.0
        return min(1.0, self.size_vram_bytes / self.size_bytes)

    _SPECIAL_TYPES = {"vision", "web", "invoice"}

    def _by_type(self, *types: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.test_type in types]

    @property
    def _text_results(self) -> List[BenchmarkResult]:
        """All results that are not vision / web / invoice (i.e. text tests)."""
        return [r for r in self.results if r.test_type not in self._SPECIAL_TYPES]

    @property
    def avg_tokens_per_second(self) -> float:
        rs = self._text_results
        valid = [r for r in rs if r.tokens_per_second > 0]
        return sum(r.tokens_per_second for r in valid) / len(valid) if valid else 0.0

    @property
    def avg_total_ms(self) -> float:
        rs = self._text_results
        return sum(r.total_duration_ms for r in rs) / len(rs) if rs else 0.0

    @property
    def sum_total_ms(self) -> float:
        """Total wall-clock time for all text tests."""
        return sum(r.total_duration_ms for r in self._text_results)

    @property
    def avg_load_ms(self) -> float:
        rs = self._text_results
        return sum(r.load_duration_ms for r in rs) / len(rs) if rs else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.eval_count for r in self.results)

    @property
    def total_response_chars(self) -> int:
        """Total characters in text responses — brevity indicator."""
        return sum(len(r.response_text) for r in self._text_results)

    @property
    def vision_result(self) -> Optional[BenchmarkResult]:
        rs = self._by_type("vision")
        return rs[0] if rs else None

    @property
    def vision_supported(self) -> Optional[bool]:
        vr = self.vision_result
        if vr is None:
            return None
        return vr.eval_count > 0

    @property
    def web_result(self) -> Optional[BenchmarkResult]:
        rs = self._by_type("web")
        return rs[0] if rs else None

    @property
    def invoice_result(self) -> Optional[BenchmarkResult]:
        rs = self._by_type("invoice")
        return rs[0] if rs else None


# ─── VRAM Status ──────────────────────────────────────────────────────────────

@dataclass
class VramStatus:
    """Snapshot of local GPU VRAM with per-app breakdown."""
    available: bool = False
    platform_name: str = ""       # "windows_nvidia" | "macos_apple_silicon" | ""
    gpu_name: str = ""
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    ollama_mb: int = 0            # VRAM held by currently loaded Ollama models
    # Legacy aggregate fields (kept for UI compat)
    browser_procs: int = 0
    browser_mb_est: int = 0
    electron_procs: int = 0
    electron_mb_est: int = 0
    cleanup_gain_mb: int = 0      # sum of all app estimates
    free_after_mb: int = 0        # free + cleanup_gain (capped at total)
    # Per-app breakdown: {display_name: estimated_mb}
    app_breakdown: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def free_pct(self) -> float:
        return self.free_mb / self.total_mb if self.total_mb > 0 else 0.0

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024.0

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024.0

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024.0

    @property
    def free_after_gb(self) -> float:
        return self.free_after_mb / 1024.0


# ─── Scan images / PDFs ───────────────────────────────────────────────────────

def scan_images(folder: Optional[Path] = None) -> List[str]:
    """Return sorted list of image filenames in the project folder."""
    folder = folder or Path(__file__).parent
    return sorted(
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def scan_pdfs(folder: Optional[Path] = None) -> List[str]:
    """Return sorted list of PDF filenames in the project folder."""
    folder = folder or Path(__file__).parent
    return sorted(
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in PDF_EXTENSIONS
    )


# ─── Network scanning ─────────────────────────────────────────────────────────

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _check_ollama(ip: str) -> Optional[str]:
    try:
        r = requests.get(
            f"http://{ip}:{OLLAMA_PORT}/api/tags",
            timeout=SCAN_TIMEOUT,
        )
        if r.status_code == 200:
            return ip
    except Exception:
        pass
    return None


def scan_network(
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> List[str]:
    local_ip = get_local_ip()
    prefix = ".".join(local_ip.split(".")[:3])
    candidates: List[str] = [f"{prefix}.{i}" for i in range(1, 255)]
    if "127.0.0.1" not in candidates:
        candidates.insert(0, "127.0.0.1")

    found: List[str] = []
    total = len(candidates)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as exe:
        futures = {exe.submit(_check_ollama, ip): ip for ip in candidates}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if progress_cb:
                progress_cb(completed / total, f"Scan {prefix}.x — {completed}/{total}")
            result = future.result()
            if result:
                found.append(result)

    def _sort_key(ip: str):
        if ip == "127.0.0.1":
            return (0, 0, 0, 0)
        try:
            return tuple(int(p) for p in ip.split("."))
        except Exception:
            return (999, 999, 999, 999)

    return sorted(found, key=_sort_key)


# ─── Ollama API ───────────────────────────────────────────────────────────────

def get_models(host: str) -> List[dict]:
    r = requests.get(f"http://{host}:{OLLAMA_PORT}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def delete_model(host: str, model_name: str) -> None:
    """Delete a model from local Ollama storage (equivalent to ollama rm)."""
    r = requests.delete(
        f"http://{host}:{OLLAMA_PORT}/api/delete",
        json={"model": model_name},
        timeout=30,
    )
    r.raise_for_status()


def unload_model(host: str, model_name: str) -> None:
    """Force Ollama to evict a model from VRAM immediately."""
    try:
        requests.post(
            f"http://{host}:{OLLAMA_PORT}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
    except Exception:
        pass


def get_model_vram_info(host: str, model_name: str) -> Tuple[int, int]:
    """Query /api/ps for a loaded model. Returns (size_bytes, size_vram_bytes)."""
    try:
        r = requests.get(f"http://{host}:{OLLAMA_PORT}/api/ps", timeout=5)
        r.raise_for_status()
        for m in r.json().get("models", []):
            name = m.get("name", "") or m.get("model", "")
            if name == model_name or name.split(":")[0] == model_name.split(":")[0]:
                return int(m.get("size", 0)), int(m.get("size_vram", 0))
    except Exception:
        pass
    return 0, 0


def get_ollama_version(host: str) -> str:
    try:
        r = requests.get(f"http://{host}:{OLLAMA_PORT}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version", "?")
    except Exception:
        return "?"


# ─── VRAM monitoring ──────────────────────────────────────────────────────────

def _run_cmd(cmd: List[str], timeout: int = 8) -> str:
    """Run a subprocess silently, return stdout or '' on any error."""
    try:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            # Windows console commands use the OEM code page — decode safely
            "encoding": "mbcs" if sys.platform == "win32" else "utf-8",
            "errors": "replace",
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        return subprocess.run(cmd, **kwargs).stdout.strip()
    except Exception:
        return ""


def get_vram_status(ollama_host: str = "127.0.0.1") -> VramStatus:
    """Snapshot local GPU VRAM with browser/app breakdown."""
    status = VramStatus()
    plat = _platform.system()
    if plat == "Windows":
        _vram_windows(status, ollama_host)
    elif plat == "Darwin":
        _vram_macos(status)
    else:
        status.error = f"VRAM monitoring not supported on {plat}"
    return status


def _vram_windows(status: VramStatus, host: str) -> None:
    # 1 ── nvidia-smi GPU summary ───────────────────────────────────────────
    out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        status.error = "nvidia-smi not found — install/update NVIDIA drivers"
        return
    try:
        parts = [p.strip() for p in out.split(",")]
        status.gpu_name      = parts[0]
        status.total_mb      = int(parts[1])
        status.used_mb       = int(parts[2])
        status.free_mb       = int(parts[3])
        status.available     = True
        status.platform_name = "windows_nvidia"
    except Exception as exc:
        status.error = f"nvidia-smi parse error: {exc}"
        return

    # 2 ── Ollama loaded models VRAM (from /api/ps) ─────────────────────────
    try:
        r = requests.get(f"http://{host}:{OLLAMA_PORT}/api/ps", timeout=3)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                status.ollama_mb += int(m.get("size_vram", 0)) // (1024 * 1024)
    except Exception:
        pass

    # 3 ── Scan processes via tasklist ─────────────────────────────────────
    tl = _run_cmd(["tasklist", "/fo", "csv", "/nh"], timeout=10)
    if tl:
        # Browsers: base VRAM (GPU compositor) + small per extra proc, capped
        # Formula: base_mb + (count-1) × per_mb, capped at cap_mb
        #          (Chrome spawns dozens of renderer procs, only compositor matters)
        BROWSER_MAP: dict = {
            "chrome.exe":   ("Chrome",  50,  8, 800),
            "msedge.exe":   ("Edge",    50,  8, 800),
            "firefox.exe":  ("Firefox", 80,  5, 600),
            "brave.exe":    ("Brave",   50,  8, 700),
            "opera.exe":    ("Opera",   50,  8, 700),
        }
        # Electron / comms: fixed estimate per running app (not per process)
        ELECTRON_MAP: dict = {
            "ms-teams.exe":  ("Teams",   300),
            "teams.exe":     ("Teams",   300),
            "discord.exe":   ("Discord", 200),
            "slack.exe":     ("Slack",   200),
            "notion.exe":    ("Notion",  150),
            "code.exe":      ("VS Code", 150),
            "spotify.exe":   ("Spotify", 120),
            "zoom.exe":      ("Zoom",    200),
            "figma.exe":     ("Figma",   200),
        }
        # Creative / streaming / gaming: heavy GPU consumers
        CREATIVE_MAP: dict = {
            "obs64.exe":              ("OBS Studio",     400),
            "obs32.exe":              ("OBS Studio",     400),
            "photoshop.exe":          ("Photoshop",      400),
            "afterfx.exe":            ("After Effects",  500),
            "resolve.exe":            ("DaVinci Resolve",800),
            "steam.exe":              ("Steam",          100),
            "nvidia broadcast.exe":   ("NVIDIA Broadcast",500),
            "gamebar.exe":            ("Xbox Game Bar",   80),
        }

        browser_counts: dict  = {}   # exe → process count
        electron_seen:  set   = set()  # display names already counted
        creative_seen:  set   = set()

        for line in tl.lower().splitlines():
            try:
                exe = line.split('"')[1]
            except IndexError:
                continue
            if exe in BROWSER_MAP:
                browser_counts[exe] = browser_counts.get(exe, 0) + 1
            elif exe in ELECTRON_MAP:
                label, _ = ELECTRON_MAP[exe]
                electron_seen.add(label)   # deduplicate (ms-teams + teams = same app)
            elif exe in CREATIVE_MAP:
                label, _ = CREATIVE_MAP[exe]
                creative_seen.add(label)

        # ── Browser estimates ──────────────────────────────────────────────
        browser_parts: List[str] = []
        for exe, cnt in browser_counts.items():
            label, base, per, cap = BROWSER_MAP[exe]
            mb = min(cap, base + (cnt - 1) * per)
            status.browser_mb_est += mb
            status.browser_procs  += cnt
            status.app_breakdown[label] = status.app_breakdown.get(label, 0) + mb
            browser_parts.append(f"{label} ~{mb:,} MB")

        # ── Electron estimates ─────────────────────────────────────────────
        electron_parts: List[str] = []
        for label in sorted(electron_seen):
            # Find MB from map (use first matching key)
            mb = next(v for k, (n, v) in ELECTRON_MAP.items() if n == label)
            status.electron_mb_est  += mb
            status.electron_procs   += 1
            status.app_breakdown[label] = mb
            electron_parts.append(f"{label} ~{mb:,} MB")

        # ── Creative estimates ─────────────────────────────────────────────
        creative_total = 0
        creative_parts: List[str] = []
        for label in sorted(creative_seen):
            mb = next(v for k, (n, v) in CREATIVE_MAP.items() if n == label)
            creative_total += mb
            status.app_breakdown[label] = mb
            creative_parts.append(f"{label} ~{mb:,} MB")

    # 4 ── Derived values ──────────────────────────────────────────────────
    total_app_mb = sum(status.app_breakdown.values())
    status.cleanup_gain_mb = total_app_mb
    status.free_after_mb   = min(status.total_mb, status.free_mb + total_app_mb)

    # 5 ── Warnings ────────────────────────────────────────────────────────
    fp = status.free_pct
    if fp < 0.35:
        status.warnings.append(
            f"⚠ Only {status.free_gb:.1f} GB free — models will overflow to CPU"
        )
    elif fp < 0.55:
        status.warnings.append(
            f"⚡ {status.free_gb:.1f} GB free — large models may partially overflow"
        )
    if tl:
        if browser_parts:
            status.warnings.append(
                "🌐 " + "  ·  ".join(browser_parts) + " — disable GPU accel to free"
            )
        if electron_parts:
            status.warnings.append(
                "💬 " + "  ·  ".join(electron_parts)
            )
        if creative_parts:
            status.warnings.append(
                "🎬 " + "  ·  ".join(creative_parts)
            )


def _vram_macos(status: VramStatus) -> None:
    if _platform.machine() != "arm64":
        status.error = "Intel Mac: VRAM monitoring not implemented"
        return

    status.platform_name = "macos_apple_silicon"
    status.gpu_name      = "Apple Silicon (unified memory)"

    # Total RAM
    out = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if out:
        try:
            status.total_mb = int(out) // (1024 * 1024)
        except Exception:
            pass

    # Available memory from vm_stat (Apple Silicon uses 16 KB pages)
    vm = _run_cmd(["vm_stat"])
    if vm and status.total_mb > 0:
        try:
            page_size = 16384

            def _pages(key: str) -> int:
                m = re.search(rf"{key}:\s+(\d+)", vm)
                return int(m.group(1)) if m else 0

            free_p      = _pages("Pages free")
            purgeable_p = _pages("Pages purgeable")
            inactive_p  = _pages("Pages inactive")
            # Available ≈ free + purgeable + half of inactive (reclaimable)
            avail_p = free_p + purgeable_p + inactive_p // 2
            status.free_mb = min(status.total_mb, (avail_p * page_size) // (1024 * 1024))
            status.used_mb = status.total_mb - status.free_mb
            status.available = True
        except Exception:
            pass

    status.free_after_mb = status.free_mb

    # Warn on memory pressure
    if status.available and status.free_pct < 0.25:
        status.warnings.append(
            f"⚠ Low unified memory: {status.free_gb:.1f} GB available — "
            "close apps before benchmarking"
        )


def _call_generate(host: str, payload: dict) -> Tuple[dict, float]:
    url = f"http://{host}:{OLLAMA_PORT}/api/generate"
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    t1 = time.perf_counter()
    resp.raise_for_status()
    return resp.json(), t1 - t0


def _parse_result(
    data: dict,
    wall_s: float,
    model: str,
    host: str,
    test_type: str,
    question: str,
    context_injected: str = "",
) -> BenchmarkResult:
    total_ns: int = data.get("total_duration", int(wall_s * 1e9))
    load_ns: int  = data.get("load_duration", 0)
    eval_count: int = data.get("eval_count", 0)
    eval_ns: int    = data.get("eval_duration", 1)
    tps = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0.0
    return BenchmarkResult(
        model=model,
        host=host,
        test_type=test_type,
        question=question,
        response_text=data.get("response", ""),
        total_duration_ms=total_ns / 1e6,
        load_duration_ms=load_ns / 1e6,
        prompt_eval_count=data.get("prompt_eval_count", 0),
        eval_count=eval_count,
        tokens_per_second=tps,
        context_injected=context_injected,
    )


# ─── Test texte ───────────────────────────────────────────────────────────────

def _benchmark_text(
    host: str, model: str, test_type: str, question: str,
    log_cb: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> BenchmarkResult:
    if log_cb:
        preview = question[:65] + ("…" if len(question) > 65 else "")
        log_cb(f"  ▶ [{test_type.upper()}]  {preview}")

    payload: dict = {"model": model, "prompt": question, "stream": False}
    if options:
        payload["options"] = options
    data, wall_s = _call_generate(host, payload)
    result = _parse_result(data, wall_s, model, host, test_type, question)

    if log_cb:
        log_cb(
            f"  ✓  {result.eval_count} tokens  |  {result.tokens_per_second:.1f} tok/s  |"
            f"  total: {result.total_duration_ms:.0f} ms  |  load: {result.load_duration_ms:.0f} ms"
        )
    return result


# ─── Test vision ──────────────────────────────────────────────────────────────

def _benchmark_vision(
    host: str, model: str, image_path: Path,
    vision_prompt: str = _DEFAULT_VISION_QUESTION,
    log_cb: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> BenchmarkResult:
    if log_cb:
        log_cb(f"  ▶ [VISION]  {vision_prompt}  ({image_path.name})")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload: dict = {
        "model": model,
        "prompt": vision_prompt,
        "images": [image_b64],
        "stream": False,
    }
    if options:
        payload["options"] = options
    data, wall_s = _call_generate(host, payload)
    result = _parse_result(data, wall_s, model, host, "vision", vision_prompt)

    if log_cb:
        if result.eval_count > 0:
            log_cb(
                f"  ✓  {result.eval_count} tokens  |  {result.tokens_per_second:.1f} tok/s  |"
                f"  total: {result.total_duration_ms:.0f} ms"
            )
            excerpt = result.response_text[:180].replace("\n", " ")
            log_cb(f"  💬  {excerpt}{'…' if len(result.response_text) > 180 else ''}")
        else:
            log_cb("  ⚠  Empty response — model is not multimodal")
    return result


# ─── Test web (cours de l'or) ─────────────────────────────────────────────────

def _fetch_gold_context(log_cb: Optional[Callable] = None) -> Tuple[str, str]:
    """
    Fetches real-time gold price (gold-api.com) and EUR/USD rate (frankfurter.app).
    Returns (context_string, status_message).
    """
    try:
        gold_resp = requests.get("https://api.gold-api.com/price/XAU", timeout=8)
        gold_resp.raise_for_status()
        gold_data = gold_resp.json()
        price_usd_oz = float(gold_data.get("price", 0))

        fx_resp = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=EUR", timeout=8
        )
        fx_resp.raise_for_status()
        eur_rate = float(fx_resp.json()["rates"]["EUR"])

        TROY_OZ_TO_GRAM = 31.1035
        price_eur_oz    = price_usd_oz * eur_rate
        price_eur_gram  = price_eur_oz / TROY_OZ_TO_GRAM
        price_usd_gram  = price_usd_oz / TROY_OZ_TO_GRAM
        today           = datetime.now().strftime("%d/%m/%Y %H:%M")

        context = (
            f"[Données de marché en temps réel — {today}]\n"
            f"- Or (XAU) : {price_usd_oz:.2f} USD / once troy\n"
            f"- Or (XAU) : {price_eur_oz:.2f} EUR / once troy\n"
            f"- Or (XAU) : {price_usd_gram:.3f} USD / gramme\n"
            f"- Or (XAU) : {price_eur_gram:.3f} EUR / gramme\n"
            f"- Taux EUR/USD : {eur_rate:.4f}\n"
            f"Source : gold-api.com + frankfurter.app"
        )
        status = f"Or: {price_eur_gram:.2f} EUR/g  |  {price_eur_oz:.2f} EUR/oz  (1 USD = {eur_rate:.4f} EUR)"
        if log_cb:
            log_cb(f"  📈  Cours récupéré : {status}")
        return context, status

    except Exception as exc:
        msg = f"Failed to fetch gold price: {exc}"
        if log_cb:
            log_cb(f"  ⚠  {msg}")
        return "", msg


def _benchmark_web(
    host: str, model: str,
    web_prompt: str = _DEFAULT_WEB_QUESTION,
    log_cb: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> BenchmarkResult:
    if log_cb:
        log_cb(f"  ▶ [WEB]  {web_prompt}")

    context, status = _fetch_gold_context(log_cb)

    if context:
        prompt = (
            f"{context}\n\n"
            f"Based only on the data above, answer clearly and concisely:\n"
            f"{web_prompt}"
        )
    else:
        # No live data — send raw question, model will have to rely on training
        prompt = web_prompt
        if log_cb:
            log_cb("  ⚠  No live data — sending raw question")

    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    data, wall_s = _call_generate(host, payload)
    result = _parse_result(data, wall_s, model, host, "web", web_prompt, context_injected=status)

    if log_cb:
        log_cb(
            f"  ✓  {result.eval_count} tokens  |  {result.tokens_per_second:.1f} tok/s  |"
            f"  total: {result.total_duration_ms:.0f} ms"
        )
        excerpt = result.response_text[:200].replace("\n", " ")
        log_cb(f"  💬  {excerpt}{'…' if len(result.response_text) > 200 else ''}")

    return result


# ─── Test facture PDF ─────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from all pages of a PDF using PyMuPDF."""
    import pymupdf  # lazy import — optional dependency
    doc = pymupdf.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def _parse_invoice_json(raw: str) -> Optional[dict]:
    """Try to extract a JSON object from model response (handles markdown fences)."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    # Find first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        return None


def _check_invoice_accuracy(data: dict) -> Tuple[int, str]:
    """
    Verify arithmetic consistency of extracted invoice data.
    Returns (score 0-2, human-readable detail).
    """
    checks = []
    score = 0

    # Check 1 : sum of line amounts ≈ sous_total_ht
    try:
        lines_sum = sum(float(l.get("montant", 0)) for l in data.get("lignes", []))
        ht = float(data.get("sous_total_ht", 0))
        if ht > 0 and abs(lines_sum - ht) <= max(0.05, ht * 0.01):
            score += 1
            checks.append("✓ lignes=HT")
        else:
            checks.append(f"✗ lignes({lines_sum:.2f})≠HT({ht:.2f})")
    except Exception:
        checks.append("✗ lignes (parse err)")

    # Check 2 : sous_total_ht + tva_montant ≈ total_ttc
    try:
        ht  = float(data.get("sous_total_ht", 0))
        tva = float(data.get("tva_montant", 0))
        ttc = float(data.get("total_ttc", 0))
        if ttc > 0 and abs(ht + tva - ttc) <= 0.05:
            score += 1
            checks.append("✓ HT+TVA=TTC")
        else:
            checks.append(f"✗ HT+TVA({ht+tva:.2f})≠TTC({ttc:.2f})")
    except Exception:
        checks.append("✗ totaux (parse err)")

    return score, f"{score}/2  " + "  ".join(checks)


def _benchmark_invoice(
    host: str, model: str, pdf_path: Path,
    invoice_prompt: str = _DEFAULT_INVOICE_PROMPT,
    log_cb: Optional[Callable] = None,
    options: Optional[dict] = None,
) -> BenchmarkResult:
    if log_cb:
        log_cb(f"  ▶ [INVOICE]  {pdf_path.name}")

    # Extract text from PDF
    try:
        pdf_text = _extract_pdf_text(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read PDF: {exc}") from exc

    prompt = invoice_prompt.replace("{text}", pdf_text)
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    data, wall_s = _call_generate(host, payload)
    result = _parse_result(data, wall_s, model, host, "invoice", pdf_path.name)

    # Verify accuracy
    parsed = _parse_invoice_json(result.response_text)
    if parsed:
        score, detail = _check_invoice_accuracy(parsed)
        result.context_injected = detail
        if log_cb:
            log_cb(
                f"  ✓  {result.eval_count} tokens  |  {result.tokens_per_second:.1f} tok/s  |"
                f"  total: {result.total_duration_ms:.0f} ms  |  accuracy: {detail}"
            )
    else:
        result.context_injected = "⚠ invalid JSON"
        if log_cb:
            log_cb(f"  ⚠  Response is not valid JSON")

    return result


# ─── Main benchmark orchestrator ─────────────────────────────────────────────

def run_benchmark(
    host: str,
    models: List[str],
    include_vision: bool = False,
    vision_image: Optional[Path] = None,
    include_web: bool = False,
    include_invoice: bool = False,
    invoice_path: Optional[Path] = None,
    log_cb: Optional[Callable] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    stop_event=None,
    num_ctx: int = 0,
    num_predict: int = -1,
    num_gpu: int = -1,
    tests: Optional[TestsConfig] = None,
) -> List[ModelSummary]:
    if tests is None:
        tests = load_tests()

    gen_options: Optional[dict] = {}
    if num_ctx > 0:
        gen_options["num_ctx"] = num_ctx
    if num_predict > 0:
        gen_options["num_predict"] = num_predict
    if num_gpu >= 0:
        gen_options["num_gpu"] = num_gpu
    gen_options = gen_options if gen_options else None

    extra = (1 if include_vision else 0) + (1 if include_web else 0) + (1 if include_invoice else 0)
    steps_per_model = len(tests.text) + extra
    total_steps = len(models) * steps_per_model
    completed = 0

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    def _tick():
        nonlocal completed
        completed += 1
        if progress_cb:
            progress_cb(completed / total_steps)

    summaries: List[ModelSummary] = []

    for model in models:
        if _stopped():
            break

        if log_cb:
            log_cb(f"\n{'─' * 55}")
            log_cb(f"  Model: {model}")
            log_cb(f"{'─' * 55}")

        summary = ModelSummary(model=model, host=host)

        # ── Text tests ────────────────────────────────────────────────────
        first_text = True
        for test_id, question in tests.text:
            if _stopped():
                break
            try:
                result = _benchmark_text(host, model, test_id, question, log_cb, gen_options)
                summary.results.append(result)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  ✗  [{test_id.upper()}] Error: {exc}")
                summary.error = str(exc)
            # After first test the model is in memory — query VRAM
            if first_text:
                first_text = False
                size, size_vram = get_model_vram_info(host, model)
                summary.size_bytes = size
                summary.size_vram_bytes = size_vram
                if log_cb and size > 0:
                    if size_vram < size:
                        offload_gb = (size - size_vram) / 1_073_741_824
                        vram_gb    = size_vram / 1_073_741_824
                        total_gb   = size / 1_073_741_824
                        pct        = int(size_vram / size * 100)
                        log_cb(f"  ⚠  VRAM overflow — {vram_gb:.1f}/{total_gb:.1f} GB on GPU ({pct}%) · {offload_gb:.1f} GB on CPU")
                    else:
                        vram_gb = size_vram / 1_073_741_824
                        log_cb(f"  ✓  100% in VRAM ({vram_gb:.1f} GB)")
            _tick()

        # ── Vision ────────────────────────────────────────────────────────
        if include_vision and not _stopped():
            if vision_image and vision_image.is_file():
                try:
                    result = _benchmark_vision(host, model, vision_image, tests.vision_prompt, log_cb, gen_options)
                    summary.results.append(result)
                except Exception as exc:
                    if log_cb:
                        log_cb(f"  ✗  [VISION] Error: {exc}")
            else:
                if log_cb:
                    log_cb("  ⚠  [VISION] Image not found — test skipped")
            _tick()

        # ── Web ───────────────────────────────────────────────────────────
        if include_web and not _stopped():
            try:
                result = _benchmark_web(host, model, tests.web_prompt, log_cb, gen_options)
                summary.results.append(result)
            except Exception as exc:
                if log_cb:
                    log_cb(f"  ✗  [WEB] Error: {exc}")
            _tick()

        # ── Invoice PDF ───────────────────────────────────────────────────
        if include_invoice and not _stopped():
            if invoice_path and invoice_path.is_file():
                try:
                    result = _benchmark_invoice(host, model, invoice_path, tests.invoice_prompt, log_cb, gen_options)
                    summary.results.append(result)
                except Exception as exc:
                    if log_cb:
                        log_cb(f"  ✗  [INVOICE] Error: {exc}")
            else:
                if log_cb:
                    log_cb("  ⚠  [INVOICE] PDF not found — test skipped")
            _tick()

        # Unload model from VRAM before testing the next one
        unload_model(host, model)
        if log_cb:
            log_cb(f"  ↩  Unloaded {model} from VRAM")

        summaries.append(summary)

    return summaries


# ─── Export ───────────────────────────────────────────────────────────────────

def export_report(summaries: List[ModelSummary], filepath: str, ollama_version: str = "?") -> None:
    lines = [
        f"# Ollama Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Ollama v{ollama_version}",
        "",
    ]
    ranked = sorted(summaries, key=lambda s: s.avg_tokens_per_second, reverse=True)

    for rank, s in enumerate(ranked, 1):
        vram_line = ""
        if s.size_bytes > 0:
            vram_gb  = s.size_vram_bytes / 1_073_741_824
            total_gb = s.size_bytes / 1_073_741_824
            if s.is_offloading:
                offload_gb = s.cpu_offload_bytes / 1_073_741_824
                vram_line = f"- **⚠ CPU offload** — {vram_gb:.1f}/{total_gb:.1f} GB in VRAM · {offload_gb:.1f} GB on CPU"
            else:
                vram_line = f"- VRAM: {vram_gb:.1f} GB (100% in GPU)"
        lines += [
            f"## #{rank} — {s.model}  (@ {s.host})",
            f"- **{s.avg_tokens_per_second:.2f} tok/s** (text avg)",
            f"- Avg time: {s.avg_total_ms:.0f} ms  |  load {s.avg_load_ms:.0f} ms",
            f"- Total tokens: {s.total_tokens}",
        ]
        if vram_line:
            lines.append(vram_line)
        lines.append("")
        for r in s.results:
            lines += [
                f"### [{r.test_type.upper()}] {r.question}",
                f"- {r.eval_count} tokens  |  {r.tokens_per_second:.2f} tok/s"
                f"  |  {r.total_duration_ms:.0f} ms  |  load {r.load_duration_ms:.0f} ms",
            ]
            if r.context_injected:
                lines.append(f"- Context: {r.context_injected}")
            excerpt = r.response_text[:400] + ("…" if len(r.response_text) > 400 else "")
            lines += [f"- Response: {excerpt}", ""]

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
