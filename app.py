"""
Ollama Benchmark — by obviousidea
UI built with CustomTkinter.
"""

import json
import socket
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from versioning import increment_build, get_full_label
from core import (
    ModelSummary,
    VramStatus,
    TestsConfig,
    DEFAULT_REFERENCE_IMAGE,
    DEFAULT_REFERENCE_INVOICE,
    load_tests,
    get_models,
    delete_model,
    get_ollama_version,
    get_vram_status,
    run_benchmark,
    scan_network,
    parse_scan_targets,
    scan_images,
    scan_pdfs,
    export_report,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_PROJECT_DIR    = Path(__file__).parent
_SELECTIONS_FILE = _PROJECT_DIR / "selections.json"


# ─── Reusable widgets ─────────────────────────────────────────────────────────

class Separator(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=1, fg_color=("gray70", "gray30"), **kwargs)


class ModelCheckbox(ctk.CTkFrame):
    def __init__(self, parent, model_name: str, size_label: str = "",
                 on_delete=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.model_name = model_name
        self.var = ctk.BooleanVar(value=False)
        self._on_delete = on_delete

        self.grid_columnconfigure(0, weight=1)

        label = model_name if not size_label else f"{model_name}  ({size_label})"
        ctk.CTkCheckBox(
            self, text=label, variable=self.var,
            font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            self, text="🗑", width=26, height=22,
            fg_color="transparent",
            hover_color="#8b0000",
            text_color=("gray50", "gray55"),
            font=ctk.CTkFont(size=13),
            command=self._ask_delete,
        ).grid(row=0, column=1, padx=(4, 0))

    def _ask_delete(self):
        if self._on_delete:
            self._on_delete(self.model_name)


class ResultCard(ctk.CTkFrame):
    _RANK_COLORS = {1: ("#FFD700", "#1a1a1a"), 2: ("#C0C0C0", "#1a1a1a"), 3: ("#CD7F32", "#1a1a1a")}
    _BAR_COLORS  = {1: "#2ecc71", 2: "#f39c12", 3: "#3498db"}

    def __init__(self, parent, rank: int, summary: ModelSummary,
                 max_tps: float, min_chars: int, min_duration: float, **kwargs):
        super().__init__(parent, corner_radius=10, **kwargs)
        self.grid_columnconfigure(1, weight=1)

        bg, fg = self._RANK_COLORS.get(rank, ("#4a4a6a", "#ffffff"))

        ctk.CTkLabel(
            self, text=f"#{rank}", width=52,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color=bg, text_color=fg, corner_radius=8,
        ).grid(row=0, column=0, rowspan=2, padx=12, pady=10, sticky="ns")

        ctk.CTkLabel(
            self, text=summary.model,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=(10, 0))

        ctk.CTkLabel(
            self, text=f"@ {summary.host}",
            font=ctk.CTkFont(size=11), text_color="gray", anchor="w",
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 5))

        stats = ctk.CTkFrame(self, fg_color="transparent")
        stats.grid(row=0, column=2, rowspan=2, padx=15, pady=5)

        badges = ""
        if summary.vision_supported is True:
            badges += "  👁"
        if summary.web_result is not None:
            badges += "  🌐"
        if summary.invoice_result is not None:
            badges += "  📄"

        ctk.CTkLabel(
            stats, text=f"{summary.avg_tokens_per_second:.1f} tok/s",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#2ecc71",
        ).pack(anchor="e")
        ctk.CTkLabel(
            stats,
            text=(
                f"moy. {summary.avg_total_ms:.0f} ms  ·  "
                f"load {summary.avg_load_ms:.0f} ms  ·  "
                f"{summary.total_tokens} tokens{badges}"
            ),
            font=ctk.CTkFont(size=10), text_color="gray",
        ).pack(anchor="e")

        metrics = ctk.CTkFrame(self, fg_color="transparent")
        metrics.grid(row=2, column=0, columnspan=3, padx=12, pady=(0, 4), sticky="ew")
        metrics.grid_columnconfigure(1, weight=1)

        def _metric_row(row_i, icon, label, bar_val, value_str, color):
            ctk.CTkLabel(
                metrics, text=f"{icon} {label}",
                font=ctk.CTkFont(size=9), text_color="gray", width=76, anchor="w",
            ).grid(row=row_i, column=0, padx=(4, 2), pady=1, sticky="w")
            b = ctk.CTkProgressBar(metrics, height=5, progress_color=color)
            b.grid(row=row_i, column=1, padx=2, pady=1, sticky="ew")
            b.set(max(0.0, min(1.0, bar_val)))
            ctk.CTkLabel(
                metrics, text=value_str,
                font=ctk.CTkFont(size=9), text_color="gray", width=80, anchor="e",
            ).grid(row=row_i, column=2, padx=(2, 4), pady=1, sticky="e")

        # ⚡ Speed tok/s — higher = better
        _metric_row(
            0, "⚡", "Speed",
            summary.avg_tokens_per_second / max_tps if max_tps > 0 else 0,
            f"{summary.avg_tokens_per_second:.1f} tok/s",
            self._BAR_COLORS.get(rank, "#5dade2"),
        )
        # 📝 Conciseness — fewer chars = bar closer to 1.0
        chars = summary.total_response_chars
        chars_str = f"{chars // 1000}k" if chars >= 1000 else f"{chars}"
        _metric_row(
            1, "📝", "Conciseness",
            min_chars / chars if chars > 0 else 0,
            f"{chars_str} chars",
            "#9b59b6",
        )
        # ⏱ Total time — lower = bar closer to 1.0
        dur = summary.sum_total_ms
        dur_str = f"{dur/1000:.1f} s" if dur >= 1000 else f"{dur:.0f} ms"
        _metric_row(
            2, "⏱", "Time",
            min_duration / dur if dur > 0 else 0,
            dur_str,
            "#e67e22",
        )
        # 🎮 VRAM — seulement si info disponible
        if summary.size_bytes > 0:
            vram_gb  = summary.size_vram_bytes / 1_073_741_824
            total_gb = summary.size_bytes / 1_073_741_824
            if summary.is_offloading:
                vram_str   = f"{vram_gb:.1f}/{total_gb:.1f} GB ({int(summary.vram_fill_pct*100)}%)"
                vram_color = "#e74c3c"
            else:
                vram_str   = f"{vram_gb:.1f} GB (100%)"
                vram_color = "#2ecc71"
            _metric_row(3, "🎮", "VRAM", summary.vram_fill_pct, vram_str, vram_color)

        # Per-question detail rows
        detail = ctk.CTkFrame(self, fg_color="transparent")
        detail.grid(row=3, column=0, columnspan=3, padx=12, pady=(0, 4), sticky="ew")
        for r in summary.results:
            if r.test_type in ("vision", "web", "invoice"):
                continue
            short_q = r.question[:60] + ("…" if len(r.question) > 60 else "")
            ctk.CTkLabel(
                detail,
                text=f"  [{r.test_type.upper()}]  {short_q}  →  "
                     f"{r.eval_count} tok  |  {r.tokens_per_second:.1f} tok/s  |  {r.total_duration_ms:.0f} ms",
                font=ctk.CTkFont(size=10), text_color=("gray40", "gray60"), anchor="w",
            ).pack(fill="x")

        # Vision excerpt
        vr = summary.vision_result
        if vr and vr.response_text:
            self._add_excerpt(vr, "👁", "#5dade2", row=4)

        # Web excerpt
        wr = summary.web_result
        if wr and wr.response_text:
            self._add_excerpt(wr, "🌐", "#f39c12", row=5)

        # Invoice excerpt
        ir = summary.invoice_result
        if ir and ir.response_text:
            self._add_invoice_excerpt(ir, row=6)

    def _add_invoice_excerpt(self, result, row: int):
        accuracy = result.context_injected or ""
        json_preview = result.response_text[:220].replace("\n", " ")
        if len(result.response_text) > 220:
            json_preview += "…"
        box = ctk.CTkFrame(self, corner_radius=6, fg_color=("gray88", "gray22"))
        box.grid(row=row, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="ew")
        if accuracy:
            ctk.CTkLabel(
                box, text=f"📄  {accuracy}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=("#b7770d", "#e67e22"), anchor="w",
            ).pack(anchor="w", padx=8, pady=(5, 1))
        ctk.CTkLabel(
            box, text=json_preview,
            font=ctk.CTkFont(family="Courier New", size=9),
            text_color=("gray30", "gray65"), anchor="w",
            wraplength=680, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 5))

    def _add_excerpt(self, result, icon: str, color: str, row: int):
        excerpt = result.response_text[:240].replace("\n", " ")
        if len(result.response_text) > 240:
            excerpt += "…"
        box = ctk.CTkFrame(self, corner_radius=6, fg_color=("gray88", "gray22"))
        box.grid(row=row, column=0, columnspan=3, padx=12, pady=(0, 6), sticky="ew")
        ctk.CTkLabel(
            box,
            text=f"{icon}  {excerpt}",
            font=ctk.CTkFont(size=10),
            text_color=("gray20", color),
            anchor="w", wraplength=680, justify="left",
        ).pack(anchor="w", padx=8, pady=5)


# ─── Main application ─────────────────────────────────────────────────────────

class OllamaBenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._build_number = increment_build()
        self._version_label = get_full_label()
        self.title(f"Ollama Benchmark — by obviousidea  [{self._version_label}]")
        self.geometry("1370x840")
        self.minsize(1020, 680)

        self._hosts: List[str] = []
        self._selected_host: Optional[str] = None
        self._model_checkboxes: List[ModelCheckbox] = []
        self._summaries: List[ModelSummary] = []
        self._scanning = False
        self._running = False
        self._stop_event = threading.Event()
        self._session_log_file = None
        self._session_log_path = None
        self._vram_status: Optional[VramStatus] = None
        self._ollama_version: str = "?"
        self._tests: TestsConfig = load_tests()

        self._build_layout()
        self._welcome()
        self.after(900, self._refresh_vram)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()

        sidebar_outer = ctk.CTkFrame(self, width=358, corner_radius=12)
        sidebar_outer.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="nsew")
        sidebar_outer.grid_propagate(False)
        sidebar_outer.grid_rowconfigure(0, weight=1)
        sidebar_outer.grid_columnconfigure(0, weight=1)
        # All sidebar content lives in a scrollable container
        sidebar_scroll = ctk.CTkScrollableFrame(sidebar_outer, corner_radius=0, fg_color="transparent")
        sidebar_scroll.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll.grid_columnconfigure(0, weight=1)
        self._build_sidebar(sidebar_scroll)

        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=1, column=1, padx=(5, 10), pady=(0, 10), sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        self._build_main(main)

    def _build_header(self):
        hdr = ctk.CTkFrame(self, height=52, corner_radius=0, fg_color=("gray85", "gray18"))
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)
        hdr.grid_columnconfigure(2, weight=0)

        ctk.CTkLabel(
            hdr, text="⚡  Ollama Benchmark",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=8, sticky="w")
        ctk.CTkLabel(
            hdr, text="by obviousidea",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).grid(row=0, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(
            hdr, text=self._version_label,
            font=ctk.CTkFont(size=10), text_color=("gray50", "gray55"),
        ).grid(row=0, column=2, padx=10, pady=8, sticky="w")

        self._theme_var = ctk.StringVar(value="dark")
        ctk.CTkSegmentedButton(
            hdr, values=["dark", "light"], variable=self._theme_var,
            command=lambda v: ctk.set_appearance_mode(v), width=130,
        ).grid(row=0, column=3, padx=16, pady=8, sticky="e")

    def _build_sidebar(self, p):
        """All controls — compact padding so everything fits without scrolling."""
        row = 0

        def _sec(text):
            nonlocal row
            ctk.CTkLabel(
                p, text=text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=("gray40", "gray60"),
            ).grid(row=row, column=0, padx=14, pady=(8, 2), sticky="w")
            row += 1

        def _sep():
            nonlocal row
            Separator(p).grid(row=row, column=0, padx=12, pady=5, sticky="ew")
            row += 1

        # ── 1: Scan ───────────────────────────────────────────────────────
        _sec("1 — Network discovery")
        self._scan_btn = ctk.CTkButton(p, text="Scan network", height=32, command=self._start_scan)
        self._scan_btn.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1

        self._scan_targets = ctk.CTkEntry(
            p, height=28, placeholder_text="Extra: 10.10.0.0/16, 10.10.13.1-50, 10.10.100.7")
        self._scan_targets.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1

        self._scan_bar = ctk.CTkProgressBar(p, height=4)
        self._scan_bar.grid(row=row, column=0, padx=12, pady=2, sticky="ew")
        self._scan_bar.set(0); row += 1

        self._scan_lbl = ctk.CTkLabel(p, text="", font=ctk.CTkFont(size=10), text_color="gray")
        self._scan_lbl.grid(row=row, column=0, padx=12, sticky="w"); row += 1
        _sep()

        # ── 2: Host ───────────────────────────────────────────────────────
        _sec("2 — Select instance")
        _host_wrap = ctk.CTkFrame(p, height=115, corner_radius=8)
        _host_wrap.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1
        _host_wrap.grid_propagate(False)
        _host_wrap.grid_rowconfigure(0, weight=1)
        _host_wrap.grid_columnconfigure(0, weight=1)
        self._host_frame = ctk.CTkScrollableFrame(_host_wrap, corner_radius=8, fg_color="transparent")
        self._host_frame.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(
            self._host_frame, text="Run a scan first",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(pady=6)
        _sep()

        # ── 3: Models ─────────────────────────────────────────────────────
        _sec("3 — Select models")
        _models_wrap = ctk.CTkFrame(p, height=170, corner_radius=8)
        _models_wrap.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1
        _models_wrap.grid_propagate(False)
        _models_wrap.grid_rowconfigure(0, weight=1)
        _models_wrap.grid_columnconfigure(0, weight=1)
        self._models_frame = ctk.CTkScrollableFrame(_models_wrap, corner_radius=8, fg_color="transparent")
        self._models_frame.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(
            self._models_frame, text="Select an instance first",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(pady=6)

        sel_row = ctk.CTkFrame(p, fg_color="transparent")
        sel_row.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1
        ctk.CTkButton(sel_row, text="All",  width=68, height=24, fg_color="gray35", hover_color="gray45", command=self._select_all).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sel_row, text="None", width=68, height=24, fg_color="gray35", hover_color="gray45", command=self._select_none).pack(side="left")
        _sep()

        # ── 4a: Vision ────────────────────────────────────────────────────
        _sec("4 — Optional tests")

        self._vision_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            p, text="Vision  (multimodal models)",
            variable=self._vision_var, font=ctk.CTkFont(size=12),
            command=self._refresh_img_status,
        ).grid(row=row, column=0, padx=14, pady=(0, 2), sticky="w"); row += 1

        # Combobox image + bouton refresh + browse
        img_row = ctk.CTkFrame(p, fg_color="transparent")
        img_row.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1
        img_row.grid_columnconfigure(0, weight=1)

        self._img_combo_values = self._get_image_list()
        default_img = DEFAULT_REFERENCE_IMAGE.name if DEFAULT_REFERENCE_IMAGE.is_file() else (self._img_combo_values[0] if self._img_combo_values else "")

        self._img_combo = ctk.CTkComboBox(
            img_row,
            values=self._img_combo_values if self._img_combo_values else ["(aucune image)"],
            font=ctk.CTkFont(size=11),
            height=28,
            command=lambda v: self._refresh_img_status(),
        )
        if default_img:
            self._img_combo.set(default_img)
        self._img_combo.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(img_row, text="↺", width=28, height=28, command=self._refresh_img_combo).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkButton(img_row, text="…", width=28, height=28, command=self._browse_image).grid(row=0, column=2)

        self._img_status = ctk.CTkLabel(p, text="", font=ctk.CTkFont(size=10), text_color="gray")
        self._img_status.grid(row=row, column=0, padx=14, pady=(0, 2), sticky="w"); row += 1
        self._refresh_img_status()

        # ── 4b: Web ───────────────────────────────────────────────────────
        self._web_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            p, text="Web  (live gold price)",
            variable=self._web_var, font=ctk.CTkFont(size=12),
        ).grid(row=row, column=0, padx=14, pady=(2, 0), sticky="w"); row += 1

        ctk.CTkLabel(
            p, text="gold-api.com + frankfurter.app",
            font=ctk.CTkFont(size=10), text_color=("gray50", "gray55"),
        ).grid(row=row, column=0, padx=26, pady=(0, 2), sticky="w"); row += 1

        # ── 4c: Invoice PDF ───────────────────────────────────────────────
        self._invoice_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            p, text="Invoice PDF  (JSON extraction)",
            variable=self._invoice_var, font=ctk.CTkFont(size=12),
            command=self._refresh_pdf_status,
        ).grid(row=row, column=0, padx=14, pady=(2, 0), sticky="w"); row += 1

        pdf_row = ctk.CTkFrame(p, fg_color="transparent")
        pdf_row.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1
        pdf_row.grid_columnconfigure(0, weight=1)

        self._pdf_combo_values = self._get_pdf_list()
        default_pdf = DEFAULT_REFERENCE_INVOICE.name if DEFAULT_REFERENCE_INVOICE.is_file() else (self._pdf_combo_values[0] if self._pdf_combo_values else "")

        self._pdf_combo = ctk.CTkComboBox(
            pdf_row,
            values=self._pdf_combo_values if self._pdf_combo_values else ["(aucun PDF)"],
            font=ctk.CTkFont(size=11), height=28,
            command=lambda v: self._refresh_pdf_status(),
        )
        if default_pdf:
            self._pdf_combo.set(default_pdf)
        self._pdf_combo.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(pdf_row, text="↺", width=28, height=28, command=self._refresh_pdf_combo).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkButton(pdf_row, text="…", width=28, height=28, command=self._browse_pdf).grid(row=0, column=2)

        self._pdf_status = ctk.CTkLabel(p, text="", font=ctk.CTkFont(size=10), text_color="gray")
        self._pdf_status.grid(row=row, column=0, padx=14, pady=(0, 2), sticky="w"); row += 1
        self._refresh_pdf_status()
        _sep()

        # ── 5: Tests config ───────────────────────────────────────────────
        tests_hdr = ctk.CTkFrame(p, fg_color="transparent")
        tests_hdr.grid(row=row, column=0, padx=12, pady=(4, 0), sticky="ew"); row += 1
        tests_hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            tests_hdr, text="5 — Tests",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray60"),
        ).grid(row=0, column=0, sticky="w")
        self._tests_reload_btn = ctk.CTkButton(
            tests_hdr, text="↺", width=26, height=20,
            font=ctk.CTkFont(size=12), fg_color="gray35", hover_color="gray45",
            command=self._reload_tests,
        )
        self._tests_reload_btn.grid(row=0, column=1, sticky="e")

        self._tests_summary_label = ctk.CTkLabel(
            p, text="", font=ctk.CTkFont(size=10), text_color="gray",
            anchor="w", wraplength=310,
        )
        self._tests_summary_label.grid(row=row, column=0, padx=14, pady=(0, 2), sticky="ew"); row += 1

        ctk.CTkButton(
            p, text="✏  Edit tests.json", height=26,
            font=ctk.CTkFont(size=11), fg_color="gray30", hover_color="gray40",
            command=self._open_tests_file,
        ).grid(row=row, column=0, padx=12, pady=(0, 4), sticky="ew"); row += 1
        self._update_tests_label()
        _sep()

        # ── 6: Model parameters ───────────────────────────────────────────
        _sec("6 — Model parameters")
        adv = ctk.CTkFrame(p, fg_color="transparent")
        adv.grid(row=row, column=0, padx=12, pady=(0, 4), sticky="ew"); row += 1
        adv.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(adv, text="num_ctx", font=ctk.CTkFont(size=11), width=72, anchor="w").grid(row=0, column=0, sticky="w")
        self._num_ctx_combo = ctk.CTkComboBox(
            adv, values=["auto", "2048", "4096", "8192", "16384", "32768", "65536"],
            font=ctk.CTkFont(size=11), height=26,
        )
        self._num_ctx_combo.set("auto")
        self._num_ctx_combo.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ctk.CTkLabel(adv, text="num_predict", font=ctk.CTkFont(size=11), width=72, anchor="w").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._num_predict_combo = ctk.CTkComboBox(
            adv, values=["unlimited", "256", "512", "1024", "2048"],
            font=ctk.CTkFont(size=11), height=26,
        )
        self._num_predict_combo.set("unlimited")
        self._num_predict_combo.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))

        ctk.CTkLabel(
            p, text="num_ctx=auto → Ollama default (2048)  ·  num_predict=unlimited → no cap",
            font=ctk.CTkFont(size=9), text_color=("gray50", "gray55"), wraplength=310,
        ).grid(row=row, column=0, padx=14, pady=(0, 4), sticky="w"); row += 1

        # ── GPU layers slider ──────────────────────────────────────────────
        gpu_hdr = ctk.CTkFrame(p, fg_color="transparent")
        gpu_hdr.grid(row=row, column=0, padx=12, pady=(0, 2), sticky="ew"); row += 1
        gpu_hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            gpu_hdr, text="GPU layers",
            font=ctk.CTkFont(size=11, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._gpu_pct_label = ctk.CTkLabel(
            gpu_hdr, text="100% — Full GPU",
            font=ctk.CTkFont(size=11), anchor="e",
        )
        self._gpu_pct_label.grid(row=0, column=1, sticky="e")

        self._gpu_slider = ctk.CTkSlider(
            p, from_=0, to=100, number_of_steps=20,
            command=self._on_gpu_slider,
        )
        self._gpu_slider.set(100)
        self._gpu_slider.grid(row=row, column=0, padx=12, pady=(0, 4), sticky="ew"); row += 1

        # Split bar GPU (green) / CPU (orange)
        split_bar = ctk.CTkFrame(p, height=10, corner_radius=5, fg_color="transparent")
        split_bar.grid(row=row, column=0, padx=12, pady=(0, 2), sticky="ew"); row += 1
        split_bar.grid_columnconfigure(0, weight=1)
        split_bar.grid_columnconfigure(1, weight=1)
        self._gpu_bar_gpu = ctk.CTkFrame(split_bar, height=10, corner_radius=5, fg_color="#2ecc71")
        self._gpu_bar_gpu.grid(row=0, column=0, sticky="ew", padx=(0, 1))
        self._gpu_bar_cpu = ctk.CTkFrame(split_bar, height=10, corner_radius=5, fg_color="#95a5a6")
        self._gpu_bar_cpu.grid(row=0, column=1, sticky="ew", padx=(1, 0))

        self._gpu_hint = ctk.CTkLabel(
            p,
            text="100% of layers run on GPU  —  no CPU fallback",
            font=ctk.CTkFont(size=9), text_color=("gray50", "gray55"), wraplength=310,
        )
        self._gpu_hint.grid(row=row, column=0, padx=14, pady=(0, 4), sticky="w"); row += 1
        _sep()

        # ── VRAM status panel ─────────────────────────────────────────────
        vram_hdr = ctk.CTkFrame(p, fg_color="transparent")
        vram_hdr.grid(row=row, column=0, padx=12, pady=(4, 0), sticky="ew"); row += 1
        vram_hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            vram_hdr, text="7 — VRAM status",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray60"),
        ).grid(row=0, column=0, sticky="w")
        self._vram_refresh_btn = ctk.CTkButton(
            vram_hdr, text="↺", width=26, height=20,
            font=ctk.CTkFont(size=12), fg_color="gray35", hover_color="gray45",
            command=self._refresh_vram,
        )
        self._vram_refresh_btn.grid(row=0, column=1, sticky="e")

        self._vram_gpu_label = ctk.CTkLabel(
            p, text="Checking GPU…",
            font=ctk.CTkFont(size=11), text_color="gray", anchor="w",
        )
        self._vram_gpu_label.grid(row=row, column=0, padx=14, pady=(2, 1), sticky="ew"); row += 1

        self._vram_bar = ctk.CTkProgressBar(p, height=7)
        self._vram_bar.set(0)
        self._vram_bar.grid(row=row, column=0, padx=12, pady=(0, 2), sticky="ew"); row += 1

        self._vram_free_label = ctk.CTkLabel(
            p, text="",
            font=ctk.CTkFont(size=10), text_color="gray", anchor="w",
        )
        self._vram_free_label.grid(row=row, column=0, padx=14, pady=(0, 1), sticky="ew"); row += 1

        # Up to 6 warning lines — hidden when not needed
        self._vram_warn_labels: List[ctk.CTkLabel] = []
        for _ in range(6):
            lbl = ctk.CTkLabel(
                p, text="",
                font=ctk.CTkFont(size=9), text_color=("#b05a00", "#e67e22"),
                anchor="w", wraplength=318,
            )
            lbl.grid(row=row, column=0, padx=14, pady=0, sticky="ew")
            lbl.grid_remove()
            self._vram_warn_labels.append(lbl)
            row += 1

        _sep()

        # ── 7: Run ────────────────────────────────────────────────────────
        self._run_btn = ctk.CTkButton(
            p, text="▶  Run benchmark", height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_benchmark, state="disabled",
        )
        self._run_btn.grid(row=row, column=0, padx=12, pady=2, sticky="ew"); row += 1

        self._stop_btn = ctk.CTkButton(
            p, text="⏹  Stop", height=30,
            font=ctk.CTkFont(size=12),
            fg_color="#922b21", hover_color="#c0392b",
            command=self._stop_benchmark, state="disabled",
        )
        self._stop_btn.grid(row=row, column=0, padx=12, pady=(0, 2), sticky="ew"); row += 1

        self._run_bar = ctk.CTkProgressBar(p, height=4)
        self._run_bar.grid(row=row, column=0, padx=12, pady=(2, 8), sticky="ew")
        self._run_bar.set(0)

    def _build_main(self, p):
        self._tabs = ctk.CTkTabview(p, corner_radius=10)
        self._tabs.grid(row=0, column=0, sticky="nsew")

        log_tab = self._tabs.add("📋  Log")
        res_tab = self._tabs.add("🏆  Analysis")

        log_tab.grid_rowconfigure(0, weight=1)
        log_tab.grid_columnconfigure(0, weight=1)
        self._log_box = ctk.CTkTextbox(log_tab, font=ctk.CTkFont(family="Courier New", size=12), wrap="word")
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 0))
        self._log_box.configure(state="disabled")

        btn_row = ctk.CTkFrame(log_tab, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(btn_row, text="Clear",    width=88, height=28, fg_color="gray35", hover_color="gray45", command=self._clear_log).pack(side="right")
        ctk.CTkButton(btn_row, text="💾  Export", width=110, height=28, command=self._export_log).pack(side="right", padx=6)

        res_tab.grid_rowconfigure(1, weight=1)
        res_tab.grid_columnconfigure(0, weight=1)
        top_row = ctk.CTkFrame(res_tab, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        ctk.CTkLabel(top_row, text="Top 5 — Best performing models", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self._analyze_btn = ctk.CTkButton(top_row, text="Analyze", width=110, height=32, command=self._show_analysis, state="disabled")
        self._analyze_btn.pack(side="right")

        self._results_scroll = ctk.CTkScrollableFrame(res_tab, corner_radius=8)
        self._results_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        ctk.CTkLabel(self._results_scroll, text="Run a benchmark then click « Analyze »", font=ctk.CTkFont(size=12), text_color="gray").pack(pady=30)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _welcome(self):
        self._log(f"Ollama Benchmark  —  by obviousidea  —  {self._version_label}")
        ids = " + ".join(f"[{t[0].upper()}]" for t in self._tests.text)
        self._log(f"Text tests: {ids}  +  [VISION]  [WEB]  [INVOICE]")
        self._log(f"Tests loaded from: {self._tests.source}")
        self._log("Click « Scan network » to get started.\n")

    # ── Tests config ──────────────────────────────────────────────────────────

    def _update_tests_label(self):
        t = self._tests
        ids = "  ·  ".join(f"[{i.upper()}]" for i, _ in t.text)
        self._tests_summary_label.configure(
            text=f"{ids}  +  [VISION] [WEB] [INVOICE]\n({t.source})"
        )

    def _reload_tests(self):
        self._tests = load_tests()
        self._update_tests_label()
        ids = " + ".join(f"[{t[0].upper()}]" for t in self._tests.text)
        self._log(f"↺  Tests reloaded from {self._tests.source} — {ids}")

    def _open_tests_file(self):
        import subprocess, sys
        tests_path = Path(__file__).parent / "tests.json"
        try:
            if sys.platform == "win32":
                subprocess.Popen(["notepad.exe", str(tests_path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-e", str(tests_path)])
            else:
                subprocess.Popen(["xdg-open", str(tests_path)])
        except Exception as exc:
            self._log(f"⚠  Cannot open tests.json: {exc}")

    # ── VRAM monitor ─────────────────────────────────────────────────────────

    def _refresh_vram(self):
        """Launch background VRAM check and update the panel when done."""
        self._vram_refresh_btn.configure(state="disabled", text="…")
        host = self._selected_host or "127.0.0.1"

        def _run():
            status = get_vram_status(host)
            self.after(0, lambda: self._on_vram_done(status))

        threading.Thread(target=_run, daemon=True).start()

    def _on_vram_done(self, status: VramStatus):
        self._vram_status = status
        self._vram_refresh_btn.configure(state="normal", text="↺")
        self._update_vram_ui(status)

    def _update_vram_ui(self, s: VramStatus):
        """Refresh all VRAM panel widgets from a VramStatus. Must be called from main thread."""
        if not s.available:
            msg = s.error or "VRAM info unavailable"
            # macOS: show unified memory even though platform_name differs
            if s.platform_name == "macos_apple_silicon" and s.total_mb > 0:
                msg = f"Unified  {s.free_gb:.1f} / {s.total_gb:.1f} GB available"
            self._vram_gpu_label.configure(text=msg, text_color="gray")
            self._vram_bar.set(0)
            self._vram_free_label.configure(text="")
            for lbl in self._vram_warn_labels:
                lbl.grid_remove()
            return

        # Bar shows used fraction; color based on how much is FREE
        used_frac = min(1.0, 1.0 - s.free_pct)
        fp = s.free_pct
        if fp >= 0.75:
            bar_color = "#2ecc71"    # green  — plenty free
            txt_color = "#2ecc71"
        elif fp >= 0.55:
            bar_color = "#f1c40f"   # yellow — moderate
            txt_color = "#f1c40f"
        elif fp >= 0.35:
            bar_color = "#e67e22"   # orange — limited
            txt_color = "#e67e22"
        else:
            bar_color = "#e74c3c"   # red    — critical
            txt_color = "#e74c3c"

        short = s.gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        self._vram_gpu_label.configure(
            text=f"🎮 {short}  ·  {s.free_gb:.1f} / {s.total_gb:.1f} GB free",
            text_color=("gray20", "gray85"),
        )
        self._vram_bar.configure(progress_color=bar_color)
        self._vram_bar.set(used_frac)

        # Status line: show cleanup hint if browsers are taking significant VRAM
        if s.cleanup_gain_mb >= 300:
            free_txt = (
                f"💡 After cleanup: ~{s.free_after_gb:.1f} GB  "
                f"(+{s.cleanup_gain_mb/1024:.1f} GB to reclaim)"
            )
        else:
            icon = "✅" if fp >= 0.75 else ("⚡" if fp >= 0.55 else "⚠")
            free_txt = f"{icon} {s.free_gb:.1f} GB free  ({int(fp*100)}%)"
        self._vram_free_label.configure(text=free_txt, text_color=txt_color)

        # Warning rows
        for i, lbl in enumerate(self._vram_warn_labels):
            if i < len(s.warnings):
                lbl.configure(text=s.warnings[i])
                lbl.grid()
            else:
                lbl.configure(text="")
                lbl.grid_remove()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        def _do():
            self._log_box.configure(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}]  {msg}\n"
            self._log_box.insert("end", line)
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
            if self._session_log_file:
                try:
                    self._session_log_file.write(line)
                    self._session_log_file.flush()
                except Exception:
                    pass
        self.after(0, _do)

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _export_log(self):
        if not self._summaries:
            messagebox.showwarning("Export", "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Texte", "*.txt"), ("Tous", "*.*")],
            initialfile=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )
        if path:
            export_report(self._summaries, path, self._ollama_version)
            self._log(f"Report exported: {path}")

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n //= 1024
        return f"{n:.1f} TB"

    # ── Image helpers ─────────────────────────────────────────────────────────

    # ── GPU slider ────────────────────────────────────────────────────────────

    def _on_gpu_slider(self, value: float):
        pct = int(round(value))

        # ── Labels & colors ───────────────────────────────────────────────
        if pct == 100:
            label     = "100% — Full GPU"
            hint      = "100% of layers run on GPU  —  no CPU fallback"
            bar_color = "#2ecc71"   # green
            lbl_color = "#2ecc71"
        elif pct >= 75:
            label     = f"{pct}% — Mostly GPU"
            hint      = f"{pct}% of layers on GPU · {100-pct}% on CPU  —  slight slowdown"
            bar_color = "#f1c40f"   # yellow
            lbl_color = "#f1c40f"
        elif pct >= 40:
            label     = f"{pct}% — Split GPU / CPU"
            hint      = f"{pct}% of layers on GPU · {100-pct}% on CPU  —  expect slower inference"
            bar_color = "#e67e22"   # orange
            lbl_color = "#e67e22"
        elif pct > 0:
            label     = f"{pct}% — Mostly CPU  ⚠"
            hint      = f"Only {pct}% on GPU · {100-pct}% on CPU  —  significantly slower"
            bar_color = "#e74c3c"   # red
            lbl_color = "#e74c3c"
        else:
            label     = "0% — CPU only  ⚠⚠"
            hint      = "All layers on CPU  —  very slow, use only if VRAM is full"
            bar_color = "#c0392b"   # dark red
            lbl_color = "#e74c3c"

        self._gpu_pct_label.configure(text=label, text_color=lbl_color)
        self._gpu_hint.configure(text=hint)

        # ── Split bar weights ─────────────────────────────────────────────
        gpu_w = max(pct, 1)
        cpu_w = max(100 - pct, 1)
        self._gpu_bar_gpu.grid_configure(columnspan=1)
        self._gpu_bar_gpu.configure(fg_color=bar_color)
        # Resize columns via weight
        self._gpu_bar_gpu.master.grid_columnconfigure(0, weight=gpu_w)
        self._gpu_bar_gpu.master.grid_columnconfigure(1, weight=cpu_w)
        cpu_color = "#95a5a6" if pct == 100 else "#e74c3c"
        self._gpu_bar_cpu.configure(fg_color=cpu_color if pct < 100 else "#2ecc71")

    def _get_gpu_num_layers(self) -> int:
        """Convert slider % to num_gpu value. 100% = -1 (auto/full GPU), else 0-999."""
        pct = int(round(self._gpu_slider.get()))
        if pct >= 100:
            return -1   # don't set — Ollama default (all GPU)
        if pct == 0:
            return 0    # CPU only
        return max(1, round(pct / 100 * 999))

    def _get_image_list(self) -> List[str]:
        return scan_images(_PROJECT_DIR)

    def _refresh_img_combo(self):
        images = self._get_image_list()
        self._img_combo_values = images
        current = self._img_combo.get()
        self._img_combo.configure(values=images if images else ["(no image)"])
        if current in images:
            self._img_combo.set(current)
        elif images:
            self._img_combo.set(images[0])
        self._refresh_img_status()

    def _browse_image(self):
        path = filedialog.askopenfilename(
            initialdir=str(_PROJECT_DIR),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
        )
        if path:
            p = Path(path)
            # If image is in project dir, show just the filename; otherwise full path
            try:
                rel = p.relative_to(_PROJECT_DIR)
                self._img_combo.set(str(rel))
            except ValueError:
                self._img_combo.set(str(p))
            self._refresh_img_status()

    def _get_vision_image(self) -> Optional[Path]:
        if not self._vision_var.get():
            return None
        val = self._img_combo.get()
        if not val or val == "(no image)":
            return None
        p = _PROJECT_DIR / val
        if not p.is_file():
            p = Path(val)
        return p if p.is_file() else None

    def _refresh_img_status(self):
        if not self._vision_var.get():
            self._img_status.configure(text="(disabled)", text_color=("gray50", "gray55"))
            return
        p = self._get_vision_image()
        if p:
            kb = p.stat().st_size // 1024
            self._img_status.configure(text=f"✓  {p.name}  ({kb} KB)", text_color="#2ecc71")
        else:
            self._img_status.configure(text="⚠  Image not found", text_color="#e74c3c")

    # ── PDF helpers ───────────────────────────────────────────────────────────

    def _get_pdf_list(self) -> List[str]:
        return scan_pdfs(_PROJECT_DIR)

    def _refresh_pdf_combo(self):
        pdfs = self._get_pdf_list()
        self._pdf_combo_values = pdfs
        current = self._pdf_combo.get()
        self._pdf_combo.configure(values=pdfs if pdfs else ["(no PDF)"])
        if current in pdfs:
            self._pdf_combo.set(current)
        elif pdfs:
            self._pdf_combo.set(pdfs[0])
        self._refresh_pdf_status()

    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            initialdir=str(_PROJECT_DIR),
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
        )
        if path:
            p = Path(path)
            try:
                rel = p.relative_to(_PROJECT_DIR)
                self._pdf_combo.set(str(rel))
            except ValueError:
                self._pdf_combo.set(str(p))
            self._refresh_pdf_status()

    def _get_invoice_pdf(self) -> Optional[Path]:
        if not self._invoice_var.get():
            return None
        val = self._pdf_combo.get()
        if not val or val == "(no PDF)":
            return None
        p = _PROJECT_DIR / val
        if not p.is_file():
            p = Path(val)
        return p if p.is_file() else None

    def _refresh_pdf_status(self):
        if not self._invoice_var.get():
            self._pdf_status.configure(text="(disabled)", text_color=("gray50", "gray55"))
            return
        p = self._get_invoice_pdf()
        if p:
            kb = p.stat().st_size // 1024
            self._pdf_status.configure(text=f"✓  {p.name}  ({kb} KB)", text_color="#2ecc71")
        else:
            self._pdf_status.configure(text="⚠  PDF not found", text_color="#e74c3c")

    # ── Scan ──────────────────────────────────────────────────────────────────

    def _start_scan(self):
        if self._scanning:
            return
        try:
            extra = parse_scan_targets(self._scan_targets.get())
        except (ValueError, OSError) as e:
            self._scan_lbl.configure(text=f"Invalid targets: {e}")
            return
        self._scanning = True
        self._scan_btn.configure(state="disabled", text="Scanning…")
        self._scan_bar.set(0)
        self._scan_lbl.configure(text="")
        self._log("Scanning local network…")
        for w in self._host_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._host_frame, text="Scan…", font=ctk.CTkFont(size=11), text_color="gray").pack(pady=6)

        def _run():
            hosts = scan_network(lambda pct, msg: (
                self.after(0, lambda: self._scan_bar.set(pct)),
                self.after(0, lambda: self._scan_lbl.configure(text=msg)),
            ), extra)
            self.after(0, lambda: self._on_scan_done(hosts))

        threading.Thread(target=_run, daemon=True).start()

    def _on_scan_done(self, hosts: List[str]):
        self._scanning = False
        self._hosts = hosts
        self._scan_btn.configure(state="normal", text="Scan network")
        self._scan_bar.set(1.0)
        for w in self._host_frame.winfo_children():
            w.destroy()

        if not hosts:
            ctk.CTkLabel(self._host_frame, text="No instance found", font=ctk.CTkFont(size=11), text_color="gray").pack(pady=6)
            self._log("No Ollama instance found on the network.")
            return

        self._log(f"{len(hosts)} instance(s): {', '.join(hosts)}")
        for host in hosts:
            try:
                hostname = socket.gethostbyaddr(host)[0].split(".")[0]
                label = f"  {host}  —  {hostname}"
            except Exception:
                label = f"  {host}"
            ctk.CTkButton(
                self._host_frame, text=label, anchor="w", height=28,
                font=ctk.CTkFont(family="Courier New", size=12),
                fg_color="gray28", hover_color="gray38",
                command=lambda h=host: self._select_host(h),
            ).pack(fill="x", pady=1, padx=2)

    # ── Host / model ──────────────────────────────────────────────────────────

    def _select_host(self, host: str):
        self._selected_host = host
        self._log(f"Instance: {host}")
        self._load_models(host)

    def _load_models(self, host: str):
        self._model_checkboxes = []
        self._run_btn.configure(state="disabled")
        for w in self._models_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._models_frame, text="Loading…", font=ctk.CTkFont(size=11), text_color="gray").pack(pady=6)

        def _run():
            try:
                models = get_models(host)
                version = get_ollama_version(host)
                self.after(0, lambda: self._on_models_loaded(models, version))
            except Exception as exc:
                self.after(0, lambda: self._on_models_error(str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_models_loaded(self, models: List[dict], version: str = "?"):
        for w in self._models_frame.winfo_children():
            w.destroy()
        self._model_checkboxes = []
        if not models:
            ctk.CTkLabel(self._models_frame, text="No models found", font=ctk.CTkFont(size=11), text_color="gray").pack(pady=6)
            return
        self._ollama_version = version
        self._log(f"{len(models)} model(s) available  —  Ollama v{version}")
        available_names = {m.get("name", "") for m in models}
        saved = self._load_selections().get(self._selected_host, None)
        saved_set = set(saved) & available_names if saved is not None else set()
        for m in models:
            cb = ModelCheckbox(
                self._models_frame,
                m.get("name", ""),
                self._fmt_size(m.get("size", 0)),
                on_delete=self._delete_model_prompt,
            )
            cb.pack(anchor="w", padx=6, pady=2, fill="x")
            if saved is not None:
                cb.var.set(m.get("name", "") in saved_set)
            self._model_checkboxes.append(cb)
        self._run_btn.configure(state="normal")

    def _on_models_error(self, err: str):
        for w in self._models_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._models_frame, text=f"Error: {err}", font=ctk.CTkFont(size=11), text_color="#e74c3c").pack(pady=6)
        self._log(f"Models error: {err}")

    # ── Model deletion ────────────────────────────────────────────────────────

    def _delete_model_prompt(self, model_name: str):
        confirmed = messagebox.askyesno(
            "Supprimer le modèle",
            f"Supprimer '{model_name}' du stockage Ollama local ?\n\nCette action est irréversible.",
            icon="warning",
        )
        if confirmed:
            self._do_delete_model(model_name)

    def _do_delete_model(self, model_name: str):
        self._log(f"Suppression de {model_name}…")

        def _run():
            try:
                delete_model(self._selected_host, model_name)
                self.after(0, lambda: (
                    self._log(f"Modèle '{model_name}' supprimé."),
                    self._load_models(self._selected_host),
                ))
            except Exception as exc:
                self.after(0, lambda: (
                    self._log(f"Erreur suppression : {exc}"),
                    messagebox.showerror("Échec de la suppression", str(exc)),
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _select_all(self):
        for cb in self._model_checkboxes:
            cb.var.set(True)

    def _select_none(self):
        for cb in self._model_checkboxes:
            cb.var.set(False)

    # ── Selections persistence ─────────────────────────────────────────────────

    def _load_selections(self) -> dict:
        try:
            return json.loads(_SELECTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_selections(self):
        if not self._selected_host:
            return
        selected = [cb.model_name for cb in self._model_checkboxes if cb.var.get()]
        data = self._load_selections()
        data[self._selected_host] = selected
        try:
            _SELECTIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def _start_benchmark(self):
        if self._running:
            return
        selected = [cb.model_name for cb in self._model_checkboxes if cb.var.get()]
        if not selected:
            messagebox.showwarning("Benchmark", "No model selected.")
            return

        include_vision  = self._vision_var.get()
        include_web     = self._web_var.get()
        include_invoice = self._invoice_var.get()
        vision_image    = self._get_vision_image()
        invoice_pdf     = self._get_invoice_pdf()

        if include_vision and vision_image is None:
            if not messagebox.askyesno("Missing image", "Reference image not found.\nContinue without vision test?"):
                return
            include_vision = False

        if include_invoice and invoice_pdf is None:
            if not messagebox.askyesno("Missing PDF", "Invoice PDF not found.\nContinue without invoice test?"):
                return
            include_invoice = False

        self._save_selections()
        self._running = True
        self._stop_event.clear()
        self._summaries = []

        # ── Ouvrir le fichier de log de session ───────────────────────────
        _logs_dir = _PROJECT_DIR / "logs"
        _logs_dir.mkdir(exist_ok=True)
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _host_safe = (self._selected_host or "unknown").replace(".", "-")
        _log_path = _logs_dir / f"{_host_safe}_{_ts}.log"
        try:
            self._session_log_file = open(_log_path, "w", encoding="utf-8")
            self._session_log_path = _log_path
        except Exception:
            self._session_log_file = None
            self._session_log_path = None
        self._run_btn.configure(state="disabled", text="⏳  Running…")
        self._stop_btn.configure(state="normal")
        self._analyze_btn.configure(state="disabled")
        self._run_bar.set(0)

        host = self._selected_host
        extras = []
        if include_vision and vision_image:
            extras.append(f"VISION ({vision_image.name})")
        if include_web:
            extras.append("WEB")
        if include_invoice and invoice_pdf:
            extras.append(f"INVOICE ({invoice_pdf.name})")

        # ── Read model parameters ─────────────────────────────────────────
        _ctx_val = self._num_ctx_combo.get()
        num_ctx = int(_ctx_val) if _ctx_val.isdigit() else 0
        _pred_val = self._num_predict_combo.get()
        num_predict = int(_pred_val) if _pred_val.isdigit() else -1
        num_gpu = self._get_gpu_num_layers()
        _gpu_pct = int(round(self._gpu_slider.get()))

        self._tabs.set("📋  Log")
        self._log(f"Benchmark on {host}  —  Ollama v{self._ollama_version}")
        self._log(f"Models: {', '.join(selected)}")
        _ctx_str  = f"num_ctx={num_ctx}" if num_ctx > 0 else "num_ctx=auto"
        _pred_str = f"  ·  num_predict={num_predict}" if num_predict > 0 else ""
        _gpu_str  = f"  ·  GPU layers=auto (100%)" if num_gpu < 0 else f"  ·  GPU layers={num_gpu} ({_gpu_pct}%)"
        self._log(f"Config: {_ctx_str}{_pred_str}{_gpu_str}")
        self._log(f"Tests: COLD + HOT{(' + ' + ' + '.join(extras)) if extras else ''}")
        # ── VRAM pre-flight report ────────────────────────────────────────
        vs = self._vram_status
        if vs and vs.available:
            short = vs.gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            self._log(f"GPU: {short}  —  {vs.free_gb:.1f} GB free of {vs.total_gb:.1f} GB  ({int(vs.free_pct*100)}%)")
            for w in vs.warnings:
                self._log(w)
            if vs.app_breakdown:
                total_est = sum(vs.app_breakdown.values())
                apps_str = "  ·  ".join(f"{n} {v:,} MB" for n, v in sorted(vs.app_breakdown.items(), key=lambda x: -x[1]))
                self._log(f"   Apps using VRAM: {apps_str}  →  ~{total_est:,} MB recouperable")
        self._log("")

        def _run():
            results = run_benchmark(
                host=host,
                models=selected,
                include_vision=include_vision,
                vision_image=vision_image,
                include_web=include_web,
                include_invoice=include_invoice,
                invoice_path=invoice_pdf,
                log_cb=self._log,
                progress_cb=lambda pct: self.after(0, lambda: self._run_bar.set(pct)),
                stop_event=self._stop_event,
                num_ctx=num_ctx,
                num_predict=num_predict,
                num_gpu=num_gpu,
                tests=self._tests,
            )
            self.after(0, lambda: self._on_benchmark_done(results))

        threading.Thread(target=_run, daemon=True).start()

    def _on_benchmark_done(self, summaries: List[ModelSummary]):
        self._running = False
        self._summaries = summaries
        self._run_btn.configure(state="normal", text="▶  Run benchmark")
        self._stop_btn.configure(state="disabled")
        self._analyze_btn.configure(state="normal")
        self._run_bar.set(1.0)
        stopped = self._stop_event.is_set()
        self._log(f"\n{'⏹  Stopped' if stopped else 'Done'} — {len(summaries)} model(s) tested.")

        # ── Close log + auto-export markdown report ───────────────────────
        if self._session_log_file:
            try:
                self._session_log_file.close()
            except Exception:
                pass
            self._session_log_file = None
        if summaries and self._session_log_path:
            try:
                _md_path = self._session_log_path.with_suffix(".md")
                export_report(summaries, str(_md_path), self._ollama_version)
                self._log(f"📁  Log + report saved → logs/{self._session_log_path.stem}")
            except Exception:
                pass

        self._show_analysis()
        self._tabs.set("🏆  Analysis")

    def _stop_benchmark(self):
        if self._running:
            self._stop_event.set()
            self._stop_btn.configure(state="disabled", text="⏳  Stopping…")
            self._log("⏹  Stop requested — current model will finish before stopping.")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _show_analysis(self):
        if not self._summaries:
            messagebox.showinfo("Analysis", "Run a benchmark first.")
            return
        for w in self._results_scroll.winfo_children():
            w.destroy()

        ranked = sorted(
            [s for s in self._summaries if s.results],
            key=lambda s: s.avg_tokens_per_second, reverse=True,
        )
        if not ranked:
            ctk.CTkLabel(self._results_scroll, text="No valid results.", text_color="gray").pack(pady=20)
            return

        top5         = ranked[:5]
        max_tps      = top5[0].avg_tokens_per_second
        min_chars    = min((s.total_response_chars for s in top5 if s.total_response_chars > 0), default=1)
        min_duration = min((s.sum_total_ms         for s in top5 if s.sum_total_ms > 0),         default=1)

        has_vision  = any(s.vision_result  for s in ranked)
        has_web     = any(s.web_result     for s in ranked)
        has_invoice = any(s.invoice_result for s in ranked)
        legend_parts = []
        if has_vision:  legend_parts.append("👁 vision")
        if has_web:     legend_parts.append("🌐 web")
        if has_invoice: legend_parts.append("📄 invoice")
        legend_parts += ["⚡ tok/s", "📝 conciseness", "⏱ duration"]
        ctk.CTkLabel(
            self._results_scroll,
            text="  ·  ".join(legend_parts),
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=5, pady=(4, 8))

        for rank, summary in enumerate(top5, 1):
            ResultCard(
                self._results_scroll, rank, summary,
                max_tps, min_chars, min_duration,
            ).pack(fill="x", padx=5, pady=5)

        rest = ranked[5:]
        if rest:
            ctk.CTkLabel(self._results_scroll, text=f"\nOther models ({len(rest)})", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=5, pady=(10, 4))
            for s in rest:
                row_f = ctk.CTkFrame(self._results_scroll, height=36, corner_radius=6)
                row_f.pack(fill="x", padx=5, pady=2)
                badges = ("  👁" if s.vision_supported else "") + ("  🌐" if s.web_result else "")
                ctk.CTkLabel(row_f, text=s.model + badges, font=ctk.CTkFont(size=12), anchor="w").pack(side="left", padx=10, pady=4)
                ctk.CTkLabel(row_f, text=f"{s.avg_tokens_per_second:.1f} tok/s  ·  {s.avg_total_ms:.0f} ms  ·  {s.total_tokens} tok", font=ctk.CTkFont(size=11), text_color="gray").pack(side="right", padx=10, pady=4)

        failed = [s for s in self._summaries if not s.results]
        if failed:
            ctk.CTkLabel(self._results_scroll, text=f"\nErreurs ({len(failed)})", font=ctk.CTkFont(size=13, weight="bold"), text_color="#e74c3c").pack(anchor="w", padx=5, pady=(10, 4))
            for s in failed:
                ctk.CTkLabel(self._results_scroll, text=f"  · {s.model}  —  {s.error or '?'}", font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", padx=5, pady=1)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = OllamaBenchmarkApp()
    app.mainloop()
