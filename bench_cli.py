"""
Headless benchmark + comparison with past reports (no GUI).

Usage:
    uv run bench_cli.py                       # re-run models seen in past reports on localhost
    uv run bench_cli.py -m ministral-3:8b     # explicit models
    uv run bench_cli.py --host 192.168.1.35
    uv run bench_cli.py --compare-only        # only print history, run nothing

Run it after each Ollama update to track tok/s evolution. Reports are written
as benchmark_*.md with the Ollama version, so future comparisons can group by version.
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import core

ROOT = Path(__file__).parent
_ENTRY = re.compile(r"^## #\d+ — (.+?)  \(@ (\S+)\)\n- \*\*([\d.]+) tok/s\*\*", re.M)


def load_history(exclude: str = ""):
    """{(model, host): [(date, version, tok_s), ...]} from benchmark_*.md, oldest first."""
    hist = defaultdict(list)
    for f in sorted(ROOT.glob("benchmark_*.md")):
        if f.name == exclude:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        date = re.search(r"benchmark_(\d{4})(\d{2})(\d{2})", f.name)
        ver = re.search(r"^Ollama v(\S+)", text, re.M)
        label = "-".join(date.groups()) if date else f.stem
        for m in _ENTRY.finditer(text):
            tps = float(m.group(3))
            if tps > 0:  # 0.0 = failed run
                hist[(m.group(1), m.group(2))].append((label, ver.group(1) if ver else "?", tps))
    return hist


def print_comparison(hist, keys, current=None):
    print("\n=== tok/s evolution (oldest -> newest) ===")
    for key in keys:
        rows = list(hist.get(key, []))
        if current and key in current:
            rows.append(("now", current[key][0], current[key][1]))
        if not rows:
            continue
        model, host = key
        print(f"\n{model} @ {host}")
        for date, ver, tps in rows:
            print(f"  {date:<11} ollama {ver:<8} {tps:7.2f} tok/s")
        if len(rows) > 1:
            prev = [r[2] for r in rows[:-1]]
            avg = sum(prev) / len(prev)
            print(f"  -> latest vs average of previous runs: {rows[-1][2] / avg - 1:+.1%}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("-m", "--models", nargs="+", help="models to run (default: those in past reports for this host)")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    hist = load_history()
    if args.compare_only:
        print_comparison(hist, sorted(k for k in hist if k[1] == args.host))
        return 0

    version = core.get_ollama_version(args.host)
    if version == "?":
        print(f"Ollama not reachable on {args.host}", file=sys.stderr)
        return 1
    installed = {m.get("name", "") for m in core.get_models(args.host)}
    wanted = args.models or sorted({k[0] for k in hist if k[1] == args.host})
    models = [m for m in wanted if m in installed]
    for m in set(wanted) - set(models):
        print(f"skip {m}: not installed on {args.host}")
    if not models:
        print("No model to run.", file=sys.stderr)
        return 1

    print(f"Ollama v{version} on {args.host}: {', '.join(models)}")
    results = core.run_benchmark(args.host, models, log_cb=lambda m="": print(m, flush=True))
    out = ROOT / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.md"
    core.export_report(results, str(out), ollama_version=version)
    print(f"\nReport: {out.name}")

    hist = load_history(exclude=out.name)
    current = {(r.model, args.host): (version, r.avg_tokens_per_second) for r in results if not r.error}
    print_comparison(hist, [(m, args.host) for m in models], current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
