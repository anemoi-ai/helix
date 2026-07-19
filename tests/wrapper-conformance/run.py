#!/usr/bin/env python3
"""
Helix wrapper conformance runner.

Usage:
    python run.py [--cases cases.yaml] [--shim-url http://localhost:8080]
                  [--wrappers python,rust,go,node,dotnet,swift]
                  [--output results/]

Each wrapper provides a conformance driver at:
    wrappers/<lang>/bin/conformance  (or .exe on Windows)

The driver receives:
    argv[1]  = path to cases.yaml
    env HELIX_SHIM_URL = shim server URL
    env HELIX_TEST_MODEL = model alias to use

It writes a JSON array to stdout:
    [{"name": "...", "passed": true/false, "error": null, "detail": {...}}, ...]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent.parent
WRAPPERS_DIR = REPO_ROOT / "wrappers"
CASES_FILE = Path(__file__).parent / "cases.yaml"

ALL_WRAPPERS = ["python", "rust", "go", "node", "dotnet", "swift"]


def find_driver(lang: str) -> Path | None:
    candidates = [
        WRAPPERS_DIR / lang / "bin" / "conformance",
        WRAPPERS_DIR / lang / "bin" / "conformance.exe",
        WRAPPERS_DIR / lang / "bin" / "conformance.py",
    ]
    # Language-specific build outputs
    if lang == "rust":
        candidates.insert(0, WRAPPERS_DIR / lang / "helix" / "target" / "debug" / "conformance")
        candidates.insert(0, WRAPPERS_DIR / lang / "helix" / "target" / "release" / "conformance")
    if lang == "go":
        candidates.insert(0, WRAPPERS_DIR / lang / "conformance")
    if lang == "dotnet":
        candidates.insert(0, WRAPPERS_DIR / lang / "tests" / "Helix.Conformance" / "bin" / "Debug" / "net8.0" / "Helix.Conformance")
    if lang == "node":
        candidates.insert(0, WRAPPERS_DIR / lang / "bin" / "conformance.js")
    return next((p for p in candidates if p.exists()), None)


def run_wrapper(lang: str, cases_file: Path, shim_url: str, model: str, timeout: int) -> list[dict]:
    driver = find_driver(lang)
    if driver is None:
        return [{"name": f"__driver_missing_{lang}", "passed": False,
                 "error": f"conformance driver not found for {lang}", "detail": {}}]

    cmd: list[str]
    if driver.suffix == ".py":
        cmd = [sys.executable, str(driver), str(cases_file)]
    elif driver.suffix == ".js":
        cmd = ["node", str(driver), str(cases_file)]
    else:
        cmd = [str(driver), str(cases_file)]

    env = {**os.environ, "HELIX_SHIM_URL": shim_url, "HELIX_TEST_MODEL": model}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if proc.returncode != 0:
            return [{"name": f"__driver_crash_{lang}", "passed": False,
                     "error": f"exit {proc.returncode}: {proc.stderr[:500]}", "detail": {}}]
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return [{"name": f"__driver_timeout_{lang}", "passed": False,
                 "error": f"timed out after {timeout}s", "detail": {}}]
    except json.JSONDecodeError as e:
        return [{"name": f"__driver_bad_json_{lang}", "passed": False,
                 "error": f"bad JSON output: {e}", "detail": {}}]


def print_results(lang: str, results: list[dict[str, Any]]) -> tuple[int, int]:
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    width = 50
    print(f"\n{'═' * 60}")
    print(f"  {lang.upper():10s}  {passed}/{total} passed")
    print(f"{'═' * 60}")
    for r in results:
        icon = "✓" if r.get("passed") else "✗"
        name = r["name"][:width].ljust(width)
        err = f"  {r['error'][:80]}" if not r.get("passed") and r.get("error") else ""
        print(f"  {icon} {name}{err}")
    return passed, total


def main() -> int:
    parser = argparse.ArgumentParser(description="Helix wrapper conformance runner")
    parser.add_argument("--cases", default=str(CASES_FILE))
    parser.add_argument("--shim-url", default=os.environ.get("HELIX_SHIM_URL", "http://localhost:8080"))
    parser.add_argument("--model", default=os.environ.get("HELIX_TEST_MODEL", "qwen-test"))
    parser.add_argument("--wrappers", default=",".join(ALL_WRAPPERS))
    parser.add_argument("--output", default=None, help="Directory to write per-language JSON results")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    wrappers = [w.strip() for w in args.wrappers.split(",") if w.strip()]
    cases_file = Path(args.cases)
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    grand_passed = grand_total = 0
    failed_langs: list[str] = []

    for lang in wrappers:
        results = run_wrapper(lang, cases_file, args.shim_url, args.model, args.timeout)
        passed, total = print_results(lang, results)
        grand_passed += passed
        grand_total += total
        if passed < total:
            failed_langs.append(lang)
        if output_dir:
            (output_dir / f"{lang}.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'═' * 60}")
    print(f"  TOTAL  {grand_passed}/{grand_total} passed  ({len(failed_langs)} wrapper(s) with failures)")
    if failed_langs:
        print(f"  Failed: {', '.join(failed_langs)}")
    print(f"{'═' * 60}\n")

    return 0 if not failed_langs else 1


if __name__ == "__main__":
    sys.exit(main())
