"""Summarise milestone benchmark reports as a Markdown table.

Usage: python benchmarks/history/make_table.py <results-dir>

Reads <label>_run<k>.json written by run_linux.sh / run_windows.ps1 and, per
milestone, reports the run with the lowest complete first-run total (the
range over all runs is shown next to it). All times in seconds:
  Setup          imports + CPU init + input load + setup build
  Compile        every trace/lower and compile phase
  Forward, Sensitivity, Gradient   execution of each stage (no compile)
  Full first run benchmark.py's complete first-run total
  Without setup  full first run minus setup
  Warm median    median of the 3 warm repeats
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETUP = ("imports", "initialize", "load", "build")


def milestones():
    for line in open(os.path.join(HERE, "milestones.txt"), encoding="utf-8"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        label, sha, _kind, _env, desc = [f.strip() for f in line.split("|")][:5]
        yield label, sha, desc


def summarise(path):
    r = json.load(open(path))
    ph = {p["key"]: p["seconds"] for p in r["phases"]}
    t = r["timing_seconds"]
    setup = sum(ph.get(k, 0.0) for k in SETUP)
    return dict(
        total=t["end_to_end"], setup=setup,
        compile=sum(v for k, v in ph.items() if k.endswith("_lower") or k.endswith("_compile")),
        forward=ph.get("forward", 0.0), sens=ph.get("sensitivity", 0.0), grad=ph.get("gradient", 0.0),
        warm=t["runtime_median"], err=r["validation"]["max_absolute_log_sensitivity_error"],
        ok=r["validation"]["upstream_sensitivity_check_passed"])


def fmt(v):
    return f"{v:.3f}" if v < 100 else f"{v:.1f}"


def main(results):
    print("| Milestone | Commit | Setup | Compile | Forward | Sensitivity | Gradient | Full first run (range) "
          "| Without setup | Warm median | Check (max log error) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for label, sha, desc in milestones():
        runs = [summarise(p) for p in sorted(glob.glob(os.path.join(results, f"{label}_run*.json")))]
        if not runs:
            print(f"| {desc} | `{sha}` | not run | | | | | | | | |")
            continue
        b = min(runs, key=lambda x: x["total"])
        tot = [x["total"] for x in runs]
        rng = f"{min(tot):.2f}–{max(tot):.2f}"
        check = "PASS" if all(x["ok"] for x in runs) else "FAIL"
        print(f"| {desc} | `{sha}` | {fmt(b['setup'])} | {fmt(b['compile'])} | {fmt(b['forward'])} | "
              f"{b['sens']:.4f} | {fmt(b['grad'])} | **{fmt(b['total'])}** ({rng}) | {fmt(b['total'] - b['setup'])} | "
              f"{fmt(b['warm'])} | {check} ({b['err']:.2e}) |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dfm-history/results"))
