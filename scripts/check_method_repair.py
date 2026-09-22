#!/usr/bin/env python3
"""Run method regression checks and verify frozen evaluation files."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "cervithink/metrics.py", "cervithink/parsing.py", "scripts/evaluate_predictions.py",
    "scripts/evaluate_dataset.py", "scripts/evaluate_hf_baseline.py",
    "scripts/aggregate_experiment_tables.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT/"outputs/method_checks"))
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(ROOT)
    checks = {}
    # Works in both a Git checkout and a source ZIP. The baseline fingerprints
    # were taken from commit 134a937 before the method-only repair.
    baseline = json.loads((ROOT/"configs/evaluator_baseline_sha256.json").read_text(encoding="utf-8"))
    for path in PROTECTED:
        actual = (ROOT/path).read_bytes()
        checks[path] = hashlib.sha256(actual.replace(b"\r\n", b"\n")).hexdigest() == baseline["sha256"][path]
    for directory in ("cervithink", "scripts", "tests"):
        for path in (ROOT/directory).glob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    shell_checks = {}
    if bash:
        for script in sorted((ROOT/"scripts").glob("*.sh")):
            rc = subprocess.run([bash, "-n", str(script.relative_to(ROOT)).replace("\\", "/")], cwd=ROOT).returncode
            shell_checks[script.name] = rc == 0
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/"tests")))
    status = {
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [str(reason) for _, reason in result.skipped],
        "protected_evaluator_files_unchanged": checks,
        "shell_syntax": shell_checks,
        "gpu_sft_grpo_verified": False,
        "paper_metrics_verified": False,
        "method_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("cervithink", "scripts", "tests")
            for path in (ROOT/folder).glob("*.py")
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"verification.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Saved method verification report: {output/'verification.json'}")
    return 0 if result.wasSuccessful() and all(checks.values()) and all(shell_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
