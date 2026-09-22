#!/usr/bin/env python3
"""Non-destructive alpha release checks; no real-model training is launched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from release_inventory import ROOT, check_inventory, release_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--report-dir", default=str(ROOT/"outputs/release_checks"))
    args = parser.parse_args()
    issues = check_inventory()
    if issues:
        for issue in issues:
            print(issue)
        return 1
    manifest = json.loads((ROOT/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["validation"]["paper_metrics_reproduced"] or manifest["validation"]["real_7b_sft_grpo_tested"]:
        raise ValueError("Do not promote validation flags without the corresponding evidence")
    if not args.skip_tests:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([
            sys.executable, "-B", str(ROOT/"scripts/check_method_repair.py"),
            "--output-dir", str(Path(args.report_dir).resolve()),
        ], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print(f"release_check: selected {len(release_files())} source files passed")
    print("Scope: alpha method-code candidate; GPU training and paper metrics remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
