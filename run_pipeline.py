"""
VeReMi Full Pipeline Runner
============================
Runs all 3 steps in sequence:
  Step 1 -> Step 2 -> Step 3

Usage:
  python run_pipeline.py

All outputs go to:
  dataset_inventory.csv
  processed_dataset.csv
  results/   (reports + charts)
"""

import sys
import subprocess
import time
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STEPS = [
    ("STEP 1 — Dataset Scanner",   "step1_scan_dataset.py"),
    ("STEP 2 — Preprocessing",     "step2_preprocess.py"),
    ("STEP 3 — Model Training",    "step3_train_evaluate.py"),
]

def run_step(name, script):
    print("\n" + "=" * 60)
    print(f"  RUNNING: {name}")
    print("=" * 60)

    if not Path(script).exists():
        print(f"[ERROR] Script not found: {script}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        encoding="utf-8",
        errors="replace"
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n[ERROR] {name} failed with exit code {result.returncode}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  VeReMi Full Pipeline — Starting")
    print("=" * 60)
    t_total = time.time()

    for name, script in STEPS:
        success = run_step(name, script)
        if not success:
            print(f"\n[ABORT] Pipeline stopped at: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE in {time.time()-t_total:.1f}s")
    print(f"  Results saved to: results/")
    print("=" * 60)
