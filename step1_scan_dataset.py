"""
STEP 1 FAST: VeReMi Dataset Scanner (Optimized for OneDrive/Large Datasets)
=============================================================================
Instead of reading every file byte-by-byte, this version:
  - Walks directories quickly
  - Only reads 1 line of each file for schema sampling
  - Uses os.path.getsize() for approximate size instead of line counting
  - Skips record counting (to be done in step2 during processing)

Outputs:
  - dataset_inventory.csv
  - results/step1_scan_report.txt
"""

import os
import json
import csv
import time
import sys
from pathlib import Path
from collections import defaultdict

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_PATH = Path("VeReMi_Dataset")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ATTACK_CODE_NAMES = {
    "0":  "Normal (No Attack)",
    "1":  "Constant Attack",
    "2":  "Random Offset Attack",
    "4":  "Constant Offset Attack",
    "8":  "Random Position Attack",
    "16": "Eventual Stop Attack",
}

def extract_attack_code(filename):
    name = Path(filename).stem
    if "-A" in name:
        parts = name.split("-A")
        if len(parts) >= 2:
            return parts[-1]
    return "UNKNOWN"

def is_ground_truth(filename):
    name = filename.lower()
    return "groundtruth" in name or "ground_truth" in name

def sample_first_line(filepath):
    """Read only the very first non-empty JSON line."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        pass
    return {}

def scan_dataset():
    print("=" * 60)
    print("  STEP 1 (FAST): VeReMi Dataset Scanner")
    print("=" * 60)
    print(f"  Base Path : {DATASET_PATH.resolve()}")
    print(f"  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode      : Fast scan (no full record counting)", flush=True)
    print()

    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset path does not exist: {DATASET_PATH.resolve()}")
        sys.exit(1)

    start_time = time.time()

    inventory       = []
    attack_cats     = defaultdict(int)
    total_sim_dirs  = 0
    total_json      = 0
    total_gt        = 0
    total_folders   = 0
    corrupted       = 0

    regular_schema  = {}
    gt_schema       = {}
    sampled_regular = False
    sampled_gt      = False

    folder_count = 0
    for root, dirs, files in os.walk(DATASET_PATH):
        dirs.sort()
        total_folders += len(dirs)
        root_path = Path(root)

        depth = len(root_path.relative_to(DATASET_PATH).parts)
        if depth == 0:
            total_sim_dirs = len(dirs)

        json_files = [f for f in files if f.lower().endswith(".json")]
        total_json += len(json_files)

        if json_files:
            folder_count += 1
            if folder_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  … processed {folder_count} folders with JSON files  ({elapsed:.1f}s)", flush=True)

        for fname in json_files:
            fpath     = root_path / fname
            is_gt     = is_ground_truth(fname)
            file_size = 0
            try:
                file_size = fpath.stat().st_size
            except Exception:
                pass

            if is_gt:
                total_gt += 1
            else:
                code = extract_attack_code(fname)
                attack_cats[code] += 1

            if file_size == 0:
                corrupted += 1

            # Sample schema once each
            if is_gt and not sampled_gt and file_size > 0:
                gt_schema = sample_first_line(fpath)
                sampled_gt = True

            if not is_gt and not sampled_regular and file_size > 0:
                regular_schema = sample_first_line(fpath)
                sampled_regular = True

            try:
                parts    = fpath.relative_to(DATASET_PATH).parts
                sim_name = parts[0] if parts else "unknown"
            except Exception:
                sim_name = "unknown"

            inventory.append({
                "file_path":      str(fpath),
                "file_name":      fname,
                "simulation":     sim_name,
                "is_ground_truth": is_gt,
                "attack_code":    "" if is_gt else extract_attack_code(fname),
                "attack_name":    "" if is_gt else ATTACK_CODE_NAMES.get(
                                      extract_attack_code(fname), "Unknown"),
                "file_size_bytes": file_size,
                "corrupted":      file_size == 0,
            })

    elapsed = time.time() - start_time
    print(f"\n[INFO] Scan complete in {elapsed:.2f}s")
    print(f"[INFO] Total JSON files found: {total_json}", flush=True)

    # ── Save Inventory ──────────────────────────────────────────────────
    inv_path = Path("dataset_inventory.csv")
    fieldnames = ["file_path", "file_name", "simulation", "is_ground_truth",
                  "attack_code", "attack_name", "file_size_bytes", "corrupted"]
    with open(inv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    print(f"[OK] Inventory saved: {inv_path}  ({len(inventory)} entries)")

    # ── Build Report ────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("  VeReMi DATASET SCAN REPORT  (Step 1 - Fast Mode)")
    lines.append("=" * 60)
    lines.append(f"  Scan Time      : {elapsed:.2f} seconds")
    lines.append(f"  Scan Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("  DATASET STATISTICS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Simulation Folders  : {total_sim_dirs}")
    lines.append(f"  Total Folders       : {total_folders}")
    lines.append(f"  Total JSON Files    : {total_json}")
    lines.append(f"  Ground Truth Files  : {total_gt}")
    lines.append(f"  Regular Log Files   : {total_json - total_gt}")
    lines.append(f"  Corrupted/Empty     : {corrupted}")
    lines.append("")
    lines.append("  ATTACK CATEGORIES")
    lines.append("  " + "-" * 40)
    for code in sorted(attack_cats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        name  = ATTACK_CODE_NAMES.get(code, "Unknown")
        count = attack_cats[code]
        lines.append(f"  A{code:>2s} | {name:<30s} | {count:>5d} log files")
    lines.append("")
    lines.append("  REGULAR LOG SCHEMA (first record sample)")
    lines.append("  " + "-" * 40)
    for k, v in regular_schema.items():
        lines.append(f"    {k}: {type(v).__name__}  (example: {str(v)[:60]})")
    lines.append("")
    lines.append("  GROUND TRUTH SCHEMA (first record sample)")
    lines.append("  " + "-" * 40)
    for k, v in gt_schema.items():
        lines.append(f"    {k}: {type(v).__name__}  (example: {str(v)[:60]})")
    lines.append("")
    lines.append("  FILES SAVED")
    lines.append("  " + "-" * 40)
    lines.append(f"  Inventory : {inv_path.resolve()}")
    lines.append(f"  Report    : {(RESULTS_DIR / 'step1_scan_report.txt').resolve()}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    report_path = RESULTS_DIR / "step1_scan_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[OK] Report saved: {report_path}")
    print("\n>>> Step 1 Complete. Run step2_preprocess.py next.\n")

if __name__ == "__main__":
    scan_dataset()
