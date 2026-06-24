"""
STEP 2 FAST (Sampled) — VeReMi Preprocessing
===============================================
Instead of all 48,015 files, picks a STRATIFIED SAMPLE:
  - Up to MAX_PER_CLASS files per attack class
  - Balanced across all 6 classes
  - Produces a real processed_dataset.csv quickly

This gives REAL, ACCURATE model results in ~5 min instead of 45+ min.
For a dataset this large, sampling is standard ML practice.

Target: ~5,000 files total -> ~6M records -> Step 3 in ~10 min
"""

import json
import time
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

INVENTORY_CSV = Path("dataset_inventory.csv")
OUTPUT_CSV    = Path("processed_dataset.csv")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Sampling config ───────────────────────────────────────────
MAX_NORMAL_FILES  = 2000   # Normal class (over-represented)
MAX_ATTACK_FILES  = 400    # Per attack class (A1/A2/A4/A8/A16)
ROLL_WINDOW       = 5

ATTACK_CODE_NAMES = {
    0: "Normal", 1: "Constant Attack", 2: "Random Offset",
    4: "Constant Offset", 8: "Random Position", 16: "Eventual Stop",
}

FEATURE_COLS = [
    "rcvTime", "sendTime", "pos_x", "pos_y", "pos_z",
    "spd_x", "spd_y", "spd_z",
    "packet_delay", "speed_magnitude", "sender_frequency",
    "message_frequency", "acceleration", "distance_from_origin",
    "rolling_speed_mean", "rolling_speed_variance",
]

# ─────────────────────────────────────────────────────────────
def load_inventory():
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError("Run step1_scan_dataset.py first.")
    inv = pd.read_csv(INVENTORY_CSV)
    print(f"[INFO] Inventory loaded: {len(inv)} entries")
    return inv

def extract_veh_index(filename: str) -> int:
    name = Path(filename).stem
    parts = name.split("-")
    if len(parts) >= 2:
        try: return int(parts[1])
        except: pass
    return -1

def sample_files(inv: pd.DataFrame) -> pd.DataFrame:
    """Stratified sample: cap each attack class."""
    logs = inv[inv["is_ground_truth"].astype(str).str.lower() == "false"].copy()
    logs["attack_code_int"] = pd.to_numeric(logs["attack_code"], errors="coerce").fillna(0).astype(int)

    sampled = []
    for code in logs["attack_code_int"].unique():
        subset = logs[logs["attack_code_int"] == code]
        limit  = MAX_NORMAL_FILES if code == 0 else MAX_ATTACK_FILES
        take   = subset.sample(n=min(limit, len(subset)), random_state=42)
        sampled.append(take)
        name = ATTACK_CODE_NAMES.get(code, str(code))
        print(f"  [SAMPLE] A{code} ({name}): {len(take)} files selected from {len(subset)}")

    result = pd.concat(sampled, ignore_index=True)
    print(f"[INFO] Total files to process: {len(result)}")
    return result

def parse_record(obj, sim_name, veh_index, attack_code):
    try:
        pos = obj.get("pos", [])
        spd = obj.get("spd", [])
        return {
            "sim_name":    sim_name,
            "veh_index":   veh_index,
            "type":        int(obj.get("type", -1)),
            "rcvTime":     float(obj.get("rcvTime", 0)),
            "sendTime":    float(obj.get("sendTime", obj.get("rcvTime", 0))),
            "pos_x":       float(pos[0]) if len(pos) > 0 else 0.0,
            "pos_y":       float(pos[1]) if len(pos) > 1 else 0.0,
            "pos_z":       float(pos[2]) if len(pos) > 2 else 0.0,
            "spd_x":       float(spd[0]) if len(spd) > 0 else 0.0,
            "spd_y":       float(spd[1]) if len(spd) > 1 else 0.0,
            "spd_z":       float(spd[2]) if len(spd) > 2 else 0.0,
            "RSSI":        float(obj["RSSI"]) if "RSSI" in obj else 0.0,
            "attack_code": str(attack_code),
        }
    except Exception:
        return None

def load_files(sampled: pd.DataFrame):
    records = []
    errors  = 0
    for i, (_, row) in enumerate(sampled.iterrows()):
        fp          = row["file_path"]
        sim_name    = str(row["simulation"])
        fname       = str(row["file_name"])
        attack_code = str(row["attack_code"]) if pd.notna(row["attack_code"]) else "0"
        # pandas reads numeric cols as float -> "0.0", "1.0" etc. Fix:
        try:
            attack_code = str(int(float(attack_code))) if attack_code.strip() != "" else "0"
        except:
            attack_code = "0"
        veh_index   = extract_veh_index(fname)
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = parse_record(json.loads(line), sim_name, veh_index, attack_code)
                        if rec: records.append(rec)
                    except: errors += 1
        except Exception as e:
            errors += 1
        if (i+1) % 500 == 0:
            print(f"  ... {i+1}/{len(sampled)} files  ({len(records):,} records)", flush=True)

    print(f"[INFO] Loaded {len(records):,} records  ({errors} errors)")
    return pd.DataFrame(records)

def assign_labels(df):
    def safe_int(c):
        # handles "0", "0.0", "1", "1.0", 1, 1.0 etc.
        try: return int(float(str(c)))
        except: return 0
    df["label"]      = df["attack_code"].apply(safe_int)
    df["label_name"] = df["label"].map(lambda c: ATTACK_CODE_NAMES.get(c, f"Unknown({c})"))
    print(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def engineer_features(df):
    print("[INFO] Engineering features...")
    df["packet_delay"]        = (df["rcvTime"] - df["sendTime"]).clip(lower=0)
    df["speed_magnitude"]     = np.sqrt(df["spd_x"]**2 + df["spd_y"]**2 + df["spd_z"]**2)
    df["distance_from_origin"]= np.sqrt(df["pos_x"]**2 + df["pos_y"]**2 + df["pos_z"]**2)
    df["sender_frequency"]    = df.groupby(["sim_name","veh_index"])["veh_index"].transform("count")

    def sim_freq(grp):
        tr = grp["rcvTime"].max() - grp["rcvTime"].min()
        freq = len(grp) / max(tr, 1e-9)
        return pd.Series([freq]*len(grp), index=grp.index)
    df["message_frequency"] = df.groupby("sim_name", group_keys=False).apply(sim_freq)

    df = df.sort_values(["sim_name","veh_index","rcvTime"]).reset_index(drop=True)

    def rolling_stats(grp):
        roll = grp["speed_magnitude"].rolling(window=ROLL_WINDOW, min_periods=1)
        grp = grp.copy()
        grp["rolling_speed_mean"]     = roll.mean()
        grp["rolling_speed_variance"] = roll.var().fillna(0)
        return grp
    df = df.groupby(["sim_name","veh_index"], group_keys=False).apply(rolling_stats).reset_index(drop=True)

    prev_spd  = df.groupby(["sim_name","veh_index"])["speed_magnitude"].shift(1)
    prev_time = df.groupby(["sim_name","veh_index"])["rcvTime"].shift(1)
    dt = (df["rcvTime"] - prev_time).replace(0, np.nan)
    df["acceleration"] = ((df["speed_magnitude"] - prev_spd) / dt).fillna(0)

    print("[INFO] Feature engineering complete.")
    return df

def clean_data(df):
    before = len(df)
    df.drop_duplicates(inplace=True)
    critical = ["rcvTime","pos_x","pos_y","spd_x","spd_y"]
    df.dropna(subset=critical, how="all", inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    print(f"[INFO] Cleaned: {len(df):,} rows (removed {before-len(df):,})")
    return df.reset_index(drop=True)

def normalise(df):
    scaler   = StandardScaler()
    cols     = [c for c in FEATURE_COLS if c in df.columns]
    df[cols] = scaler.fit_transform(df[cols])
    return df

def save_report(df):
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved: {OUTPUT_CSV}  ({len(df):,} rows, {OUTPUT_CSV.stat().st_size/1e6:.1f} MB)")

    # Class distribution chart
    counts = df["label_name"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VeReMi - Class Distribution (Sampled)", fontsize=14, fontweight="bold")
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("tab10", len(counts)))
    axes[0].set_title("Proportion")
    sns.barplot(x=counts.values, y=counts.index, palette="tab10", ax=axes[1])
    axes[1].set_xlabel("Record Count")
    axes[1].set_title("Counts")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    lines = [
        "="*60, "  VeReMi PREPROCESSING REPORT (Step 2 - Sampled)", "="*60,
        f"  Rows       : {len(df):,}",
        f"  Columns    : {len(df.columns)}",
        f"  Sampling   : {MAX_NORMAL_FILES} normal + {MAX_ATTACK_FILES} per attack class",
        "", "  LABEL DISTRIBUTION", "  "+"-"*40,
    ]
    for name, cnt in counts.items():
        lines.append(f"    {name:<28}: {cnt:>8,}  ({100*cnt/len(df):.2f}%)")
    lines += ["", "="*60]
    rpath = RESULTS_DIR / "step2_preprocessing_report.txt"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report: {rpath}")

def main():
    t0 = time.time()
    print("="*60)
    print("  STEP 2 FAST (Sampled) — Preprocessing Pipeline")
    print("="*60)

    inv     = load_inventory()
    sampled = sample_files(inv)
    df      = load_files(sampled)

    if df.empty:
        print("[ERROR] No records loaded.")
        return

    df = assign_labels(df)
    df = engineer_features(df)
    df = clean_data(df)
    df = normalise(df)
    save_report(df)

    print(f"\n>>> Step 2 FAST complete in {time.time()-t0:.1f}s")
    print(">>> Run: python step3_train_evaluate.py\n")

if __name__ == "__main__":
    main()
