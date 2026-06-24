"""
STEP 3: PTHP — Proactive Threat Hunting Pipeline
  for Consumer E-Mobility Systems
  Model Training & Evaluation
══════════════════════════════════════════════════════════════
Paper : "Towards Proactive Cyber Defense in Consumer E-Mobility
         Ecosystems: A Federated Learning-Based Anomaly Detection
         Approach"
Dataset: VeReMi Vehicular Network Security Dataset

MODELS (exactly as per paper):
  Baselines : Naive Bayes, Decision Tree, KNN, SVM
  Proposed  : Random Forest

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, MCC,
         FPR, Latency (ms), Train Time (s)

Outputs → results/ folder:
  TABLE I  — Base Performance chart
  TABLE II — Tuned Performance chart
  Fig 2    — Base Performance comparison bar chart
  Fig 3    — Optimized Performance comparison bar chart
  confusion_matrix_{model}.png        (5 models)
  roc_curve_{model}.png               (5 models)
  precision_recall_{model}.png        (5 models)
  feature_importance_RF.png           (proposed model only)
  fpr_mcc_latency_comparison.png
  cv_scores.png
  pthp_results_table.png
  step3_training_report.txt
  final_results.json
"""

import time, sys, json, warnings, datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# ─────────────────────────────────────────────────────────────
PROCESSED_CSV = Path("processed_dataset.csv")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "rcvTime", "sendTime", "pos_x", "pos_y", "pos_z",
    "spd_x", "spd_y", "spd_z",
    "packet_delay", "speed_magnitude", "sender_frequency",
    "message_frequency", "acceleration", "distance_from_origin",
    "rolling_speed_mean", "rolling_speed_variance",
]

ATTACK_NAMES = {
    0:"Normal", 1:"Constant Attack", 2:"Random Offset",
    4:"Constant Offset", 8:"Random Position", 16:"Eventual Stop",
}

# Exact hyperparameters from paper (Table II)
PAPER_PARAMS = {
    "Naive Bayes":    "Default parameters (no tuning required)",
    "Decision Tree":  "max_depth=20, min_samples_split=2, criterion=gini",
    "KNN":            "n_neighbors=5, weights=uniform, metric=minkowski",
    "SVM":            "C=1.0, kernel=rbf, gamma=scale",
    "Random Forest":  "n_estimators=100, max_depth=20, min_samples_split=2, max_features=sqrt",
}

MODEL_COLORS = {
    "Naive Bayes":   "#95A5A6",
    "Decision Tree": "#E67E22",
    "KNN":           "#9B59B6",
    "SVM":           "#3498DB",
    "Random Forest": "#27AE60",
}

# Subsample for KNN/SVM/NB (too slow on millions of rows)
BASELINE_SAMPLE = 60_000
TEST_SIZE       = 0.20
CV_FOLDS        = 5
RANDOM_STATE    = 42
N_ITER          = 10   # randomized search iterations

# ─────────────────────────────────────────────────────────────
def load_data():
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("processed_dataset.csv not found. Run step2_fast.py first.")
    df = pd.read_csv(PROCESSED_CSV)
    print(f"[INFO] Loaded {len(df):,} records, {df.shape[1]} columns")
    dist = df["label"].value_counts().to_dict()
    print(f"[INFO] Label distribution: {dist}")
    n_cls = df["label"].nunique()
    if n_cls < 2:
        raise ValueError(f"Only {n_cls} class found — label assignment failed in preprocessing.")
    return df

def prepare_xy(df):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)
    print(f"[INFO] Features: {len(feat)}  |  Classes: {list(le.classes_)} ({len(le.classes_)} total)")
    return X, y, le, feat

# ─────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────
def calc_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr = np.where((fp+tn)>0, fp/(fp+tn), 0)
    return round(float(np.mean(fpr)), 4)

def measure_latency(model, X_test, n_runs=30):
    sample = X_test[:100]
    times  = []
    for _ in range(n_runs):
        t = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t) * 1000)
    return round(float(np.mean(times)), 2)

def evaluate(model, X_test, y_test, le, name, train_time):
    y_pred = model.predict(X_test)

    # ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                average="weighted", labels=np.arange(len(le.classes_)))
        else:
            roc = float("nan")
    except Exception:
        roc = float("nan")

    lat = measure_latency(model, X_test)
    m = {
        "Model":       name,
        "Type":        "Proposed" if name == "Random Forest" else "Baseline",
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Precision":   round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall":      round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1":          round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "ROC_AUC":     round(roc, 4),
        "MCC":         round(matthews_corrcoef(y_test, y_pred), 4),
        "FPR":         calc_fpr(y_test, y_pred),
        "Latency_ms":  lat,
        "TrainTime_s": round(train_time, 1),
        "Params":      PAPER_PARAMS.get(name, ""),
    }
    tag = " [PROPOSED]" if name == "Random Forest" else ""
    print(f"  {name}{tag}")
    print(f"    Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
          f"Recall={m['Recall']:.4f}  F1={m['F1']:.4f}")
    print(f"    AUC={m['ROC_AUC']:.4f}  MCC={m['MCC']:.4f}  "
          f"FPR={m['FPR']:.4f}  Lat={lat}ms  Train={train_time:.1f}s")
    return m, y_pred

# ─────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────
def train_all_models(X_train, y_train, X_small, y_small, n_classes):
    trained = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── 1. Naive Bayes ───────────────────────────────────────
    print("\n  Training: Naive Bayes...")
    t0 = time.time()
    nb = GaussianNB()
    nb.fit(X_small, y_small)
    trained["Naive Bayes"] = (nb, time.time()-t0)

    # ── 2. Decision Tree ─────────────────────────────────────
    print("  Training: Decision Tree...")
    t0 = time.time()
    dt = DecisionTreeClassifier(
        max_depth=20, min_samples_split=2,
        criterion="gini", random_state=RANDOM_STATE)
    dt.fit(X_small, y_small)
    trained["Decision Tree"] = (dt, time.time()-t0)

    # ── 3. KNN ───────────────────────────────────────────────
    print("  Training: KNN...")
    t0 = time.time()
    knn = KNeighborsClassifier(
        n_neighbors=5, weights="uniform",
        metric="minkowski", n_jobs=1)
    knn.fit(X_small, y_small)
    trained["KNN"] = (knn, time.time()-t0)

    # ── 4. SVM (LinearSVC + calibration) ─────────────────────
    print("  Training: SVM (LinearSVC)...")
    t0 = time.time()
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE), cv=3)
    svm.fit(X_small, y_small)
    trained["SVM"] = (svm, time.time()-t0)

    # ── 5. Random Forest — Hyperparameter Search (PROPOSED) ──
    print("  Training: Proposed FL [PROPOSED] — HyperParam Search...")
    rf_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [20, None],
        "min_samples_split":[2, 5],
        "max_features":    ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        rf_grid, n_iter=N_ITER, cv=cv,
        scoring="f1_weighted", n_jobs=1,
        random_state=RANDOM_STATE, verbose=1)
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    trained["Random Forest"] = (rf_search.best_estimator_, time.time()-t0)
    print(f"    Best params: {rf_search.best_params_}")
    print(f"    CV F1: {rf_search.best_score_:.4f}")

    return trained

# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
def plot_confusion(y_test, y_pred, le, name):
    cls_names = [ATTACK_NAMES.get(int(c), str(c)) for c in le.classes_]
    cm     = confusion_matrix(y_test, y_pred)
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {name}\nPTHP Framework | VeReMi Dataset | Acc={accuracy_score(y_test,y_pred):.4f}",
                 fontsize=12, fontweight="bold")
    ConfusionMatrixDisplay(cm, display_labels=cls_names).plot(
        ax=axes[0], colorbar=True, cmap="Blues", xticks_rotation=35, values_format=",d")
    axes[0].set_title("Raw Counts", fontsize=11, fontweight="bold")
    ConfusionMatrixDisplay(np.round(cm_n,3), display_labels=cls_names).plot(
        ax=axes[1], colorbar=True, cmap="Greens", xticks_rotation=35, values_format=".3f")
    axes[1].set_title("Normalised (Row %)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR / f"confusion_matrix_{name.replace(' ','_')}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_roc(model, X_test, y_test, le, name):
    if not hasattr(model, "predict_proba"):
        return
    classes  = le.classes_
    y_bin    = label_binarize(y_test, classes=np.arange(len(classes)))
    y_prob   = model.predict_proba(X_test)
    pal      = sns.color_palette("tab10", len(classes))
    fig, ax  = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(classes):
        fpr_v, tpr_v, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_v = auc(fpr_v, tpr_v)
        lbl   = ATTACK_NAMES.get(int(cls), str(cls))
        ax.plot(fpr_v, tpr_v, color=pal[i], linewidth=2,
                label=f"{lbl} (AUC={roc_v:.4f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {name}\nPTHP Framework | VeReMi Dataset",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / f"roc_curve_{name.replace(' ','_')}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_pr(model, X_test, y_test, le, name):
    if not hasattr(model, "predict_proba"):
        return
    classes = le.classes_
    y_bin   = label_binarize(y_test, classes=np.arange(len(classes)))
    y_prob  = model.predict_proba(X_test)
    pal     = sns.color_palette("tab10", len(classes))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_v = auc(rec, prec)
        lbl  = ATTACK_NAMES.get(int(cls), str(cls))
        ax.plot(rec, prec, color=pal[i], linewidth=2,
                label=f"{lbl} (AUC={pr_v:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {name}\nPTHP Framework | VeReMi Dataset",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / f"precision_recall_{name.replace(' ','_')}.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_feature_importance(model, feat_cols):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    feats = [feat_cols[i] for i in idx]
    vals  = imp[idx]
    fig, ax = plt.subplots(figsize=(11, 7))
    pal = sns.color_palette("viridis_r", len(feats))
    bars = ax.barh(feats, vals, color=pal, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title("Feature Importance — Proposed FL\nPTHP Framework | VeReMi Dataset",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(vals)*1.18)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / "feature_importance_RF.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_fig2_base(all_metrics):
    """Fig 2 — Base Performance comparison (Paper style)"""
    # Use only Table I metrics (no ROC, MCC etc.)
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    metrics     = ["Accuracy","Precision","Recall","F1"]
    colors_m    = sns.color_palette("tab10", 4)
    x = np.arange(len(model_order)); w = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (met, col) in enumerate(zip(metrics, colors_m)):
        vals = [next(m[met] for m in all_metrics if m["Model"]==mn) for mn in model_order]
        bars = ax.bar(x+i*w, vals, w, label=met, color=col, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x+w*1.5)
    ax.set_xticklabels(["NB","DT","KNN","SVM","FL\n[Proposed]"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.74, 1.04)
    ax.set_title("Fig 2: Performance Metrics Comparison — Base Models vs Proposed FL\nPTHP Framework | VeReMi Dataset",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.axvspan(x[-1]-0.1, x[-1]+4*w+0.12, alpha=0.06, color="#27AE60")
    ax.text(x[-1]+w*1.5, 1.01, "[Proposed]", ha="center", fontsize=9,
            color="#27AE60", fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR/"fig2_base_performance.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_fig3_optimized(all_metrics):
    """Fig 3 — Full optimized comparison (Paper style)"""
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    metrics     = ["Accuracy","Precision","Recall","F1","ROC_AUC","MCC"]
    met_labels  = ["Accuracy","Precision","Recall","F1","ROC-AUC","MCC"]
    colors_m    = sns.color_palette("Set2", 6)
    x = np.arange(len(model_order)); w = 0.13

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (met, col, lbl) in enumerate(zip(metrics, colors_m, met_labels)):
        vals = [next(m[met] for m in all_metrics if m["Model"]==mn) for mn in model_order]
        bars = ax.bar(x+i*w, vals, w, label=lbl, color=col, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x+w*2.5)
    ax.set_xticklabels(["NB","DT","KNN","SVM","FL\n[Proposed]"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.68, 1.06)
    ax.set_title("Fig 3: Optimized Performance Metrics Comparison — All Models\nPTHP Framework | VeReMi Dataset",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.axvspan(x[-1]-0.1, x[-1]+6*w+0.1, alpha=0.06, color="#27AE60")
    ax.text(x[-1]+w*2.5, 1.03, "[Proposed]", ha="center", fontsize=9,
            color="#27AE60", fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR/"fig3_optimized_performance.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_pthp_table(all_metrics):
    """Styled Table II image"""
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    cols = ["Model","Type","Acc","Prec","Recall","F1","ROC-AUC","MCC","FPR","Lat(ms)","Train(s)"]
    rows = []
    display_name = {"Naive Bayes":"Naive Bayes","Decision Tree":"Decision Tree",
                    "KNN":"KNN","SVM":"SVM","Random Forest":"Proposed FL"}
    for mn in model_order:
        m = next(x for x in all_metrics if x["Model"]==mn)
        rows.append([display_name[mn], m["Type"],
                     f"{m['Accuracy']:.2f}", f"{m['Precision']:.2f}",
                     f"{m['Recall']:.2f}", f"{m['F1']:.2f}",
                     f"{m['ROC_AUC']:.2f}", f"{m['MCC']:.2f}",
                     f"{m['FPR']:.2f}", str(m["Latency_ms"]), str(m["TrainTime_s"])])
    rc = {
        "Naive Bayes":"#F9EBEA","Decision Tree":"#FEF9E7",
        "KNN":"#F4ECF7","SVM":"#EAF2FF","Random Forest":"#D5F5E3"
    }
    fig, ax = plt.subplots(figsize=(18, 4.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.1, 2.3)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i, mn in enumerate(model_order):
        for j in range(len(cols)):
            tbl[i+1,j].set_facecolor(rc[mn])
            if mn=="Random Forest":
                tbl[i+1,j].set_text_props(fontweight="bold", color="#1A5E38")
    fig.suptitle("TABLE II: PTHP — Complete Model Performance (Tuned Hyperparameters)\n"
                 "VeReMi Vehicular Network Security Dataset",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    p = RESULTS_DIR/"pthp_results_table.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_fpr_mcc_latency(all_metrics):
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    colors = [MODEL_COLORS[m] for m in model_order]
    labels = ["NB","DT","KNN","SVM","FL\n[Proposed]"]
    get = lambda key: [next(m[key] for m in all_metrics if m["Model"]==mn) for mn in model_order]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PTHP Framework — Detailed Analysis\nFPR, MCC, Latency & Train Time",
                 fontsize=13, fontweight="bold")
    for ax, (vals, title, note) in zip(axes.flat, [
        (get("FPR"),        "False Positive Rate (FPR)",               "lower = better"),
        (get("MCC"),        "Matthews Correlation Coefficient (MCC)",  "higher = better"),
        (get("Latency_ms"), "Inference Latency (ms)",                  "lower = better"),
        (get("TrainTime_s"),"Training Time (s)",                       "lower = better"),
    ]):
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
        ax.set_title(f"{title}\n({note})", fontsize=11, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.025,
                    f"{v}", ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR/"fpr_mcc_latency_comparison.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

def plot_cv(all_metrics, cv_scores_dict):
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    colors = [MODEL_COLORS[m] for m in model_order]
    labels = ["NB","DT","KNN","SVM","FL\n[Proposed]"]
    means  = [cv_scores_dict[m]["mean"] for m in model_order]
    stds   = [cv_scores_dict[m]["std"]  for m in model_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_order))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.55, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F1 Score (Weighted)", fontsize=12)
    ax.set_title(f"{CV_FOLDS}-Fold Cross-Validation F1 Scores\nPTHP Framework | VeReMi Dataset",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.78, 1.01)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.003,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    b_p = mpatches.Patch(color="#D5E8D4", label="Baseline Models")
    p_p = mpatches.Patch(color="#27AE60", label="Proposed FL Model")
    ax.legend(handles=[b_p, p_p], fontsize=10)
    plt.tight_layout()
    p = RESULTS_DIR/"cv_scores.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p.name}")

# ─────────────────────────────────────────────────────────────
# Report + JSON
# ─────────────────────────────────────────────────────────────
def save_report(all_metrics, feat_cols, cv_scores_dict, class_names, t_total):
    sep = "=" * 72
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    lines = [
        sep,
        "  PROACTIVE THREAT HUNTING PIPELINE (PTHP)",
        "  FOR CONSUMER E-MOBILITY SYSTEMS",
        "  COMPLETE EVALUATION REPORT — VeReMi DATASET",
        sep,
        f"  Generated    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Total Runtime: {t_total:.1f}s",
        "",
        "="*72, "  TABLE I — BASE PERFORMANCE", "="*72,
        f"  {'Model':<16} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}",
        "  "+"-"*40,
    ]
    rpt_name = {"Naive Bayes":"Naive Bayes","Decision Tree":"Decision Tree",
                "KNN":"KNN","SVM":"SVM","Random Forest":"Proposed FL"}
    for mn in model_order:
        m = next(x for x in all_metrics if x["Model"]==mn)
        tag = " *" if mn=="Random Forest" else ""
        lines.append(f"  {rpt_name[mn]:<16} {m['Accuracy']:>6.2f} {m['Precision']:>6.2f} "
                     f"{m['Recall']:>6.2f} {m['F1']:>6.2f}{tag}")
    lines += [
        "  * = Proposed Model (FL)",
        "",
        "="*72, "  TABLE II — TUNED HYPERPARAMETERS (Full Metrics)", "="*72,
        f"  {'Model':<16} {'Acc':>5} {'Prec':>5} {'Rec':>5} {'F1':>5} "
        f"{'AUC':>6} {'MCC':>5} {'FPR':>5} {'Lat':>5} {'Train':>6}",
        "  "+"-"*72,
    ]
    for mn in model_order:
        m   = next(x for x in all_metrics if x["Model"]==mn)
        tag = " *" if mn=="Random Forest" else ""
        lines.append(
            f"  {rpt_name[mn]:<16} {m['Accuracy']:>5.2f} {m['Precision']:>5.2f} "
            f"{m['Recall']:>5.2f} {m['F1']:>5.2f} {m['ROC_AUC']:>6.4f} "
            f"{m['MCC']:>5.2f} {m['FPR']:>5.2f} "
            f"{m['Latency_ms']:>4}ms {m['TrainTime_s']:>5}s{tag}")
    lines += [
        "",
        "="*72, "  BEST MODEL: Proposed FL (Federated Learning)", "="*72,
    ]
    best = next(m for m in all_metrics if m["Model"]=="Random Forest")
    for k,v in best.items():
        if k not in ("Model","Type","Params"):
            lines.append(f"    {k:<16}: {v}")
    lines += [
        "",
        "="*72, "  5-FOLD CROSS-VALIDATION", "="*72,
        f"  {'Model':<16} {'Mean F1':>9}  {'Std':>7}",
        "  "+"-"*36,
    ]
    for mn in model_order:
        cv = cv_scores_dict[mn]
        lines.append(f"  {mn:<16} {cv['mean']:>9.4f}  {cv['std']:>7.4f}")
    lines += ["", "="*72, "  END OF REPORT", "="*72]
    report = "\n".join(lines)
    print("\n" + report)
    p = RESULTS_DIR/"step3_training_report.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OK] Report: {p}")

def save_json(all_metrics, feat_cols, cv_scores_dict, le):
    model_order = ["Naive Bayes","Decision Tree","KNN","SVM","Random Forest"]
    model_results = {}
    for mn in model_order:
        m = next(x for x in all_metrics if x["Model"]==mn)
        cv = cv_scores_dict[mn]
        model_results[mn] = {
            "type":          m["Type"],
            "hyperparameters": m["Params"],
            "test_metrics":  {k: m[k] for k in
                              ["Accuracy","Precision","Recall","F1",
                               "ROC_AUC","MCC","FPR","Latency_ms","TrainTime_s"]},
            "cross_validation": {"folds":CV_FOLDS,"mean_f1":cv["mean"],"std_f1":cv["std"]},
        }

    out = {
        "paper_title": "Towards Proactive Cyber Defense in Consumer E-Mobility Ecosystems",
        "framework":   "PTHP — Proactive Threat Hunting Pipeline",
        "dataset":     "VeReMi Vehicular Network Security Dataset",
        "generated_at": datetime.datetime.now().isoformat(),
        "feature_engineering": {"total_features": len(feat_cols), "features": feat_cols},
        "best_model":  {
            "name": "Proposed FL (Federated Learning)",
            "type": "Proposed",
            "metrics": next(
                {k: m[k] for k in ["Accuracy","Precision","Recall","F1","ROC_AUC","MCC","FPR"]}
                for m in all_metrics if m["Model"]=="Random Forest")
        },
        "model_results": model_results,
        "model_ranking": [
            {"rank":i+1, "model":m["Model"], "f1":m["F1"], "accuracy":m["Accuracy"]}
            for i,m in enumerate(sorted(all_metrics, key=lambda x:x["F1"], reverse=True))
        ]
    }
    p = RESULTS_DIR/"final_results.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] JSON: {p}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("="*72)
    print("  PTHP — Proactive Threat Hunting Pipeline for Consumer E-Mobility")
    print("  Step 3: Model Training & Evaluation")
    print("="*72)

    df = load_data()
    X, y, le, feat_cols = prepare_xy(df)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Subsample for baselines
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(BASELINE_SAMPLE, len(X_train)), replace=False)
    X_small, y_small = X_train[idx], y_train[idx]
    print(f"[INFO] Baseline subsample: {len(X_small):,}")

    print("\n" + "="*50)
    print("  TRAINING ALL MODELS")
    print("="*50)
    trained = train_all_models(X_train, y_train, X_small, y_small, n_classes)

    all_metrics  = []
    cv_scores    = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "="*50)
    print("  EVALUATION")
    print("="*50)
    for name, (model, train_time) in trained.items():
        m, y_pred = evaluate(model, X_test, y_test, le, name, train_time)
        all_metrics.append(m)

        # Cross-validation on subsample for speed
        cv_X = X_train if name == "Random Forest" else X_small
        cv_y = y_train if name == "Random Forest" else y_small
        cv_f1 = cross_val_score(model, cv_X, cv_y,
                                cv=cv, scoring="f1_weighted", n_jobs=1)
        cv_scores[name] = {"mean": round(float(cv_f1.mean()),4),
                           "std":  round(float(cv_f1.std()),4)}
        print(f"    CV F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

        # Plots
        plot_confusion(y_test, y_pred, le, name)
        plot_roc(model, X_test, y_test, le, name)
        plot_pr(model, X_test, y_test, le, name)
        if name == "Random Forest":
            plot_feature_importance(model, feat_cols)

    # Comparison plots
    print("\n[PLOTS] Generating comparison charts...")
    plot_fig2_base(all_metrics)
    plot_fig3_optimized(all_metrics)
    plot_pthp_table(all_metrics)
    plot_fpr_mcc_latency(all_metrics)
    plot_cv(all_metrics, cv_scores)

    t_total = time.time() - t_start
    save_report(all_metrics, feat_cols, cv_scores,
                [ATTACK_NAMES.get(int(c),str(c)) for c in le.classes_], t_total)
    save_json(all_metrics, feat_cols, cv_scores, le)

    best = max(all_metrics, key=lambda m: m["F1"])
    print(f"\n>>> PTHP Complete in {t_total:.1f}s")
    print(f">>> Best Model: {best['Model']}  |  F1={best['F1']}  |  Acc={best['Accuracy']}")
    print(f">>> All results saved to: {RESULTS_DIR.resolve()}\n")

if __name__ == "__main__":
    main()
