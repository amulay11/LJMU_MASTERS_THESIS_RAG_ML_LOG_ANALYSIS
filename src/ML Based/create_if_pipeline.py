"""
Generator — creates E01_B_BGL_IsolationForest_Pipeline.ipynb.

Structural differences from the DeepSVDD reference:
  - torch / nn removed; IsolationForest (sklearn) replaces the neural network
  - Model section: IsolationForest.fit() + compute_if_scores() (no epochs)
  - Threshold calibration: identical F-beta PR-curve logic, but on IF scores
  - Plot 10a: validation score distribution (replaces training-loss curve)
  - Plot 10b-10e: identical structure to reference (score dist, PR, CM, ROC)
  - Anomaly score convention: negated score_samples() so higher = more anomalous
    (matches DeepSVDD distance convention, keeping calibration code unchanged)

Everything else is a verbatim copy from E01_A_BGL_DeepSVDD_Pipeline:
  data loading, splits, embeddings, evaluate_predictions, DATASET_CONFIGS.
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# Cell 0 — Title
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
"""# Log Anomaly Detection (BGL Dataset): Isolation Forest Pipeline
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Package installation  (torch removed — not needed for IF)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
'''# =============================================================================
# CELL 1 — PACKAGE INSTALLATION
# =============================================================================

import subprocess, sys

def install_packages(packages):
    for pkg in packages:
        print(f"Installing: {pkg}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: may have failed — {result.stderr[:200]}")
        else:
            print("  OK")

install_packages([
    "pandas", "numpy", "scikit-learn",
    "matplotlib", "seaborn",
    "sentence-transformers",
])
print("\\nAll packages ready.")
'''
))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
'''# =============================================================================
# CELL 2 — IMPORTS AND CONFIGURATION
# =============================================================================

import os, re, warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Dataset selection ─────────────────────────────────────────────────────────
DATASET = "BGL"   # change to "HDFS" to run on the HDFS dataset

# ── Dataset-specific configurations ──────────────────────────────────────────
DATASET_CONFIGS = {
    "BGL": {
        "structured_log_path": "../../Datasets/BGL/Sample/BGL_2k.log_structured.csv",
        "templates_path":      "../../Datasets/BGL/Sample/BGL_2k.log_templates.csv",
        "label_col":     "Label",
        "normal_value":  "-",
        "event_id_col":  "EventId",
        "component_col": "Component",
        "level_col":     "Level",
        "content_col":   "Content",
        "template_col":  "EventTemplate",
    },
    "HDFS": {
        "traces_path":    "../../Datasets/HDFS/Full_HDFS_v1/preprocessed/Event_traces.csv",
        "templates_path": "../../Datasets/HDFS/Full_HDFS_v1/preprocessed/HDFS.log_templates.csv",
        "label_col":    "Label",
        "normal_value": "Success",
        "features_col": "Features",
        # Sampling caps (see E01_A_HDFS notebook for analysis rationale)
        "normal_sample_cap":  10000,
        "anomaly_type_col":   "Type",
        "anomaly_n_per_type": 20,
    },
}

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM        = 384

# ── Isolation Forest hyper-parameters ────────────────────────────────────────
IF_N_ESTIMATORS = 200    # number of isolation trees
IF_MAX_SAMPLES  = "auto" # samples per tree: "auto" = min(256, n_train)
IF_N_JOBS       = -1     # -1 = use all available CPU cores

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20

# ── Threshold calibration ─────────────────────────────────────────────────────
PSEUDO_ANOMALY_PERCENTILE = 93   # top 7% of val scores = pseudo-anomalies
FBETA = 0.5                      # beta < 1 weights precision over recall

print(f"  Dataset           : {DATASET}")
print(f"  Embedding model   : {EMBED_MODEL_NAME}")
print(f"  IF n_estimators   : {IF_N_ESTIMATORS}")
print(f"  IF max_samples    : {IF_MAX_SAMPLES}")
print(f"  IF n_jobs         : {IF_N_JOBS}")
print(f"  Train ratio       : {TRAIN_RATIO}")
print(f"  Val ratio         : {VAL_RATIO}")
print(f"  Test normals      : remaining {1.0 - TRAIN_RATIO - VAL_RATIO:.0%} of normals")
print(f"  Test anomalies    : ALL available (no cap)")
print(f"  F-beta            : {FBETA}")


# =============================================================================
# GENERIC DATA LOADING
#
# Output contract (identical for every dataset):
#   log_text     (str)  — enriched text fed to the sentence transformer
#   is_normal    (bool) — True if the sample is labelled Normal
#   binary_label (int)  — 0 = Normal, 1 = Anomalous
#
# BGL  loader : line-level;  one row = one log line
# HDFS loader : block-level; one row = one block trace (event sequence)
# =============================================================================


# ── Dataset-size sampling (applied before log_text construction) ──────────────

def _apply_sampling(df, config, random_seed=42):
    """Apply per-dataset sampling caps to keep dataset size manageable.

    Called inside each loader BEFORE building log_text, so the expensive
    DataFrame.apply(...) step only runs on the sampled rows.

    Normal blocks   : random sample up to config["normal_sample_cap"].
    Anomalous blocks: stratified by config["anomaly_type_col"], taking up to
                      config["anomaly_n_per_type"] samples per category.
    For datasets without caps (e.g. BGL-2k), all values are None and this
    function returns df unchanged.
    """
    normal_cap    = config.get("normal_sample_cap")
    anom_type_col = config.get("anomaly_type_col")
    n_per_type    = config.get("anomaly_n_per_type")
    anom_cap      = config.get("anomaly_sample_cap")

    df_n = df[df["is_normal"]].copy()
    df_a = df[~df["is_normal"]].copy()

    if normal_cap is not None and len(df_n) > normal_cap:
        df_n = df_n.sample(n=normal_cap, random_state=random_seed)

    if anom_type_col and n_per_type:
        parts = [
            grp.sample(n=min(n_per_type, len(grp)), random_state=random_seed)
            for _, grp in df_a.groupby(anom_type_col)
            if len(grp) > 0
        ]
        df_a = pd.concat(parts) if parts else df_a
    elif anom_cap is not None and len(df_a) > anom_cap:
        df_a = df_a.sample(n=anom_cap, random_state=random_seed)

    return pd.concat([df_n, df_a], ignore_index=True)


# ── BGL helpers ───────────────────────────────────────────────────────────────

def build_bgl_log_text(row, config):
    """Construct enriched text for a single BGL log line.

    Format: [COMPONENT] [LEVEL] <content> | Template: <drain_template>
    """
    component = str(row.get(config["component_col"], "")).strip()
    level     = str(row.get(config["level_col"],     "")).strip()
    content   = str(row.get(config["content_col"],   "")).strip()
    template  = str(row.get(config["template_col"],  "")).strip()
    return f"[{component}] [{level}] {content} | Template: {template}"


def load_bgl_data(config):
    """Load BGL structured log CSV, merge templates, build log_text."""
    print("  Loading BGL structured log ...")
    df_logs      = pd.read_csv(config["structured_log_path"])
    df_templates = pd.read_csv(config["templates_path"])
    print(f"    Rows: {len(df_logs)}, Templates: {len(df_templates)}")

    df = df_logs.merge(
        df_templates, on=config["event_id_col"],
        how="left", suffixes=("", "_tmpl"),
    )
    df["is_normal"]    = df[config["label_col"]] == config["normal_value"]
    df["binary_label"] = (~df["is_normal"]).astype(int)
    df = _apply_sampling(df, config)
    df["log_text"]     = df.apply(
        lambda row: build_bgl_log_text(row, config), axis=1
    )
    return df


# ── HDFS helpers ──────────────────────────────────────────────────────────────

def _clean_template(text):
    """Remove Drain wildcards [*] and collapse whitespace."""
    cleaned = re.sub(r\'\\[\\*\\]\', \'\', text)
    return re.sub(r\'\\s+\', \' \', cleaned).strip()


def build_hdfs_log_text(row, config, template_lookup):
    """Construct enriched text for an HDFS block trace.

    Parses the ordered event-ID sequence, resolves to Drain templates,
    joins with \' -> \' to represent the block lifecycle.
    """
    features_str = str(row.get(config["features_col"], "[]"))
    event_ids    = re.findall(r\'E\\d+\', features_str)
    seen, unique_templates = set(), []
    for eid in event_ids:
        if eid not in seen:
            seen.add(eid)
            tmpl = template_lookup.get(eid, eid)
            unique_templates.append(_clean_template(tmpl))
    return "HDFS Block Trace | " + " -> ".join(unique_templates)


def load_hdfs_data(config):
    """Load HDFS Event_traces.csv, build block-level log_text."""
    print("  Loading HDFS block traces ...")
    df_traces    = pd.read_csv(config["traces_path"])
    df_templates = pd.read_csv(config["templates_path"])
    print(f"    Block traces: {len(df_traces)}, Templates: {len(df_templates)}")

    template_lookup = dict(
        zip(df_templates["EventId"], df_templates["EventTemplate"])
    )
    df_traces["is_normal"]    = (
        df_traces[config["label_col"]] == config["normal_value"]
    )
    df_traces["binary_label"] = (~df_traces["is_normal"]).astype(int)

    df_traces = _apply_sampling(df_traces, config)
    n_s = int(df_traces["is_normal"].sum())
    n_a = int((~df_traces["is_normal"]).sum())
    print(f"    After sampling  : {len(df_traces)} blocks "
          f"({n_s} normal, {n_a} anomalous)")

    df_traces["log_text"] = df_traces.apply(
        lambda row: build_hdfs_log_text(row, config, template_lookup), axis=1
    )
    return df_traces


# ── Public entry point ────────────────────────────────────────────────────────

def load_data(dataset_name, configs):
    """Dispatch to the dataset-specific loader.

    Returns a DataFrame guaranteed to have:
        log_text (str), is_normal (bool), binary_label (int).
    """
    if dataset_name not in configs:
        raise ValueError(
            f"Unknown dataset \'{dataset_name}\'. "
            f"Available: {list(configs.keys())}"
        )
    if dataset_name == "BGL":
        return load_bgl_data(configs[dataset_name])
    elif dataset_name == "HDFS":
        return load_hdfs_data(configs[dataset_name])


# ── Execute ───────────────────────────────────────────────────────────────────

print(f"Loading dataset: {DATASET}")
df = load_data(DATASET, DATASET_CONFIGS)

n_normal    = int(df["is_normal"].sum())
n_anomalous = int((~df["is_normal"]).sum())
print(f"\\n  Total samples     : {len(df)}")
print(f"  Normal            : {n_normal}  ({n_normal / len(df) * 100:.1f}%)")
print(f"  Anomalous         : {n_anomalous}  ({n_anomalous / len(df) * 100:.1f}%)")
print(f"\\n  Sample log_text:\\n    {df[\'log_text\'].iloc[0]}")


# =============================================================================
# GENERIC TRAIN / VALIDATION / TEST SPLIT
#
# Train and validation : normal samples only (one-class learning)
# Test                 : remaining normals + ALL anomalous samples
# =============================================================================


def create_splits(df, train_ratio=0.60, val_ratio=0.20, random_seed=42):
    """Split dataset into train, validation, and test subsets.

    Train and validation contain only normal samples. Test contains the
    remaining normals PLUS ALL anomalous samples — no cap applied.
    """
    assert train_ratio + val_ratio <= 1.0

    df_normal    = (df[df["is_normal"]]
                    .sample(frac=1, random_state=random_seed)
                    .reset_index(drop=True))
    df_anomalous = df[~df["is_normal"]].reset_index(drop=True)

    n_total = len(df_normal)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_df    = df_normal.iloc[:n_train].reset_index(drop=True)
    val_df      = df_normal.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test_normal = df_normal.iloc[n_train + n_val:].reset_index(drop=True)

    test_df = (
        pd.concat([test_normal, df_anomalous], ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    return train_df, val_df, test_df


# ── Execute ───────────────────────────────────────────────────────────────────

print("Creating train / validation / test splits ...")
train_df, val_df, test_df = create_splits(
    df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED,
)

n_test_normal    = int(test_df["is_normal"].sum())
n_test_anomalous = int((~test_df["is_normal"]).sum())

print(f"\\n  Train  : {len(train_df):>6}  normal only")
print(f"  Val    : {len(val_df):>6}  normal only  (threshold calibration)")
print(f"  Test   : {len(test_df):>6}  total")
print(f"           {n_test_normal:>6}  normal     ({n_test_normal / len(test_df) * 100:.1f}%)")
print(f"           {n_test_anomalous:>6}  anomalous  ({n_test_anomalous / len(test_df) * 100:.1f}%)")


# =============================================================================
# SENTENCE EMBEDDINGS
#
# all-MiniLM-L6-v2 encodes BGL and HDFS log texts into the same 384-dim space.
# =============================================================================

print(f"Loading sentence transformer \'{EMBED_MODEL_NAME}\' ...")
embedder  = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print(f"  Embedding dimension : {EMBED_DIM}")

print(f"\\nEncoding train set ({len(train_df)} samples) ...")
train_embeddings = embedder.encode(
    train_df["log_text"].tolist(), batch_size=64, show_progress_bar=True
)

print(f"Encoding validation set ({len(val_df)} samples) ...")
val_embeddings = embedder.encode(
    val_df["log_text"].tolist(), batch_size=64, show_progress_bar=True
)

print(f"Encoding test set ({len(test_df)} samples) ...")
test_embeddings = embedder.encode(
    test_df["log_text"].tolist(), batch_size=64, show_progress_bar=True
)

print(f"\\n  Train embeddings  : {train_embeddings.shape}")
print(f"  Val embeddings    : {val_embeddings.shape}")
print(f"  Test embeddings   : {test_embeddings.shape}")


# =============================================================================
# ISOLATION FOREST MODEL AND TRAINING
#
# IsolationForest is trained on normal embeddings only (one-class approach).
# It partitions the embedding space with random feature splits; anomalies
# are isolated with fewer splits (shorter average path lengths) and receive
# lower raw anomaly scores from score_samples().
#
# Anomaly score convention:
#   score_samples() output is negated so that the resulting anomaly_score
#   is HIGHER for anomalies — matching the "distance" convention used in
#   Deep SVDD and keeping all downstream threshold calibration code identical.
#
# No training loop / epoch losses:
#   IF fits in a single call and requires no iterative optimisation.
#   The validation set is retained for threshold calibration only.
# =============================================================================


def compute_if_scores(model, embeddings):
    """Compute anomaly scores using the fitted Isolation Forest.

    Negates score_samples() so that higher score = more anomalous.
    This mirrors the DeepSVDD distance convention:
      DeepSVDD : distance from hypersphere centre (higher = more anomalous)
      IF       : -score_samples()              (higher = more anomalous)

    Args:
        model      : Fitted IsolationForest instance.
        embeddings : (N, embed_dim) NumPy float32 array.

    Returns:
        (N,) NumPy array of anomaly scores. Higher = more anomalous.
    """
    return -model.score_samples(embeddings)


# ── Fit ───────────────────────────────────────────────────────────────────────

print("Fitting Isolation Forest on training embeddings ...")
print(f"  n_estimators : {IF_N_ESTIMATORS}")
print(f"  max_samples  : {IF_MAX_SAMPLES}")
print(f"  n_jobs       : {IF_N_JOBS}")

if_model = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    max_samples=IF_MAX_SAMPLES,
    contamination="auto",   # internal boundary only; we set our own threshold below
    random_state=RANDOM_SEED,
    n_jobs=IF_N_JOBS,
)
if_model.fit(train_embeddings)
print(f"\\n  Fitted on {len(train_embeddings)} normal training samples.")
print(f"  Actual max_samples per tree : {if_model.max_samples_}")


# =============================================================================
# THRESHOLD CALIBRATION (VALIDATION SET)
#
# Identical logic to Deep SVDD:
#   1. Score held-out normal logs with compute_if_scores().
#   2. Label the top PSEUDO_ANOMALY_PERCENTILE% as pseudo-anomalous.
#   3. Sweep thresholds via precision_recall_curve and pick the one that
#      maximises F-beta (beta=0.5) — weighting precision over recall.
# =============================================================================

print("Computing validation-set anomaly scores ...")
val_scores = compute_if_scores(if_model, val_embeddings)

print(f"  Val score stats (normal logs only):")
print(f"    Min    : {val_scores.min():.4f}")
print(f"    Median : {np.median(val_scores):.4f}")
print(f"    Max    : {val_scores.max():.4f}")

pseudo_labels = (
    val_scores >= np.percentile(val_scores, PSEUDO_ANOMALY_PERCENTILE)
).astype(int)

pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(
    pseudo_labels, val_scores
)

beta_sq      = FBETA ** 2
fbeta_scores = (
    (1 + beta_sq) * pr_precisions * pr_recalls
    / (beta_sq * pr_precisions + pr_recalls + 1e-9)
)
best_idx     = int(np.argmax(fbeta_scores))
if_threshold = (
    float(pr_thresholds[best_idx])
    if best_idx < len(pr_thresholds)
    else float(pr_thresholds[-1])
)

print(f"\\n  Threshold selection via F-beta (beta={FBETA}) on validation PR curve:")
print(f"    Best threshold       : {if_threshold:.4f}")
print(
    f"    At threshold         : Precision={pr_precisions[best_idx]:.3f}  "
    f"Recall={pr_recalls[best_idx]:.3f}  "
    f"F{FBETA}={fbeta_scores[best_idx]:.3f}"
)
print(f"    Pseudo-anomaly count : {pseudo_labels.sum()}  "
      f"(top {100 - PSEUDO_ANOMALY_PERCENTILE}% of {len(val_scores)} val normals)")


# =============================================================================
# EVALUATION
#
# evaluate_predictions() is fully generic — identical to Deep SVDD pipeline.
# =============================================================================


def evaluate_predictions(y_true, y_pred, dataset_name, model_name="Isolation Forest"):
    """Compute and print standard one-class anomaly detection metrics.

    Args:
        y_true       : List / array of ground-truth labels (0=Normal, 1=Anomalous).
        y_pred       : List / array of predicted labels.
        dataset_name : Used for display only.
        model_name   : Used for display only.

    Returns:
        dict with accuracy, precision, recall, f1, tp, tn, fp, fn, cm.
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    accuracy  = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list, pos_label=1, zero_division=0)
    recall    = recall_score(y_true_list, y_pred_list, pos_label=1, zero_division=0)
    f1        = f1_score(y_true_list, y_pred_list, pos_label=1, zero_division=0)
    cm        = confusion_matrix(y_true_list, y_pred_list)
    tn, fp, fn, tp = cm.ravel()

    n_normal    = y_true_list.count(0)
    n_anomalous = y_true_list.count(1)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\\n" + "=" * 64)
    print(f"  EVALUATION  --  {model_name}  [{dataset_name}]")
    print("=" * 64)
    print(f"  Test set size          : {len(y_true_list)}")
    print(f"    Normal   (neg)       : {n_normal}  "
          f"({n_normal / len(y_true_list) * 100:.1f}%)")
    print(f"    Anomalous (pos)      : {n_anomalous}  "
          f"({n_anomalous / len(y_true_list) * 100:.1f}%)")
    print("-" * 64)
    print(f"  TP (caught anomalies)  : {tp}")
    print(f"  TN (correct normals)   : {tn}")
    print(f"  FP (false alarms)      : {fp}")
    print(f"  FN (missed anomalies)  : {fn}")
    print("-" * 64)
    print(f"  Accuracy               : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision              : {precision:.4f}")
    print(f"  Recall                 : {recall:.4f}")
    print(f"  F1-Score               : {f1:.4f}")
    print(f"  FPR (false alarm rate) : {fpr:.4f}")
    print(f"  FNR (miss rate)        : {fnr:.4f}")
    print("=" * 64)

    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn), cm=cm,
    )


# ── Score test set and evaluate ───────────────────────────────────────────────

print("Scoring test set ...")
test_scores = compute_if_scores(if_model, test_embeddings)

y_pred = (test_scores > if_threshold).astype(int).tolist()
y_true = test_df["binary_label"].tolist()

metrics = evaluate_predictions(y_true, y_pred, DATASET)

# ── AUC-ROC ───────────────────────────────────────────────────────────────────
# IF anomaly score = -score_samples(); higher = more anomalous.
# Used directly as the continuous ranking score for AUC computation.
auroc = roc_auc_score(y_true, test_scores.tolist())
fpr_roc, tpr_roc, roc_thresholds = roc_curve(y_true, test_scores)
metrics["auroc"] = auroc
print(f"  AUC-ROC                : {auroc:.4f}")


# =============================================================================
# VISUALISATIONS
#
# plot_results() is fully generic across BGL and HDFS.
# PNG filenames are prefixed with dataset_name.lower() so runs do not
# overwrite each other.
#
# Plots:
#   10a  Validation score distribution with calibrated threshold
#        (replaces training-loss curve from Deep SVDD — IF has no epochs)
#   10b  Test-set score distributions: Normal vs Anomalous
#   10c  Precision-Recall curve with F-beta threshold marker
#   10d  Confusion matrix
#   10e  ROC curve with AUC and operating-point marker
# =============================================================================


def plot_results(
    val_scores,
    test_scores,
    y_true_arr,
    threshold,
    pr_recalls,
    pr_precisions,
    fbeta_scores,
    best_pr_idx,
    cm,
    dataset_name,
    fbeta=0.5,
    save_prefix=None,
    fpr_roc=None,
    tpr_roc=None,
    roc_thresholds=None,
):
    """Render five Isolation Forest diagnostic plots and save as PNGs.

    Args:
        val_scores     : (N_val,) anomaly scores on validation normals.
        test_scores    : (N_test,) anomaly scores on the test set.
        y_true_arr     : (N_test,) true binary labels (0/1).
        threshold      : Calibrated anomaly threshold.
        pr_recalls     : Recall array from precision_recall_curve (validation).
        pr_precisions  : Precision array from precision_recall_curve (validation).
        fbeta_scores   : F-beta scores at each PR threshold.
        best_pr_idx    : Index of the selected threshold in PR arrays.
        cm             : 2x2 confusion matrix.
        dataset_name   : Shown in plot titles ("BGL" or "HDFS").
        fbeta          : Beta used in F-beta label text.
        save_prefix    : PNG filename prefix (defaults to dataset_name.lower()).
        fpr_roc        : FPR array from roc_curve.
        tpr_roc        : TPR array from roc_curve.
        roc_thresholds : Threshold array from roc_curve.
    """
    prefix = save_prefix or dataset_name.lower()

    # 10a — Validation score distribution with threshold
    # Shows the IF anomaly-score distribution on held-out normal logs and
    # where the F-beta calibrated threshold falls — analogous to the
    # training-loss curve in Deep SVDD (diagnostic view of model output).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(val_scores, bins=40, alpha=0.75, color="steelblue",
            label=f"Val normals (n={len(val_scores)})", edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--", lw=2,
               label=f"Calibrated threshold = {threshold:.4f}")
    pct_val = np.percentile(val_scores, PSEUDO_ANOMALY_PERCENTILE)
    ax.axvline(pct_val, color="darkorange", linestyle=":", lw=1.5,
               label=f"Top {100 - PSEUDO_ANOMALY_PERCENTILE}% percentile = {pct_val:.4f}")
    ax.set_xlabel("Anomaly Score  (negated IF score_samples)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Isolation Forest: Validation Score Distribution -- {dataset_name}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_if_val_score_dist.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

    # 10b — Test score distributions: Normal vs Anomalous
    normal_mask    = y_true_arr == 0
    anomalous_mask = y_true_arr == 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(test_scores[normal_mask], bins=40, alpha=0.6, color="steelblue",
            label=f"Normal (n={normal_mask.sum()})", edgecolor="white")
    ax.hist(test_scores[anomalous_mask], bins=40, alpha=0.6, color="crimson",
            label=f"Anomalous (n={anomalous_mask.sum()})", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score  (negated IF score_samples)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Isolation Forest: Test Score Distribution -- {dataset_name}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_if_score_dist.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

    # 10c — Precision-Recall curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pr_recalls, pr_precisions, color="darkorange", lw=2, label="PR Curve")
    ax.scatter(
        [pr_recalls[best_pr_idx]], [pr_precisions[best_pr_idx]],
        color="black", zorder=5, s=80,
        label=f"Selected (F{fbeta}={fbeta_scores[best_pr_idx]:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        f"Precision-Recall Curve (Validation) -- {dataset_name}",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_if_pr_curve.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

    # 10d — Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Anomalous"],
        yticklabels=["Actual Normal",    "Actual Anomalous"],
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"}, ax=ax,
    )
    for (r, c), lbl in {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}.items():
        ax.text(c + 0.5, r + 0.72, lbl, ha="center", va="center",
                fontsize=10, color="grey")
    ax.set_title(
        f"Confusion Matrix -- Isolation Forest [{dataset_name}]",
        fontweight="bold", pad=14,
    )
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fname = f"{prefix}_if_confusion_matrix.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

    # 10e — ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_roc, tpr_roc, color="darkorchid", lw=2,
            label=f"ROC Curve (AUC = {metrics[\'auroc\']:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1,
            label="Random classifier (AUC = 0.50)")
    op_idx = int(np.argmin(np.abs(roc_thresholds - threshold)))
    ax.scatter([fpr_roc[op_idx]], [tpr_roc[op_idx]],
               color="black", zorder=5, s=80,
               label=f"Operating point (threshold={threshold:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(
        f"ROC Curve -- Isolation Forest [{dataset_name}]  |  AUC = {metrics[\'auroc\']:.4f}",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{prefix}_if_roc_curve.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


# ── Execute ───────────────────────────────────────────────────────────────────

print("Generating visualisations ...")
plot_results(
    val_scores=val_scores,
    test_scores=test_scores,
    y_true_arr=np.array(y_true),
    threshold=if_threshold,
    pr_recalls=pr_recalls,
    pr_precisions=pr_precisions,
    fbeta_scores=fbeta_scores,
    best_pr_idx=best_idx,
    cm=metrics["cm"],
    dataset_name=DATASET,
    fbeta=FBETA,
    fpr_roc=fpr_roc,
    tpr_roc=tpr_roc,
    roc_thresholds=roc_thresholds,
)
'''
))

# ─────────────────────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

out = "E01_B_BGL_IsolationForest_Pipeline.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {out}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
import json, pathlib
src = ''.join(json.loads(pathlib.Path(out).read_text(encoding='utf-8'))['cells'][2]['source'])

checks = [
    ('from sklearn.ensemble import IsolationForest',   'IsolationForest imported'),
    ('import torch',                                   'torch absent'),  # should be MISSING
    ('IF_N_ESTIMATORS = 200',                          'IF hyperparams present'),
    ('def compute_if_scores(',                         'compute_if_scores defined'),
    ('if_model = IsolationForest(',                    'if_model instantiated'),
    ('if_model.fit(train_embeddings)',                 'model fitted'),
    ('val_scores = compute_if_scores(',                'val scores computed'),
    ('if_threshold',                                   'if_threshold used'),
    ('test_scores = compute_if_scores(',               'test scores computed'),
    ('def evaluate_predictions(',                      'evaluate_predictions defined'),
    ('model_name="Isolation Forest"',                  'model name = IF'),
    ('auroc = roc_auc_score(',                         'AUC-ROC computed'),
    ('def plot_results(',                              'plot_results defined'),
    ('val_scores,',                                    'val_scores in plot_results'),
    ('if_val_score_dist.png',                          'plot 10a (val dist) present'),
    ('if_score_dist.png',                              'plot 10b (test dist) present'),
    ('if_pr_curve.png',                                'plot 10c (PR) present'),
    ('if_confusion_matrix.png',                        'plot 10d (CM) present'),
    ('if_roc_curve.png',                               'plot 10e (ROC) present'),
    ('def _apply_sampling(',                           '_apply_sampling present'),
    ('def load_bgl_data(',                             'load_bgl_data present'),
    ('def load_hdfs_data(',                            'load_hdfs_data present'),
    ('def create_splits(',                             'create_splits present'),
]

print()
missing = False
for needle, label in checks:
    if needle == 'import torch':
        ok = needle not in src   # we WANT this to be absent
        status = 'OK' if ok else 'FAIL (torch still present!)'
    else:
        ok = needle in src
        status = 'OK' if ok else 'MISSING'
    print(f"  [{status}]  {label}")
    if not ok:
        missing = True

print()
print("All checks passed." if not missing else "WARNING: some checks failed.")
