# modeling/evaluation_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Tuple

def ranked_precision_at_cutoffs(y_true: np.ndarray, y_proba: np.ndarray,
                                cutoffs: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5)) -> pd.DataFrame:
    """
    Compute cumulative precision at top p% ranked by predicted probability (descending).
    cutoffs are fractions in (0,1], e.g. 0.1 = top 10%.
    """
    assert len(y_true) == len(y_proba)
    order = np.argsort(-y_proba)  # descending
    y_true_sorted = np.asarray(y_true)[order]

    rows = []
    n = len(y_true_sorted)
    for p in cutoffs:
        k = max(1, int(np.ceil(p * n)))
        y_top = y_true_sorted[:k]
        prec = y_top.mean() if k > 0 else 0.0
        rows.append({"cutoff_pct": int(p * 100), "k": k, "precision": float(prec)})
    return pd.DataFrame(rows)

def ranked_precision_by_bins(y_true: np.ndarray, y_proba: np.ndarray,
                             bins: Sequence[Tuple[float, float]] = ((0.0, 0.1),
                                                                    (0.1, 0.2),
                                                                    (0.2, 0.3),
                                                                    (0.3, 0.4),
                                                                    (0.4, 0.5),
                                                                    (0.5, 0.6),
                                                                    (0.6, 0.7),
                                                                    (0.7, 0.8),
                                                                    (0.8, 0.9),
                                                                    (0.9, 1.0))) -> pd.DataFrame:
    """
    Precision within probability bins (non-cumulative).
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    rows = []
    for lo, hi in bins:
        m = (df["p"] >= lo) & (df["p"] < hi) if hi < 1.0 else (df["p"] >= lo) & (df["p"] <= hi)
        sub = df[m]
        prec = sub["y"].mean() if len(sub) else np.nan
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(len(sub)), "precision": (float(prec) if not np.isnan(prec) else None)})
    return pd.DataFrame(rows)

def plot_cumulative_precision(curve_df: pd.DataFrame, title: str = "Cumulative Precision by Top %"):
    """
    curve_df expected from ranked_precision_at_cutoffs()
    """
    plt.figure(figsize=(6, 4))
    plt.plot(curve_df["cutoff_pct"], curve_df["precision"], marker="o")
    plt.xlabel("Top % of ranked predictions")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bin_precision(bin_df: pd.DataFrame, title: str = "Precision by Probability Bin"):
    """
    bar plot of precision by probability bins
    """
    centers = 100 * (bin_df["bin_lo"] + bin_df["bin_hi"]) / 2.0
    plt.figure(figsize=(7, 4))
    plt.bar(centers, bin_df["precision"], width=10)
    plt.xlabel("Predicted probability bin center (%)")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_predictions_csv(model, X, ids_df: pd.DataFrame, y_true: np.ndarray = None,
                         path: str = "predictions.csv"):
    """
    Save predictions with user_id & request_id for auditability.
    Assumes ids_df has ['user_id','request_id'] aligned to X rows.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = ids_df.copy()
    out["y_pred"] = pred
    out["y_pred_proba"] = proba
    if y_true is not None and len(y_true) == len(out):
        out["y_true"] = y_true
    out.to_csv(path, index=False)
    print(f"Saved {len(out):,} rows to {path}")
    return out
