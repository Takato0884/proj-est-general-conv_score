"""
Evaluation metrics for conversation score prediction.

Functions:
  ccc(y_true, y_pred)        -> float
  mae(y_true, y_pred)        -> float
  pearson_r(y_true, y_pred)  -> (r, p_value)
  compute_metrics(df)        -> dict  (global metrics)
  print_all_metrics(results) -> None
"""

import numpy as np
from scipy import stats


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mu_a = y_true.mean()
    mu_p = y_pred.mean()
    sigma_a = y_true.std()
    sigma_p = y_pred.std()
    r = np.corrcoef(y_true, y_pred)[0, 1]
    denom = sigma_a**2 + sigma_p**2 + (mu_a - mu_p)**2
    if denom == 0.0:
        return 0.0
    return float(2.0 * r * sigma_a * sigma_p / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns (r, p_value). Returns (nan, nan) if constant input."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


def compute_metrics(df) -> dict:
    """
    Compute global metrics.

    Parameters
    ----------
    df : DataFrame with columns y_true, y_pred

    Returns
    -------
    dict with keys: mae, pearson_r, pearson_p, ccc
    """
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    g_r, g_p = pearson_r(y_true, y_pred)
    return {
        "mae":       mae(y_true, y_pred),
        "pearson_r": g_r,
        "pearson_p": g_p,
        "ccc":       ccc(y_true, y_pred),
    }


def print_all_metrics(results: dict) -> None:
    """
    Print global metrics for all modalities.

    Parameters
    ----------
    results : dict  { modality -> {"df": DataFrame} }
              modality in {"text_only", "audio_only", "late_fusion"}
    """
    for modality, info in results.items():
        df = info["df"]
        m  = compute_metrics(df)

        print(f"\n{'='*55}")
        print(f"  {modality}")
        print(f"{'='*55}")
        print(f"  {'Metric':<20} {'global':>12}")
        print(f"  {'-'*32}")
        print(f"  {'MAE':<20} {m['mae']:>12.4f}")
        print(f"  {'Pearson r':<20} {m['pearson_r']:>12.4f}  (p={m['pearson_p']:.3e})")
        print(f"  {'CCC':<20} {m['ccc']:>12.4f}")
