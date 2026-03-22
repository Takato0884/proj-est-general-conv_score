"""
main.py  –  Conversation score prediction  (Leave-One-Person-Out CV)

Usage
-----
  python main.py --target <score_key> [options]

  python main.py --target score
  python main.py --target score --verbose

Run `python main.py --help` for full option list.
"""

import argparse
import re
import csv
import sys
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Default CONFIG values ─────────────────────────────────────────────────────

_DEFAULT_TARGET     = "score"
_DEFAULT_EMB_DIR    = "./data/preprocessed"
_DEFAULT_OUT_DIR    = "./log/inference"
_DEFAULT_LR          = 0.001
_DEFAULT_PATIENCE    = 20
_DEFAULT_BATCH_SIZE  = 6
_DEFAULT_MAX_EPOCHS  = 500

# ── Filename regex ─────────────────────────────────────────────────────────────
# session_XX.pt  e.g. session_01.pt
_PT_RE = re.compile(r"^(session_\d+)\.pt$")

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_emb_index(emb_dir: Path) -> dict:
    """
    Scan emb_dir and return a dict:
        session_id -> Path of the .pt file
    """
    index: dict[str, Path] = {}
    for pt_path in emb_dir.glob("*.pt"):
        m = _PT_RE.match(pt_path.name)
        if m is None:
            continue
        session_id = m.group(1)
        index[session_id] = pt_path
    return index


def load_sample(pt_path: Path, target: str) -> dict:
    """
    Load a single .pt file and return a sample dict:
        session_ID       : str
        text_embeddings  : Tensor[n_utts, 768]
        audio_embeddings : Tensor[n_utts, 1024]
        _target_orig     : float (original scale)
        _target_scaled   : float (z-scored, filled later by apply_scaler)
        _speaker_id      : str
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return {
        "session_ID":       data["session_ID"],
        "text_embeddings":  data["text_embeddings"],
        "audio_embeddings": data["audio_embeddings"],
        "_target_orig":     float(data[target]),
        "_target_scaled":   0.0,
        "_speaker_id":      data["speaker_id"],
    }


def fit_scaler(train_samples: list) -> tuple[float, float]:
    """Compute mean and std of target from training samples only."""
    vals = np.array([s["_target_orig"] for s in train_samples], dtype=np.float64)
    mu  = float(vals.mean())
    sig = float(vals.std())
    if sig == 0.0:
        sig = 1.0
    return mu, sig


def apply_scaler(samples: list, mu: float, sig: float) -> None:
    """In-place z-score normalization of _target_scaled."""
    for s in samples:
        s["_target_scaled"] = (s["_target_orig"] - mu) / sig


def append_csv(out_path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV file, writing header only if the file is new."""
    fieldnames = ["session_ID", "y_true", "y_pred"]
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _save_scatter_plots(eval_results: dict, out_dir: Path) -> None:
    """Save a scatter plot (y_true vs y_pred) for each modality."""
    modalities = [m for m in ("text_only", "audio_only", "late_fusion")
                  if m in eval_results]
    if not modalities:
        return

    fig, axes = plt.subplots(1, len(modalities),
                             figsize=(5 * len(modalities), 5),
                             squeeze=False)

    for ax, modality in zip(axes[0], modalities):
        df = eval_results[modality]["df"]
        yt = df["y_true"].values
        yp = df["y_pred"].values
        ax.scatter(yt, yp, alpha=0.5, s=20)
        lim_min = min(yt.min(), yp.min()) - 0.2
        lim_max = max(yt.max(), yp.max()) + 0.2
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(modality.replace("_", " "))

    fig.tight_layout()
    save_path = out_dir / "scatter_predictions.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nScatter plot saved to: {save_path}")


def _save_result_txt(eval_results: dict, out_path: Path) -> None:
    """Save final evaluation results to result.txt."""
    from evaluation_metrics import compute_metrics

    lines = []
    lines.append("=" * 60)
    lines.append("FINAL EVALUATION  (all folds combined)")
    lines.append("=" * 60)

    for modality, info in eval_results.items():
        df = info["df"]
        m  = compute_metrics(df)

        n_pairs   = df["session_ID"].nunique()
        n_samples = len(df)

        lines.append(f"\n{'='*55}")
        lines.append(f"  {modality}")
        lines.append(f"{'='*55}")
        lines.append(f"  Pairs   : {n_pairs}")
        lines.append(f"  Samples : {n_samples}")
        lines.append(f"  {'Metric':<20} {'global':>12}")
        lines.append(f"  {'-'*32}")
        lines.append(f"  {'MAE':<20} {m['mae']:>12.4f}")
        lines.append(
            f"  {'Pearson r':<20} {m['pearson_r']:>12.4f}  (p={m['pearson_p']:.3e})"
        )
        lines.append(f"  {'CCC':<20} {m['ccc']:>12.4f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nResult saved to: {out_path}")


# ── Main CV loop ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Conversation score prediction – Leave-One-Person-Out CV"
    )
    p.add_argument("--target",     type=str,   default=_DEFAULT_TARGET,
                   help=f"Prediction target key in .pt files (default: {_DEFAULT_TARGET})")
    p.add_argument("--emb-dir",    type=Path,  default=Path(_DEFAULT_EMB_DIR),
                   help=f"Directory containing .pt embedding files (default: {_DEFAULT_EMB_DIR})")
    p.add_argument("--out-dir",    type=Path,  default=Path(_DEFAULT_OUT_DIR),
                   help=f"Output directory for result CSVs (default: {_DEFAULT_OUT_DIR})")
    p.add_argument("--lr",         type=float, default=_DEFAULT_LR,
                   help=f"Learning rate (default: {_DEFAULT_LR})")
    p.add_argument("--patience",   type=int,   default=_DEFAULT_PATIENCE,
                   help=f"Early stopping patience in epochs (default: {_DEFAULT_PATIENCE})")
    p.add_argument("--batch-size", type=int,   default=_DEFAULT_BATCH_SIZE,
                   help=f"Number of pairs per batch (default: {_DEFAULT_BATCH_SIZE})")
    p.add_argument("--max-epochs", type=int,   default=_DEFAULT_MAX_EPOCHS,
                   help=f"Maximum training epochs (default: {_DEFAULT_MAX_EPOCHS})")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-epoch training logs")
    return p.parse_args()


def main():
    args = parse_args()

    TARGET     = args.target
    LOSS_FN    = "ccc"
    EMB_DIR    = args.emb_dir
    OUT_DIR    = args.out_dir
    LR         = args.lr
    PATIENCE   = args.patience
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    VERBOSE    = args.verbose

    sys.path.insert(0, str(Path(__file__).parent))
    from training import TextModel, AudioModel, fit as train_fit
    from inference import run_inference
    from evaluation_metrics import compute_metrics, print_all_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"TARGET    : {TARGET}")
    print(f"EMB_DIR   : {EMB_DIR}")
    print(f"OUT_DIR   : {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = [d for d in OUT_DIR.iterdir()
                if d.is_dir() and d.name.startswith("exp_") and d.name[4:].isdigit()]
    next_n = max((int(d.name[4:]) for d in existing), default=0) + 1
    EXP_DIR = OUT_DIR / f"exp_{next_n}"
    EXP_DIR.mkdir()
    print(f"EXP_DIR   : {EXP_DIR}")

    config_lines = [
        f"timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"target     : {TARGET}",
        f"loss_fn    : {LOSS_FN}",
        f"lr         : {LR}",
        f"patience   : {PATIENCE}",
        f"batch_size : {BATCH_SIZE}",
        f"max_epochs : {MAX_EPOCHS}",
        f"emb_dir    : {EMB_DIR}",
        f"cv_scheme  : leave-one-person-out",
    ]
    (EXP_DIR / "config.txt").write_text("\n".join(config_lines) + "\n")

    print("\nBuilding embedding index...")
    emb_index = build_emb_index(EMB_DIR)
    all_sessions = sorted(emb_index.keys())
    print(f"  Found {len(all_sessions)} sessions: {all_sessions}\n")

    # ── Leave-One-Person-Out loop ───────────────────────────────────────────────
    for fold_idx, test_session in enumerate(all_sessions, start=1):
        print(f"{'─'*60}")
        print(f"LOPO fold {fold_idx:3d} / {len(all_sessions)}  (test session: {test_session})")
        print(f"{'─'*60}")

        test_samples = [load_sample(emb_index[test_session], TARGET)]

        train_sessions = [s for s in all_sessions if s != test_session]
        train_samples  = [load_sample(emb_index[s], TARGET) for s in train_sessions]

        # Use training set as validation for early-stopping
        # (LOPO has only 1 test sample, so no separate val set)
        val_samples = train_samples

        print(f"  train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

        mu, sig = fit_scaler(train_samples)
        apply_scaler(train_samples, mu, sig)
        apply_scaler(test_samples,  mu, sig)
        # val_samples is the same list as train_samples, already scaled

        common_fit_kwargs = dict(
            train_samples=train_samples,
            val_samples=val_samples,
            device=device,
            loss_fn=LOSS_FN,
            lr=LR,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            verbose=VERBOSE,
        )

        print(f"  Training text model  (loss={LOSS_FN}, patience={PATIENCE})...")
        text_model  = train_fit(TextModel(),  "text",  **common_fit_kwargs)

        print(f"  Training audio model (loss={LOSS_FN}, patience={PATIENCE})...")
        audio_model = train_fit(AudioModel(), "audio", **common_fit_kwargs)

        result_rows = run_inference(
            text_model, audio_model, test_samples, device,
            scaler_mean=mu, scaler_std=sig,
        )

        for modality in ("text_only", "audio_only", "late_fusion"):
            append_csv(EXP_DIR / f"results_{modality}.csv",
                       result_rows[modality])

        print(f"  Written {len(test_samples)} test rows to each CSV.")

        if VERBOSE:
            for modality in ("text_only", "audio_only", "late_fusion"):
                rows = result_rows[modality]
                yt = rows[0]["y_true"]
                yp = rows[0]["y_pred"]
                print(f"    {modality:<15}: y_true={yt:.4f}  y_pred={yp:.4f}")

    # ── Final evaluation ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION  (all folds combined)")
    print(f"{'='*60}")

    import pandas as pd

    eval_results = {}
    for modality in ("text_only", "audio_only", "late_fusion"):
        csv_path = EXP_DIR / f"results_{modality}.csv"
        if not csv_path.exists():
            print(f"  [MISSING] {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        eval_results[modality] = {"df": df}

    if eval_results:
        print_all_metrics(eval_results)
        _save_scatter_plots(eval_results, EXP_DIR)
        _save_result_txt(eval_results, EXP_DIR / "result.txt")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
