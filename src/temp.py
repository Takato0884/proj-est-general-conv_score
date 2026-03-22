"""
temp.py  –  Temporary utility scripts

Subcommands:
  make_toy_dataset  Generate 30 toy .pt files simulating human–robot dialogue
                    for dialogue-level score estimation.
"""

import argparse
from pathlib import Path

import torch


# ── Subcommand: make_toy_dataset ───────────────────────────────────────────────

def cmd_make_toy_dataset(args):
    """
    Generate 30 toy .pt files simulating human–robot dialogue.

    Each file represents one speaker's (F001-F030) features from a single
    interaction with one robot partner (M000).  File: {spk_id}__M000.pt

    Feature dimensions
    ──────────────────
    text_embeddings  : Tensor[n_utts, 768]   L2-normalised per utterance
    audio_embeddings : Tensor[n_utts, 1024]  L2-normalised per utterance

    Signal design
    ─────────────
    The leading SIGNAL_DIMS (32) dimensions of both embedding types receive a
    positive bias of  true_score * SIGNAL_SCALE  before L2 normalisation.
    All remaining dimensions are i.i.d. Gaussian noise.
    After normalisation the pooled embedding in those dimensions correlates
    monotonically with true_score, giving the model a learnable signal.

    Target key: "score"  (float in [0, 1]) — matches main.py --target score.
    CV splits are not generated; use leave-one-person-out inside main.py.
    """
    import numpy as np

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    N_SPEAKERS    = 30
    TEXT_DIM      = 768
    AUDIO_DIM     = 1024
    SIGNAL_DIMS   = 32      # leading dims that carry the signal
    SIGNAL_SCALE  = 6.0     # bias magnitude before L2 normalisation
    PARTNER_ID    = "M000"  # robot represented as M000

    # Uniformly spread true scores so fold groups have good score variance.
    true_scores = rng.uniform(0.05, 0.95, size=N_SPEAKERS)
    rng.shuffle(true_scores)  # decouple speaker index from score ordering

    for i in range(N_SPEAKERS):
        spk_id     = f"F{i + 1:03d}"
        session_ID = f"session_{i + 1:02d}"
        true_score = float(true_scores[i])
        n_utts     = int(rng.integers(args.n_utts_min, args.n_utts_max + 1))

        # ── Text embeddings [n_utts, 768] ───────────────────────────────────
        text_raw = rng.standard_normal((n_utts, TEXT_DIM)).astype(np.float32)
        text_raw[:, :SIGNAL_DIMS] += true_score * SIGNAL_SCALE
        text_raw /= np.linalg.norm(text_raw, axis=1, keepdims=True)

        # ── Audio embeddings [n_utts, 1024] ─────────────────────────────────
        audio_raw = rng.standard_normal((n_utts, AUDIO_DIM)).astype(np.float32)
        audio_raw[:, :SIGNAL_DIMS] += true_score * SIGNAL_SCALE
        audio_raw /= np.linalg.norm(audio_raw, axis=1, keepdims=True)

        payload = {
            "session_ID":       session_ID,
            "speaker_id":       spk_id,
            "partner_id":       PARTNER_ID,
            "speaker":          "human",
            "first_speaker":    "human",
            "text_embeddings":  torch.from_numpy(text_raw),   # Tensor[n_utts, 768]
            "audio_embeddings": torch.from_numpy(audio_raw),  # Tensor[n_utts, 1024]
            "score":            true_score,                   # float [0, 1]
        }

        fname    = f"{session_ID}.pt"  # e.g. session_01.pt
        out_path = out_dir / fname
        torch.save(payload, out_path)

        print(f"[SAVED] {fname:<25}  n_utts={n_utts:2d}  score={true_score:.3f}")

    print(f"\nDone. {N_SPEAKERS} .pt files saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Temporary utility scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy_parser = subparsers.add_parser(
        "make_toy_dataset",
        help="Generate 30 toy .pt files for dialogue-level score estimation.",
    )
    toy_parser.add_argument(
        "--out-dir", default="./data/toy",
        help="Directory to save .pt files (default: ./data/toy)",
    )
    toy_parser.add_argument(
        "--n-utts-min", type=int, default=8,
        help="Minimum number of utterances per sample (default: 8)",
    )
    toy_parser.add_argument(
        "--n-utts-max", type=int, default=20,
        help="Maximum number of utterances per sample (default: 20)",
    )
    toy_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    if args.command == "make_toy_dataset":
        cmd_make_toy_dataset(args)


if __name__ == "__main__":
    main()
