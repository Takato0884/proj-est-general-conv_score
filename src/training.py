"""
Model architecture and training loop for conversation score prediction.

Model
-----
  AdditiveAttentionPooling  : [B, T, d] -> [B, d]
  PredictionHead            : [B, d] -> [B, 1]
  TextModel                 : text stream  (d=768)
  AudioModel                : audio stream (d=1024)

Training
--------
  make_participant_batches  : group samples by pair_key, yield batches
  ccc_loss                  : 1 - mean_pair_CCC (differentiable)
  train_one_epoch           : one pass over participant batches
  evaluate_val              : compute val CCC (no grad)
  fit                       : full training with Early Stopping
"""

import random
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ── Model ──────────────────────────────────────────────────────────────────────

class AdditiveAttentionPooling(nn.Module):
    """
    Additive attention over an utterance sequence.

    s_i = w^T * e_i + b
    α   = softmax(s, masked)
    v   = Σ α_i * e_i
    """
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, T, d]
        mask : [B, T]  bool, True = valid position
        returns v : [B, d]
        """
        scores = self.w(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha  = F.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)


class PredictionHead(nn.Module):
    """Linear(d → d//5) → ReLU → Dropout → Linear(d//5 → 1)"""
    def __init__(self, d: int, dropout: float = 0.2):
        super().__init__()
        hidden = d // 5
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d]  ->  [B]"""
        return self.net(x).squeeze(-1)


class TextModel(nn.Module):
    """Text stream (d=768): additive-attention pool → prediction head."""
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.attn = AdditiveAttentionPooling(768)
        self.head = PredictionHead(768, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(x, mask))


class AudioModel(nn.Module):
    """Audio stream (d=1024): additive-attention pool → prediction head."""
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.attn = AdditiveAttentionPooling(1024)
        self.head = PredictionHead(1024, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(x, mask))


# ── Differentiable CCC ─────────────────────────────────────────────────────────

def _ccc_tensor(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """CCC for 1-D tensors (at least 2 elements)."""
    mu_a  = y_true.mean()
    mu_p  = y_pred.mean()
    var_a = ((y_true - mu_a) ** 2).mean()
    var_p = ((y_pred - mu_p) ** 2).mean()
    cov   = ((y_true - mu_a) * (y_pred - mu_p)).mean()
    denom = var_a + var_p + (mu_a - mu_p) ** 2
    return 2.0 * cov / (denom + 1e-8)


def _ccc_loss_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    participant_ids: list,
) -> torch.Tensor:
    unique_ids = list(dict.fromkeys(participant_ids))
    cccs = []
    for pid in unique_ids:
        idx = [i for i, p in enumerate(participant_ids) if p == pid]
        if len(idx) < 2:
            continue
        cccs.append(_ccc_tensor(targets[idx], scores[idx]))
    if not cccs:
        return F.mse_loss(scores, targets)
    return 1.0 - torch.stack(cccs).mean()


def _mse_loss_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    participant_ids: list,
) -> torch.Tensor:
    unique_ids = list(dict.fromkeys(participant_ids))
    mses = []
    for pid in unique_ids:
        idx = [i for i, p in enumerate(participant_ids) if p == pid]
        mses.append(F.mse_loss(scores[idx], targets[idx]))
    return torch.stack(mses).mean()


# ── Batching helpers ───────────────────────────────────────────────────────────

def make_participant_batches(samples: list, batch_size: int = 6, seed: int = None):
    grouped: dict[str, list] = defaultdict(list)
    for s in samples:
        grouped[s["session_ID"]].append(s)
    participants = list(grouped.keys())
    random.Random(seed).shuffle(participants)
    for start in range(0, len(participants), batch_size):
        batch = []
        for pid in participants[start: start + batch_size]:
            batch.extend(grouped[pid])
        yield batch


def _collate_batch(batch: list, device: torch.device):
    """Pack a list of sample dicts into padded tensors."""
    text_seqs  = [s["text_embeddings"]  for s in batch]
    audio_seqs = [s["audio_embeddings"] for s in batch]

    text_padded  = pad_sequence(text_seqs,  batch_first=True).to(device).float()
    audio_padded = pad_sequence(audio_seqs, batch_first=True).to(device).float()

    t_lens = torch.tensor([t.shape[0] for t in text_seqs],  device=device)
    a_lens = torch.tensor([t.shape[0] for t in audio_seqs], device=device)
    T_t, T_a = text_padded.shape[1], audio_padded.shape[1]
    text_mask  = torch.arange(T_t, device=device).unsqueeze(0) < t_lens.unsqueeze(1)
    audio_mask = torch.arange(T_a, device=device).unsqueeze(0) < a_lens.unsqueeze(1)

    targets  = torch.tensor([s["_target_scaled"] for s in batch],
                             dtype=torch.float32, device=device)
    part_ids = [s["session_ID"] for s in batch]

    return text_padded, text_mask, audio_padded, audio_mask, targets, part_ids


# ── Training / evaluation ──────────────────────────────────────────────────────

def _make_sample_batches(samples: list, batch_size: int, seed: int = None):
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[i] for i in indices[start: start + batch_size]]


def train_one_epoch(
    model,
    modality: str,
    train_samples: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: str,
    batch_size: int = 6,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    epoch_seed = random.randint(0, 2**31)

    if loss_fn == "mse":
        batches = _make_sample_batches(train_samples, batch_size, seed=epoch_seed)
    else:
        batches = make_participant_batches(train_samples, batch_size, seed=epoch_seed)

    for batch in batches:
        text_x, text_m, audio_x, audio_m, targets, part_ids = _collate_batch(batch, device)
        optimizer.zero_grad()
        scores = model(text_x, text_m) if modality == "text" else model(audio_x, audio_m)
        loss = (_ccc_loss_from_scores(scores, targets, part_ids) if loss_fn == "ccc"
                else F.mse_loss(scores, targets))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_val(
    model,
    modality: str,
    val_samples: list,
    device: torch.device,
) -> float:
    """Compute validation CCC (per-pair average, no grad)."""
    model.eval()
    all_pair_ids = []
    all_trues    = []
    all_preds    = []

    with torch.no_grad():
        for s in val_samples:
            text_x  = s["text_embeddings"].unsqueeze(0).to(device).float()
            audio_x = s["audio_embeddings"].unsqueeze(0).to(device).float()
            text_m  = torch.ones(1, text_x.shape[1],  dtype=torch.bool, device=device)
            audio_m = torch.ones(1, audio_x.shape[1], dtype=torch.bool, device=device)
            score = (model(text_x, text_m) if modality == "text"
                     else model(audio_x, audio_m))
            all_preds.append(score.item())
            all_trues.append(s["_target_scaled"])
            all_pair_ids.append(s["session_ID"])

    grouped: dict[str, tuple] = defaultdict(lambda: ([], []))
    for pid, yt, yp in zip(all_pair_ids, all_trues, all_preds):
        grouped[pid][0].append(yt)
        grouped[pid][1].append(yp)

    cccs = []
    for pid, (trues, preds) in grouped.items():
        if len(trues) < 2:
            continue
        yt = np.array(trues)
        yp = np.array(preds)
        mu_a, mu_p   = yt.mean(), yp.mean()
        var_a, var_p = yt.var(), yp.var()
        cov   = np.mean((yt - mu_a) * (yp - mu_p))
        denom = var_a + var_p + (mu_a - mu_p) ** 2
        cccs.append(2.0 * cov / (denom + 1e-8) if denom > 0 else 0.0)

    if cccs:
        return float(np.mean(cccs))

    # Fallback: global CCC over all samples (used in LOPO where each pair has 1 sample)
    yt = np.array(all_trues)
    yp = np.array(all_preds)
    if len(yt) < 2:
        return 0.0
    mu_a, mu_p   = yt.mean(), yp.mean()
    var_a, var_p = yt.var(), yp.var()
    cov   = np.mean((yt - mu_a) * (yp - mu_p))
    denom = var_a + var_p + (mu_a - mu_p) ** 2
    return float(2.0 * cov / (denom + 1e-8)) if denom > 0 else 0.0


def fit(
    model,
    modality: str,
    train_samples: list,
    val_samples: list,
    device: torch.device,
    loss_fn: str,
    lr: float = 0.001,
    patience: int = 20,
    batch_size: int = 6,
    max_epochs: int = 500,
    min_epochs: int = 20,
    verbose: bool = True,
):
    """Train with Early Stopping based on validation CCC."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_ccc   = -float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, modality, train_samples, optimizer,
                                     device, loss_fn, batch_size)
        val_ccc    = evaluate_val(model, modality, val_samples, device)

        if epoch >= min_epochs:
            if val_ccc > best_val_ccc:
                best_val_ccc   = val_ccc
                best_state     = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

        if verbose:
            flag = " *" if patience_count == 0 else ""
            print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_ccc={val_ccc:.4f}{flag}")

        if epoch >= min_epochs and patience_count >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}  "
                      f"(best val CCC={best_val_ccc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
