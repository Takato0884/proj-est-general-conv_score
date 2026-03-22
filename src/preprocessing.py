"""
Generalized preprocessing pipeline for conversation score prediction.

Subcommands:
  feature_extraction  Extract text/audio embeddings and save as .pt files.

--- feature_extraction ---

Reads a pairs CSV (--pairs-csv) with the following columns:

  Required:
    speaker_id      : identifier for the focal speaker
    partner_id      : identifier for the conversation partner
    score           : numeric score to predict

  Path columns (resolved in order: explicit column → base-dir + fallback name):
    transcript_path : path to transcript CSV
                      fallback: {transcript_dir}/{speaker_id}__{partner_id}.csv
    wav_path        : path to speaker WAV file
                      fallback: {audio_dir}/{speaker_id}__{partner_id}.wav

  Optional:
    session_ID      : defaults to "{speaker_id}__{partner_id}"
    speaker_label   : value in transcript `speaker` column that identifies the
                      focal speaker's turns; defaults to speaker_id

Transcript CSV columns (required): start, end, speaker, text

Processing steps per sample:
  0. Merge consecutive same-speaker utterances
  1. Extract audio segment (start/end timestamps)
  2. VAD trimming (webrtcvad, 30ms frames, 300ms padding)
  3. Text embedding: sentence-transformers/sentence-t5-large -> [n_utts, 768] L2-normalized
  4. Audio embedding: facebook/hubert-large-ll60k -> [n_utts, 1024] L2-normalized
  5. Save .pt file

Output: {output_dir}/{session_ID}.pt
  Fields: session_ID, speaker_id, partner_id, speaker,
          first_speaker, text_embeddings, audio_embeddings, score
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import webrtcvad
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_SR          = 16000
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS       = 30
VAD_PADDING_MS     = 300


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_wav_mono_16k(wav_path: Path) -> torch.Tensor:
    """Load WAV, convert to mono float32 at 16 kHz. Returns 1-D tensor."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)  # [T]


def slice_audio(wav: torch.Tensor, start: float, end: float) -> torch.Tensor:
    s = int(start * TARGET_SR)
    e = min(int(end * TARGET_SR), wav.shape[0])
    return wav[s:e]


def vad_trim(wav: torch.Tensor) -> torch.Tensor:
    """
    Remove silence with webrtcvad (30 ms frames, 300 ms padding).
    Falls back to original waveform if no speech is detected.
    """
    if wav.numel() == 0:
        return wav

    frame_samples = TARGET_SR * VAD_FRAME_MS // 1000  # 480 at 16 kHz / 30 ms
    pad_frames    = VAD_PADDING_MS // VAD_FRAME_MS     # 10 frames

    # webrtcvad requires int16 PCM bytes
    wav_int16  = (wav.float() * 32767).clamp(-32768, 32767).short()
    raw_bytes  = wav_int16.numpy().tobytes()
    frame_bytes = frame_samples * 2  # 2 bytes per int16 sample

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # Build per-frame speech mask
    speech_mask = []
    for i in range(0, len(raw_bytes), frame_bytes):
        frame = raw_bytes[i: i + frame_bytes]
        if len(frame) < frame_bytes:
            frame = frame + b'\x00' * (frame_bytes - len(frame))
        speech_mask.append(vad.is_speech(frame, TARGET_SR))

    # Expand mask with ±pad_frames padding
    padded = list(speech_mask)
    for i, v in enumerate(speech_mask):
        if v:
            lo = max(0, i - pad_frames)
            hi = min(len(padded), i + pad_frames + 1)
            for j in range(lo, hi):
                padded[j] = True

    # Collect voiced frames
    voiced_chunks = []
    for i, keep in enumerate(padded):
        if keep:
            s = i * frame_samples
            voiced_chunks.append(wav[s: s + frame_samples])

    return torch.cat(voiced_chunks) if voiced_chunks else wav  # fallback: keep original


# ── Transcript helpers ─────────────────────────────────────────────────────────

def merge_consecutive(df_all: pd.DataFrame, speaker_label: str) -> pd.DataFrame:
    """
    STEP 0: Sort utterances by start time, then merge adjacent same-speaker
    runs (i.e. when no other speaker's turn intervenes).
    """
    df_sorted = df_all.sort_values("start").reset_index(drop=True)

    result = []
    cur = None
    last_was_target = False

    for _, row in df_sorted.iterrows():
        if row["speaker"] == speaker_label:
            if last_was_target and cur is not None:
                cur["end"]  = row["end"]
                cur["text"] = cur["text"] + " " + str(row["text"])
            else:
                if cur is not None:
                    result.append(cur)
                cur = {"start": row["start"], "end": row["end"], "text": str(row["text"])}
            last_was_target = True
        else:
            last_was_target = False

    if cur is not None:
        result.append(cur)

    return pd.DataFrame(result) if result else pd.DataFrame(columns=["start", "end", "text"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")


def _resolve_path(
    row: pd.Series,
    col: str,
    base_dir: Path | None,
    fallback_name: str,
) -> Path | None:
    """Return the path from a CSV column, falling back to base_dir/fallback_name."""
    if col in row.index and pd.notna(row[col]):
        p = Path(str(row[col]))
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        return p
    if base_dir is not None:
        return base_dir / fallback_name
    return None


# ── Subcommand: feature_extraction ────────────────────────────────────────────

def cmd_feature_extraction(args):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    from sentence_transformers import SentenceTransformer

    output_dir     = Path(args.output_dir)
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None
    audio_dir      = Path(args.audio_dir)      if args.audio_dir      else None

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.read_csv(args.pairs_csv)
    _check_columns(pairs_df, ["speaker_id", "partner_id", "score"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading text model (sentence-t5-large)...")
    text_model = SentenceTransformer(
        "sentence-transformers/sentence-t5-large", device=str(device)
    )
    for p in text_model.parameters():
        p.requires_grad_(False)

    print("Loading audio model (hubert-large-ll60k)...")
    audio_feat_ext = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
    audio_model    = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)
    for p in audio_model.parameters():
        p.requires_grad_(False)
    audio_model.eval()

    print(f"Processing {len(pairs_df)} rows from {args.pairs_csv}\n")

    for _, row in pairs_df.iterrows():
        speaker_id    = str(row["speaker_id"])
        partner_id    = str(row["partner_id"])
        score         = float(row["score"])
        session_ID    = str(row["session_ID"])    if "session_ID"    in row.index and pd.notna(row.get("session_ID"))    else f"{speaker_id}__{partner_id}"
        speaker_label = str(row["speaker_label"]) if "speaker_label" in row.index and pd.notna(row.get("speaker_label")) else speaker_id

        out_path = output_dir / f"{session_ID}.pt"
        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue

        # Resolve paths
        transcript_path = _resolve_path(
            row, "transcript_path", transcript_dir,
            f"{speaker_id}__{partner_id}.csv",
        )
        wav_path = _resolve_path(
            row, "wav_path", audio_dir,
            f"{speaker_id}__{partner_id}.wav",
        )

        if transcript_path is None or not transcript_path.exists():
            print(f"[WARN] Transcript not found for {session_ID}: {transcript_path}")
            continue
        if wav_path is None or not wav_path.exists():
            print(f"[WARN] WAV not found for {session_ID}: {wav_path}")
            continue

        df = pd.read_csv(transcript_path)
        if "text" not in df.columns:
            print(f"[WARN] 'text' column missing in {transcript_path.name}, skipping")
            continue

        first_speaker = str(df.sort_values("start").iloc[0]["speaker"])

        # STEP 0: merge consecutive same-speaker utterances
        utts = merge_consecutive(df, speaker_label)
        if utts.empty:
            print(f"[WARN] No utterances for speaker '{speaker_label}' in {transcript_path.name}")
            continue

        # Load full WAV once
        wav = load_wav_mono_16k(wav_path)

        texts          = []
        audio_segments = []

        for _, utt in utts.iterrows():
            texts.append(utt["text"])

            # STEP 1: extract audio segment
            seg = slice_audio(wav, utt["start"], utt["end"])
            if seg.numel() == 0:
                seg = torch.zeros(TARGET_SR // 10)  # 100 ms silence fallback

            # STEP 2: VAD trimming
            audio_segments.append(vad_trim(seg))

        # STEP 3: text embeddings [n_utts, 768]
        with torch.no_grad():
            text_embs = text_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False,
            )
            text_embs = F.normalize(text_embs.to(device), p=2, dim=-1)

        # STEP 4: audio embeddings [n_utts, 1024]
        audio_embs = []
        with torch.no_grad():
            for seg in audio_segments:
                inputs = audio_feat_ext(
                    seg.numpy(), sampling_rate=TARGET_SR,
                    return_tensors="pt", padding=True,
                )
                outputs = audio_model(inputs["input_values"].to(device))
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                emb = F.normalize(emb, p=2, dim=-1)
                audio_embs.append(emb.cpu())

        # STEP 5: save
        payload = {
            "session_ID":       session_ID,
            "speaker_id":       speaker_id,
            "partner_id":       partner_id,
            "speaker":          speaker_label,
            "first_speaker":    first_speaker,
            "text_embeddings":  text_embs.cpu(),         # Tensor[n_utts, 768]
            "audio_embeddings": torch.stack(audio_embs), # Tensor[n_utts, 1024]
            "score":            score,
        }
        torch.save(payload, out_path)
        print(f"[SAVED] {out_path.name}  ({len(texts)} utterances)")

    print("\nDone.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing tools for conversation score prediction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- feature_extraction ---
    fe = subparsers.add_parser(
        "feature_extraction",
        help="Extract text/audio embeddings from transcripts+audio and save as .pt files.",
    )
    fe.add_argument(
        "--pairs-csv", required=True,
        help="CSV with speaker_id, partner_id, score, and optional columns "
             "(session_ID, speaker_label, transcript_path, wav_path).",
    )
    fe.add_argument(
        "--output-dir", required=True,
        help="Directory to write .pt embedding files.",
    )
    fe.add_argument(
        "--transcript-dir", default=None,
        help="Base directory for transcript CSV files "
             "(used when transcript_path column is absent or relative).",
    )
    fe.add_argument(
        "--audio-dir", default=None,
        help="Base directory for WAV files "
             "(used when wav_path column is absent or relative).",
    )

    args = parser.parse_args()

    if args.command == "feature_extraction":
        cmd_feature_extraction(args)


if __name__ == "__main__":
    main()
