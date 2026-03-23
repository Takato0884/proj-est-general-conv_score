# Conversation Score Prediction Pipeline

A regression pipeline that predicts the score a speaker assigns to their conversation partner, from arbitrary dialogue audio and transcript text.

> Japanese README: [README_ja.md](README_ja.md)

---

## Task Definition

**Input**: A transcript CSV for one session (utterance-level start/end times, speaker, text) and the corresponding audio WAV.

**Output**: Predicted score (continuous regression) that the target speaker assigns in the session.

**Features**:

| Stream | Model | Embedding dim |
|--------|-------|---------------|
| Text   | sentence-transformers/sentence-t5-large | 768 (L2-normalized) |
| Audio  | facebook/hubert-large-ll60k | 1024 (L2-normalized) |

For each stream, the utterance sequence is aggregated into a session embedding via Additive Attention Pooling, then an MLP head predicts the score. The final output also includes **Late Fusion** (simple average of text and audio predictions).

**Cross-validation**: Leave-One-Session-Out (LOSO) — each session is used as the test set once. The scaler is fit on the training set only and applied to the test set (metrics are computed after inverse-transforming predictions).

---

## Pipeline Overview

```
[Audio WAV / Transcript CSV / Score CSV]
        ↓  Step 2
[Feature Extraction]  →  data/preprocessed/{session_ID}.pt
        ↓  Step 3
[Model Training & Evaluation (LOSO CV)]  →  log/inference/exp_{n}/
```

---

## Step 1: Environment Setup

```bash
pip install -r requirements.txt
```

---

## Step 2: Feature Extraction

Extracts text and audio embeddings from transcript CSVs and audio WAVs, and saves them as `.pt` files.

```bash
python src/preprocessing.py feature_extraction \
  --pairs-csv      data/pairs.csv \
  --output-dir     data/preprocessed \
  --transcript-dir data/transcript \
  --audio-dir      data/audio
```

### Input

**`data/pairs.csv`** — each row represents one sample (speaker × session).

| Column | Required | Description |
|--------|:--------:|-------------|
| `speaker_id` | yes | ID of the target speaker |
| `partner_id` | yes | ID of the conversation partner |
| `score` | yes | Target score to predict |
| `session_ID` | | Output file stem (default: `{speaker_id}__{partner_id}`) |
| `speaker_label` | | Value in the transcript CSV `speaker` column that identifies the target speaker (default: `speaker_id`) |
| `transcript_path` | | Path to the transcript CSV (default: `{transcript_dir}/{speaker_id}__{partner_id}.csv`) |
| `wav_path` | | Path to the WAV file (default: `{audio_dir}/{speaker_id}__{partner_id}.wav`) |

**Transcript CSV** — columns: `start` (sec), `end` (sec), `speaker`, `text`

### Processing Steps (per sample)

0. Merge consecutive utterances by the same speaker
1. Extract audio segments by timestamp
2. VAD trimming (webrtcvad, 30 ms frames, 300 ms padding)
3. Extract text embeddings → `[n_utts, 768]` (L2-normalized)
4. Extract audio embeddings → `[n_utts, 1024]` (L2-normalized)
5. Save as `.pt` file

### Output

`data/preprocessed/{session_ID}.pt` contains:

```python
{
    "session_ID":       str,
    "speaker_id":       str,
    "partner_id":       str,
    "speaker":          str,                  # value of speaker_label
    "first_speaker":    str,                  # speaker of the first utterance
    "text_embeddings":  Tensor[n_utts, 768],  # Sentence-T5-large (L2-normalized)
    "audio_embeddings": Tensor[n_utts, 1024], # HuBERT-large (L2-normalized)
    "score":            float,                # target score
}
```

---

## Step 3: Model Training & Evaluation

Evaluates text, audio, and Late Fusion modalities with LOSO CV.

```bash
python src/main.py [options]
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | `score` | Key of the prediction target in `.pt` files |
| `--emb-dir` | `./data/preprocessed` | Directory containing `.pt` embedding files |
| `--out-dir` | `./log/inference` | Output directory for results |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `20` | Early stopping patience (epochs) |
| `--batch-size` | `6` | Number of sessions per batch |
| `--max-epochs` | `500` | Maximum training epochs |
| `--verbose` | `False` | Print per-epoch training logs |

### Examples

```bash
# Default settings
python src/main.py

# Custom target and hyperparameters
python src/main.py --target rapport --lr 0.0005 --patience 30 --verbose
```

### Output

Saved under `log/inference/exp_{n}/`:

| File | Contents |
|------|----------|
| `config.txt` | Run configuration (timestamp & hyperparameters) |
| `results_text_only.csv` | Text model predictions (all folds) |
| `results_audio_only.csv` | Audio model predictions (all folds) |
| `results_late_fusion.csv` | Late Fusion predictions (all folds) |
| `scatter_predictions.png` | y_true vs y_pred scatter plots (3 modalities) |
| `result.txt` | Final evaluation metrics aggregated over all folds (MAE / Pearson r / CCC) |

---

## Toy Dataset Generation (for sanity checks)

Generates a toy dataset (30 sessions) to verify the pipeline without real data.

```bash
python src/temp.py make_toy_dataset \
  --out-dir    data/toy \
  --n-utts-min 8 \
  --n-utts-max 20 \
  --seed       42
```

The generated `.pt` files can be passed to `main.py` via `--emb-dir`:

```bash
python src/main.py --emb-dir data/toy
```

**Design**: A learnable signal is embedded by adding a bias of `true_score × 6.0` to the first 32 dimensions of both text and audio embeddings, followed by L2 normalization.

---

## Directory Structure

```
proj-general-est-conv/
├── data/
│   ├── pairs.csv            # Pair and score information
│   ├── transcript/          # Transcript CSVs
│   ├── audio/               # Per-speaker WAV files
│   ├── preprocessed/        # .pt embedding files (generated in Step 2)
│   └── toy/                 # Toy .pt files (generated by temp.py)
├── log/
│   └── inference/           # Prediction results (generated in Step 3)
├── src/
│   ├── preprocessing.py     # Feature extraction (Step 2)
│   ├── main.py              # Training & evaluation main script (Step 3)
│   ├── training.py          # Model definition & training loop
│   ├── inference.py         # Inference & Late Fusion
│   ├── evaluation_metrics.py
│   └── temp.py              # Toy dataset generation utility
└── requirements.txt
```
