# 会話スコア予測パイプライン

任意の対話音声・書き起こしテキストから、話者が会話相手に抱くスコアを予測する回帰パイプライン。

---

## タスク定義

**入力**: 1 セッション分の書き起こし CSV（発話ごとの開始・終了時刻・話者・テキスト）と対応する音声 WAV。

**出力**: 注目話者が当該セッションで抱くスコアの予測値（連続値回帰）。

**特徴量**:

| ストリーム | モデル | 埋め込み次元 |
|-----------|--------|-------------|
| テキスト | sentence-transformers/sentence-t5-large | 768 (L2 正規化) |
| 音声 | facebook/hubert-large-ll60k | 1024 (L2 正規化) |

各ストリームについて、発話列を Additive Attention Pooling でセッション埋め込みに集約し、MLP ヘッドでスコアを予測する。最終予測は **Late Fusion**（テキスト予測と音声予測の単純平均）も出力する。

**交差検証**: Leave-One-Session-Out (LOSO) — セッションを 1 件ずつテストに回す。スケーラは訓練セットのみで fitting し、テストセットに適用（スケール逆変換後に評価）。

---

## 全体の流れ

```
[音声 WAV / 書き起こし CSV / スコア CSV]
        ↓  Step 2
[特徴量抽出]  →  data/preprocessed/{session_ID}.pt
        ↓  Step 3
[モデル学習・評価 (LOSO CV)]  →  log/inference/exp_{n}/
```

---

## Step 1: 環境構築

```bash
pip install -r requirements.txt
```

---

## Step 2: 特徴量抽出

書き起こし CSV と音声 WAV からテキスト・音声埋め込みを抽出し、`.pt` ファイルとして保存する。

```bash
python src/preprocessing.py feature_extraction \
  --pairs-csv      data/pairs.csv \
  --output-dir     data/preprocessed \
  --transcript-dir data/transcript \
  --audio-dir      data/audio
```

### 入力

**`data/pairs.csv`** — 1 行が 1 サンプル（話者×セッション）に対応する。

| 列名 | 必須 | 説明 |
|------|:----:|------|
| `speaker_id` | ○ | 注目話者の ID |
| `partner_id` | ○ | 会話相手の ID |
| `score` | ○ | 予測対象スコア |
| `session_ID` | | 出力ファイルのステム（省略時: `{speaker_id}__{partner_id}`） |
| `speaker_label` | | 書き起こし CSV の `speaker` 列で注目話者を示す値（省略時: `speaker_id`） |
| `transcript_path` | | 書き起こし CSV のパス（省略時: `{transcript_dir}/{speaker_id}__{partner_id}.csv`） |
| `wav_path` | | WAV ファイルのパス（省略時: `{audio_dir}/{speaker_id}__{partner_id}.wav`） |

**書き起こし CSV** — 列: `start`（秒）, `end`（秒）, `speaker`, `text`

### 処理ステップ（サンプルごと）

0. 連続同一話者発話のマージ
1. タイムスタンプによる音声セグメント抽出
2. VAD トリミング（webrtcvad、30 ms フレーム、300 ms パディング）
3. テキスト埋め込み抽出 → `[n_utts, 768]`（L2 正規化）
4. 音声埋め込み抽出 → `[n_utts, 1024]`（L2 正規化）
5. `.pt` ファイル保存

### 出力

`data/preprocessed/{session_ID}.pt` に以下を格納:

```python
{
    "session_ID":       str,
    "speaker_id":       str,
    "partner_id":       str,
    "speaker":          str,                  # speaker_label の値
    "first_speaker":    str,                  # 会話の最初の発話者
    "text_embeddings":  Tensor[n_utts, 768],  # Sentence-T5-large（L2 正規化）
    "audio_embeddings": Tensor[n_utts, 1024], # HuBERT-large（L2 正規化）
    "score":            float,                # 予測対象スコア
}
```

---

## Step 3: モデル学習・評価

LOSO CV でテキスト・音声・Late Fusion の 3 モダリティを評価する。

```bash
python src/main.py [options]
```

### オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--target` | `score` | `.pt` ファイル内の予測対象キー |
| `--emb-dir` | `./data/preprocessed` | `.pt` 埋め込みファイルのディレクトリ |
| `--out-dir` | `./log/inference` | 結果出力ディレクトリ |
| `--lr` | `0.001` | 学習率 |
| `--patience` | `20` | Early Stopping patience（エポック数） |
| `--batch-size` | `6` | バッチあたりのセッション数 |
| `--max-epochs` | `500` | 最大学習エポック数 |
| `--verbose` | `False` | エポックごとの訓練ログを表示 |

### 実行例

```bash
# デフォルト設定
python src/main.py

# ターゲット・ハイパーパラメータ指定
python src/main.py --target rapport --lr 0.0005 --patience 30 --verbose
```

### 出力

`log/inference/exp_{n}/` に以下を保存:

| ファイル | 内容 |
|---------|------|
| `config.txt` | 実行時設定（タイムスタンプ・ハイパーパラメータ） |
| `results_text_only.csv` | テキストモデルの予測結果（全 fold） |
| `results_audio_only.csv` | 音声モデルの予測結果（全 fold） |
| `results_late_fusion.csv` | Late Fusion の予測結果（全 fold） |
| `scatter_predictions.png` | y_true vs y_pred 散布図（3 モダリティ） |
| `result.txt` | 全 fold 集計の最終評価指標（MAE / Pearson r / CCC） |

---

## トイデータ生成（動作確認用）

実データなしで動作を確認するためのトイデータセット（30 セッション）を生成する。

```bash
python src/temp.py make_toy_dataset \
  --out-dir    data/toy \
  --n-utts-min 8 \
  --n-utts-max 20 \
  --seed       42
```

生成された `.pt` ファイルは `--emb-dir` に指定して `main.py` で使用できる:

```bash
python src/main.py --emb-dir data/toy
```

**設計**: テキスト・音声埋め込みの先頭 32 次元に `true_score × 6.0` のバイアスを加えて L2 正規化することで、モデルが学習できる信号を埋め込んでいる。

---

## ディレクトリ構成

```
proj-general-est-conv/
├── data/
│   ├── pairs.csv            # ペア・スコア情報
│   ├── transcript/          # 書き起こし CSV
│   ├── audio/               # 話者別 WAV ファイル
│   ├── preprocessed/        # .pt 埋め込みファイル（Step 2 で生成）
│   └── toy/                 # トイデータ .pt ファイル（temp.py で生成）
├── log/
│   └── inference/           # 予測結果（Step 3 で生成）
├── src/
│   ├── preprocessing.py     # 特徴量抽出（Step 2）
│   ├── main.py              # 学習・評価メインスクリプト（Step 3）
│   ├── training.py          # モデル定義・訓練ループ
│   ├── inference.py         # 推論・Late Fusion
│   ├── evaluation_metrics.py
│   └── temp.py              # トイデータ生成ユーティリティ
└── requirements.txt
```
