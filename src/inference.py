"""
Inference helpers for conversation score prediction.

run_inference : run independently trained text/audio models on a list of samples,
                return rows ready for CSV output.
                Late Fusion = simple average of the two models' predictions.
"""

import torch


def run_inference(
    text_model,
    audio_model,
    samples: list,
    device: torch.device,
    scaler_mean: float,
    scaler_std: float,
) -> dict[str, list]:
    """
    Run inference on samples using independently trained text and audio models,
    and return three lists of result dicts (text_only, audio_only, late_fusion).

    Late Fusion is the simple average of text and audio predictions.

    Returns
    -------
    { "text_only": [...], "audio_only": [...], "late_fusion": [...] }
    """
    text_model.eval();  text_model.to(device)
    audio_model.eval(); audio_model.to(device)

    text_rows   = []
    audio_rows  = []
    fusion_rows = []

    def inv(score: float) -> float:
        return score * scaler_std + scaler_mean

    with torch.no_grad():
        for s in samples:
            text_x  = s["text_embeddings"].unsqueeze(0).to(device).float()
            audio_x = s["audio_embeddings"].unsqueeze(0).to(device).float()
            text_m  = torch.ones(1, text_x.shape[1],  dtype=torch.bool, device=device)
            audio_m = torch.ones(1, audio_x.shape[1], dtype=torch.bool, device=device)

            text_s  = text_model(text_x,  text_m)
            audio_s = audio_model(audio_x, audio_m)
            fusion_s = (text_s + audio_s) / 2.0
            y_true   = s["_target_orig"]

            base = dict(
                session_ID=s["session_ID"],
                y_true=round(float(y_true), 6),
            )
            text_rows.append({**base,  "y_pred": round(inv(text_s.item()),  6)})
            audio_rows.append({**base, "y_pred": round(inv(audio_s.item()), 6)})
            fusion_rows.append({**base,"y_pred": round(inv(fusion_s.item()),6)})

    return {
        "text_only":   text_rows,
        "audio_only":  audio_rows,
        "late_fusion": fusion_rows,
    }
