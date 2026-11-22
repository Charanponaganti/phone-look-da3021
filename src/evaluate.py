"""Evaluate models on held-out scenes: ΔE proxy + SSIM."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .fit_models import load_aligned_scene
from .io_utils import bgr_to_lab01, lab01_to_bgr
from .models import apply_model_lab, load_model
from .paths import ALIGNED, FIGURES, MODELS


def delta_e_approx(lab_a: np.ndarray, lab_b: np.ndarray, mask: np.ndarray) -> float:
    """Simple Euclidean distance in scaled Lab (proxy for ΔE). Good enough for ranking models."""
    d = lab_a - lab_b
    # Weight L a bit less than chroma for ranking stability
    dist = np.sqrt((0.8 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    vals = dist[mask > 0]
    return float(vals.mean()) if len(vals) else float("nan")


def eval_scene(model_obj: dict, scene_id: str) -> dict:
    src, tgt, mask = load_aligned_scene(ALIGNED / scene_id)
    src_lab = bgr_to_lab01(src)
    tgt_lab = bgr_to_lab01(tgt)
    pred_lab = apply_model_lab(src_lab, model_obj)
    pred = lab01_to_bgr(pred_lab)

    de = delta_e_approx(pred_lab, tgt_lab, mask)
    de_base = delta_e_approx(src_lab, tgt_lab, mask)

    # SSIM on luminance
    pred_y = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    tgt_y = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    # Only where mask is valid: approximate by zeroing outside
    pred_y = pred_y.copy()
    tgt_y = tgt_y.copy()
    pred_y[mask == 0] = 0
    tgt_y[mask == 0] = 0
    s = float(ssim(tgt_y, pred_y, data_range=255))
    s_base = float(ssim(tgt_y, cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) * (mask > 0), data_range=255))

    return {
        "delta_e_pred": de,
        "delta_e_src": de_base,
        "ssim_pred": s,
        "ssim_src": s_base,
    }


def main():
    split_path = MODELS / "split.json"
    if not split_path.exists():
        print("Run python -m src.fit_models first")
        return
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = split["test"]
    model_files = sorted(MODELS.glob("*.json"))
    model_files = [p for p in model_files if p.name != "split.json"]

    rows = []
    for mf in model_files:
        obj = load_model(mf)
        scene_stats = [eval_scene(obj, sid) for sid in test_ids]
        de = np.nanmean([s["delta_e_pred"] for s in scene_stats])
        de0 = np.nanmean([s["delta_e_src"] for s in scene_stats])
        ss = np.nanmean([s["ssim_pred"] for s in scene_stats])
        rows.append(
            {
                "model": mf.name,
                "kind": obj["kind"],
                "test_delta_e": round(float(de), 4),
                "baseline_delta_e": round(float(de0), 4),
                "improvement_pct": round(100.0 * (1 - de / de0), 2) if de0 else None,
                "test_ssim": round(float(ss), 4),
            }
        )
        print(f"{mf.name:30s}  ΔE={de:.4f} (base {de0:.4f})  SSIM={ss:.4f}")

    rows = sorted(rows, key=lambda r: r["test_delta_e"])
    out = MODELS / "eval_test.json"
    out.write_text(json.dumps({"test_scenes": test_ids, "ranking": rows}, indent=2), encoding="utf-8")
    print(f"\nBest on held-out ΔE: {rows[0]['model']}")
    print(f"Wrote {out}")

    # Remember best model path for the app
    (MODELS / "best_model.txt").write_text(rows[0]["model"], encoding="utf-8")


if __name__ == "__main__":
    main()
