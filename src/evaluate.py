"""Evaluate models + Auto cascade: ΔE, SSIM, chroma penalty, composite rank."""

from __future__ import annotations

import json

import cv2
import numpy as np

from .fit_models import load_aligned_scene
from .io_utils import bgr_to_lab01, lab01_to_bgr
from .models import apply_model_lab, load_model
from .paths import ALIGNED, MODELS


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den else 0.0


def delta_e_approx(lab_a: np.ndarray, lab_b: np.ndarray, mask: np.ndarray) -> float:
    d = lab_a - lab_b
    dist = np.sqrt((0.8 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    vals = dist[mask > 0]
    return float(vals.mean()) if len(vals) else float("nan")


def chroma_penalty(src_lab: np.ndarray, pred_lab: np.ndarray, mask: np.ndarray) -> dict:
    d = pred_lab - src_lab
    m = mask > 0
    da = np.abs(d[..., 1][m])
    db = np.abs(d[..., 2][m])
    return {
        "mean_abs_da": float(da.mean()) if len(da) else 0.0,
        "mean_abs_db": float(db.mean()) if len(db) else 0.0,
        "max_abs_da": float(da.max()) if len(da) else 0.0,
        "max_abs_db": float(db.max()) if len(db) else 0.0,
        "chroma_penalty": float(da.mean() + db.mean()) if len(da) else 0.0,
    }


def _metrics(src, tgt, pred, mask, src_lab, tgt_lab, pred_lab) -> dict:
    de = delta_e_approx(pred_lab, tgt_lab, mask)
    de_base = delta_e_approx(src_lab, tgt_lab, mask)
    pred_y = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    tgt_y = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    src_y = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).copy()
    pred_y = pred_y.copy()
    tgt_y = tgt_y.copy()
    pred_y[mask == 0] = 0
    tgt_y[mask == 0] = 0
    src_y[mask == 0] = 0
    ch = chroma_penalty(src_lab, pred_lab, mask)
    return {
        "delta_e_pred": de,
        "delta_e_src": de_base,
        "ssim_pred": ssim_gray(tgt_y, pred_y),
        "ssim_src": ssim_gray(tgt_y, src_y),
        **ch,
    }


def eval_scene_model(model_obj: dict, scene_id: str) -> dict:
    src, tgt, mask = load_aligned_scene(ALIGNED / scene_id)
    src_lab = bgr_to_lab01(src)
    tgt_lab = bgr_to_lab01(tgt)
    pred_lab = apply_model_lab(src_lab, model_obj, strength=1.0)
    pred = lab01_to_bgr(pred_lab)
    return _metrics(src, tgt, pred, mask, src_lab, tgt_lab, pred_lab)


def eval_scene_auto(scene_id: str, strength: float = 0.75, preset: str = "Balanced") -> dict:
    from .pipeline import enhance_auto

    src, tgt, mask = load_aligned_scene(ALIGNED / scene_id)
    src_lab = bgr_to_lab01(src)
    tgt_lab = bgr_to_lab01(tgt)
    pred = enhance_auto(src, strength=strength, preset=preset)
    pred_lab = bgr_to_lab01(pred)
    return _metrics(src, tgt, pred, mask, src_lab, tgt_lab, pred_lab)


def _aggregate(scene_stats: list[dict]) -> dict:
    de = float(np.nanmean([s["delta_e_pred"] for s in scene_stats]))
    de0 = float(np.nanmean([s["delta_e_src"] for s in scene_stats]))
    ss = float(np.nanmean([s["ssim_pred"] for s in scene_stats]))
    cp = float(np.nanmean([s["chroma_penalty"] for s in scene_stats]))
    mx = float(np.nanmean([max(s["max_abs_da"], s["max_abs_db"]) for s in scene_stats]))
    # Composite: lower better. Penalize chroma blow-ups hard.
    composite = de + 0.35 * cp + 0.15 * mx
    return {
        "test_delta_e": round(de, 4),
        "baseline_delta_e": round(de0, 4),
        "improvement_pct": round(100.0 * (1 - de / de0), 2) if de0 else None,
        "test_ssim": round(ss, 4),
        "chroma_penalty": round(cp, 4),
        "max_chroma_shift": round(mx, 4),
        "composite": round(composite, 4),
    }


def main():
    split_path = MODELS / "split.json"
    if not split_path.exists():
        print("Run python -m src.fit_models first")
        return
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = split["test"]
    skip = {"split.json", "eval_test.json"}
    model_files = sorted(p for p in MODELS.glob("*.json") if p.name not in skip)

    rows = []
    for mf in model_files:
        obj = load_model(mf)
        scene_stats = [eval_scene_model(obj, sid) for sid in test_ids]
        agg = _aggregate(scene_stats)
        rows.append({"model": mf.name, "kind": obj["kind"], **agg})
        print(
            f"{mf.name:30s}  ΔE={agg['test_delta_e']:.4f}  "
            f"chroma={agg['chroma_penalty']:.4f}  composite={agg['composite']:.4f}  "
            f"SSIM={agg['test_ssim']:.4f}"
        )

    # Auto cascade
    try:
        auto_stats = [eval_scene_auto(sid) for sid in test_ids]
        auto_agg = _aggregate(auto_stats)
        rows.append({"model": "AUTO", "kind": "auto_cascade", **auto_agg})
        print(
            f"{'AUTO':30s}  ΔE={auto_agg['test_delta_e']:.4f}  "
            f"chroma={auto_agg['chroma_penalty']:.4f}  composite={auto_agg['composite']:.4f}  "
            f"SSIM={auto_agg['test_ssim']:.4f}"
        )
    except Exception as e:
        print(f"[skip Auto eval] {e}")

    rows = sorted(rows, key=lambda r: r["composite"])
    out = MODELS / "eval_test.json"
    out.write_text(json.dumps({"test_scenes": test_ids, "ranking": rows}, indent=2), encoding="utf-8")
    print(f"\nBest composite: {rows[0]['model']}")
    print(f"Wrote {out}")

    # App default single-model fallback: best non-Auto JSON
    singles = [r for r in rows if r["model"] != "AUTO"]
    best_single = singles[0]["model"] if singles else rows[0]["model"]
    (MODELS / "best_model.txt").write_text(best_single, encoding="utf-8")
    (MODELS / "app_default.txt").write_text("AUTO", encoding="utf-8")


if __name__ == "__main__":
    main()
