"""Make before/after and ranking figures (OpenCV only)."""

from __future__ import annotations

import json

import cv2
import numpy as np

from .apply import enhance_bgr, get_best_model_path
from .fit_models import load_aligned_scene
from .models import load_model
from .paths import ALIGNED, FIGURES, MODELS


def fig_triptych(scene_id: str, pred_bgr: np.ndarray, title_suffix: str = ""):
    src, tgt, _mask = load_aligned_scene(ALIGNED / scene_id)
    h = 280

    def resize(im):
        scale = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), h))

    row = np.hstack([resize(src), resize(pred_bgr), resize(tgt)])
    bar = np.full((36, row.shape[1], 3), 245, dtype=np.uint8)
    label = f"{scene_id}: source | enhanced{title_suffix} | target"
    cv2.putText(bar, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    out_img = np.vstack([bar, row])
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"triptych_{scene_id}.png"
    cv2.imwrite(str(out), out_img)
    print(f"Wrote {out}")


def fig_ranking():
    path = MODELS / "eval_test.json"
    if not path.exists():
        print("No eval_test.json — run evaluate first")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["ranking"]
    names = [r["model"].replace(".json", "") for r in rows]
    vals = [r.get("composite", r["test_delta_e"]) for r in rows]
    base = rows[0]["baseline_delta_e"] if rows else 0

    w, h = 960, max(300, 50 + 28 * len(rows))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Composite score (lower better): ΔE + chroma penalty",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )
    if not vals:
        return
    vmax = max(max(vals), base, 1e-6)
    left, right, top = 280, w - 40, 55
    bar_h = 18
    for i, (name, v) in enumerate(zip(names, vals)):
        y = top + i * 28
        cv2.putText(canvas, name[:36], (10, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        bw = int((right - left) * (v / vmax))
        color = (40, 160, 80) if name == "AUTO" else ((60, 140, 60) if v <= base else (60, 60, 180))
        cv2.rectangle(canvas, (left, y), (left + max(bw, 1), y + bar_h), color, -1)
        cv2.putText(
            canvas,
            f"{v:.4f}",
            (left + bw + 6, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    out = FIGURES / "model_ranking.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"Wrote {out}")


def main():
    split = json.loads((MODELS / "split.json").read_text(encoding="utf-8"))
    for sid in split["test"][:3]:
        pred = enhance_bgr(
            load_aligned_scene(ALIGNED / sid)[0],
            mode="auto",
            strength=0.75,
            preset="Balanced",
        )
        fig_triptych(sid, pred, title_suffix=" (Auto)")
    fig_ranking()
    print(f"Figures in {FIGURES}")


if __name__ == "__main__":
    main()
