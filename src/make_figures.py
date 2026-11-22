"""Make before/after and ranking figures for the slides/video."""

from __future__ import annotations

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .apply import enhance_bgr, get_best_model_path
from .fit_models import load_aligned_scene
from .models import load_model
from .paths import ALIGNED, FIGURES, MODELS


def fig_triptych(scene_id: str, model_obj: dict):
    src, tgt, _mask = load_aligned_scene(ALIGNED / scene_id)
    pred = enhance_bgr(src, model_obj)

    h = 280

    def resize(im):
        scale = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), h))

    row = np.hstack([resize(src), resize(pred), resize(tgt)])
    row = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 4))
    plt.imshow(row)
    plt.axis("off")
    plt.title(f"{scene_id}:  source  |  enhanced  |  target (Night/HDR)")
    out = FIGURES / f"triptych_{scene_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Wrote {out}")


def fig_ranking():
    eval_path = MODELS / "eval_test.json"
    if not eval_path.exists():
        print("No eval_test.json — run evaluate first")
        return
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = data["ranking"]
    names = [r["model"].replace(".json", "") for r in rows]
    des = [r["test_delta_e"] for r in rows]
    base = rows[0]["baseline_delta_e"]

    plt.figure(figsize=(9, 4.5))
    y = np.arange(len(names))
    plt.barh(y, des, color="#BE123C")
    plt.axvline(base, color="#64748B", linestyle="--", label=f"baseline ΔE={base}")
    plt.yticks(y, names)
    plt.xlabel("Held-out colour error (lower is better)")
    plt.title("Model ranking on test scenes")
    plt.legend()
    plt.tight_layout()
    out = FIGURES / "model_ranking.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Wrote {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    split = MODELS / "split.json"
    if not split.exists():
        print("Run align_pairs → fit_models → evaluate first")
        return
    test_ids = json.loads(split.read_text(encoding="utf-8"))["test"]
    model_obj = load_model(get_best_model_path())
    for sid in test_ids[:3]:
        fig_triptych(sid, model_obj)
    fig_ranking()
    print(f"Figures in {FIGURES}")


if __name__ == "__main__":
    main()
