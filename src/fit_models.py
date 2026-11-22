"""Fit tone / matrix / root-polynomial models on aligned pairs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import bgr_to_lab01, sample_pixels
from .models import fit_color_matrix, fit_root_poly, fit_tone_curves, save_model
from .paths import ALIGNED, MODELS


def load_aligned_scene(scene_dir: Path):
    src = cv2.imread(str(scene_dir / "src.png"))
    tgt = cv2.imread(str(scene_dir / "tgt_warped.png"))
    mask = cv2.imread(str(scene_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if src is None or tgt is None or mask is None:
        raise FileNotFoundError(scene_dir)
    return src, tgt, (mask > 127).astype(np.uint8)


def gather_pixels(scene_ids: list[str], n_per_scene: int = 25000, seed: int = 0):
    xs, ys = [], []
    for i, sid in enumerate(scene_ids):
        src, tgt, mask = load_aligned_scene(ALIGNED / sid)
        s_lab = bgr_to_lab01(src)
        t_lab = bgr_to_lab01(tgt)
        s, t = sample_pixels(s_lab, t_lab, mask, n=n_per_scene, seed=seed + i)
        xs.append(s)
        ys.append(t)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main():
    scenes = sorted([p.name for p in ALIGNED.iterdir() if p.is_dir()])
    if len(scenes) < 2:
        print("Need at least 2 aligned scenes. Put pairs in data/raw/pairs/ then run:")
        print("  python -m src.align_pairs")
        return

    # Scene-level split (important: never split pixels from same scene)
    rng = np.random.default_rng(42)
    order = list(scenes)
    rng.shuffle(order)
    n_test = max(1, len(order) // 4)
    test_ids = order[:n_test]
    train_ids = order[n_test:]
    if not train_ids:
        train_ids, test_ids = order[:-1], order[-1:]

    print(f"Train scenes ({len(train_ids)}): {train_ids}")
    print(f"Test scenes  ({len(test_ids)}): {test_ids}")

    Xtr, Ytr = gather_pixels(train_ids)
    print(f"Training pixels: {len(Xtr)}")

    # Model ladder
    tone = fit_tone_curves(Xtr, Ytr, degree=3)
    save_model(MODELS / "tone_curves.json", "tone_curves", tone, {"train": train_ids, "test": test_ids})

    mat = fit_color_matrix(Xtr, Ytr)
    save_model(MODELS / "color_matrix.json", "color_matrix", mat, {"train": train_ids, "test": test_ids})

    for deg, alpha in [(2, 1.0), (2, 10.0), (3, 1.0), (3, 10.0)]:
        rp = fit_root_poly(Xtr, Ytr, degree=deg, alpha=alpha)
        name = f"root_poly_d{deg}_a{alpha:g}.json"
        save_model(
            MODELS / name,
            "root_poly",
            rp,
            {"train": train_ids, "test": test_ids, "degree": deg, "alpha": alpha},
        )
        print(f"Saved {name}")

    split = {"train": train_ids, "test": test_ids}
    (MODELS / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Models written to {MODELS}")


if __name__ == "__main__":
    main()
