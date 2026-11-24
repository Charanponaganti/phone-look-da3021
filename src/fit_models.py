"""Fit course transforms + neural nets on aligned pairs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import bgr_to_lab01, sample_pixels
from .models import (
    fit_color_matrix,
    fit_lab_mlp,
    fit_root_poly,
    fit_root_poly_mlp,
    fit_tone_curves,
    save_model,
)
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

    print(f"Train scenes ({len(train_ids)}): {train_ids[:8]}{'…' if len(train_ids) > 8 else ''}")
    print(f"Test scenes  ({len(test_ids)}): {test_ids[:8]}{'…' if len(test_ids) > 8 else ''}")

    # Always keep personal Neo/scene captures in the pixel + CNN pools
    own_train = [s for s in train_ids if s.startswith("scene")]
    dped_train = [s for s in train_ids if not s.startswith("scene")]

    # Deadline path: cap DPED scenes so gather/fit finish in hours, not overnight
    pixel_ids = list(own_train)
    if len(dped_train) > 1500:
        pixel_ids += list(rng.choice(dped_train, size=1500, replace=False))
    else:
        pixel_ids += dped_train
    n_pix = 6000 if len(pixel_ids) > 200 else 25000
    if len(pixel_ids) > 800:
        n_pix = 2000
    print(f"Pixel pool: {len(pixel_ids)} scenes ({len(own_train)} personal)")
    Xtr, Ytr = gather_pixels(pixel_ids, n_per_scene=n_pix)
    print(f"Training pixels: {len(Xtr)} (n_per_scene={n_pix})", flush=True)

    meta = {"train": train_ids, "test": test_ids}

    # --- Course transforms ---
    tone = fit_tone_curves(Xtr, Ytr, degree_L=3)
    save_model(MODELS / "tone_curves.json", "tone_curves", tone, meta)
    print("Saved tone_curves.json")

    mat = fit_color_matrix(Xtr, Ytr)
    save_model(MODELS / "color_matrix.json", "color_matrix", mat, meta)
    print("Saved color_matrix.json")

    # Prefer stronger ridge — more stable OOD colour
    for deg, alpha in [(2, 10.0), (2, 50.0), (3, 10.0), (3, 50.0)]:
        rp = fit_root_poly(Xtr, Ytr, degree=deg, alpha=alpha)
        name = f"root_poly_d{deg}_a{alpha:g}.json"
        save_model(MODELS / name, "root_poly", rp, {**meta, "degree": deg, "alpha": alpha})
        print(f"Saved {name}")

    # --- Neural nets (CPU MLP) ---
    # Cap pixels so Adam stays responsive on laptop CPU
    rng = np.random.default_rng(0)
    n_nn = min(len(Xtr), 80_000)
    idx = rng.choice(len(Xtr), size=n_nn, replace=False)
    Xnn, Ynn = Xtr[idx], Ytr[idx]

    print(f"Fitting Lab MLP on {len(Xnn)} pixels…")
    mlp = fit_lab_mlp(Xnn, Ynn, hidden=(64, 32), max_iter=120)
    save_model(MODELS / "lab_mlp.json", "lab_mlp", mlp, {**meta, "hidden": [64, 32]})
    print(f"Saved lab_mlp.json (iters={mlp['n_iter']})")

    print(f"Fitting root-poly + MLP on {len(Xnn)} pixels…")
    rp_mlp = fit_root_poly_mlp(Xnn, Ynn, degree=2, hidden=(64, 32), max_iter=120)
    save_model(
        MODELS / "root_poly_mlp.json",
        "root_poly_mlp",
        rp_mlp,
        {**meta, "degree": 2, "hidden": [64, 32]},
    )
    print(f"Saved root_poly_mlp.json (iters={rp_mlp['n_iter']})")

    # --- Tiny residual CNN v2 (DPED-style local filters, CPU) ---
    print("Fitting TinyResCNN v2 (full patches + multi-scale, CPU)…")
    from .tiny_cnn import fit_tiny_cnn

    # Cap DPED for CNN; always include personal scenes
    rng_cnn = np.random.default_rng(7)
    cnn_dped = dped_train
    if len(cnn_dped) > 1000:
        cnn_dped = list(rng_cnn.choice(cnn_dped, size=1000, replace=False))
    cnn_ids = list(own_train) + cnn_dped
    print(f"  CNN using {len(cnn_ids)} scenes ({len(own_train)} personal)", flush=True)
    n_per = 4 if len(cnn_ids) > 400 else (8 if len(cnn_ids) > 80 else 16)
    epochs = 25 if len(cnn_ids) > 400 else 40
    train_dirs = [ALIGNED / sid for sid in cnn_ids]
    cnn_payload = fit_tiny_cnn(
        train_dirs,
        out_weights=MODELS / "tiny_cnn.pt",
        channels=32,
        depth=6,
        patch=64,
        epochs=epochs,
        batch_size=16,
        n_per_scene=n_per,
    )
    save_model(MODELS / "tiny_cnn.json", "tiny_cnn", cnn_payload, meta)
    print("Saved tiny_cnn.json + tiny_cnn.pt")

    split = {"train": train_ids, "test": test_ids}
    (MODELS / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Models written to {MODELS}")


if __name__ == "__main__":
    main()
