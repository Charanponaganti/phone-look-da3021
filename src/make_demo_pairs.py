"""Create synthetic demo pairs so the pipeline can be smoke-tested without phone photos."""

from __future__ import annotations

import cv2
import numpy as np

from .paths import RAW_PAIRS


def make_scene(seed: int, size: int = 480) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), np.uint8)
    base[..., 0] = np.clip(40 + 0.2 * xx + rng.normal(0, 4, (size, size)), 0, 255)
    base[..., 1] = np.clip(60 + 0.15 * yy + rng.normal(0, 4, (size, size)), 0, 255)
    base[..., 2] = np.clip(90 + 0.1 * (xx + yy) / 2 + rng.normal(0, 4, (size, size)), 0, 255)

    # Add a few coloured blocks for texture/features
    for _ in range(8):
        x0, y0 = rng.integers(20, size - 80, size=2)
        w, h = rng.integers(30, 70, size=2)
        color = rng.integers(40, 220, size=3)
        base[y0 : y0 + h, x0 : x0 + w] = color

    src = base.copy()
    # Target = brighter + warmer (fake Night/HDR look)
    tgt = cv2.convertScaleAbs(src, alpha=1.25, beta=18)
    tgt[..., 2] = np.clip(tgt[..., 2].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    # Tiny shift to force alignment to work
    M = np.float32([[1, 0, 3], [0, 1, -2]])
    tgt = cv2.warpAffine(tgt, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    return src, tgt


def main():
    RAW_PAIRS.mkdir(parents=True, exist_ok=True)
    for i in range(1, 9):
        src, tgt = make_scene(i)
        sid = f"scene{i:02d}"
        cv2.imwrite(str(RAW_PAIRS / f"{sid}_src.jpg"), src)
        cv2.imwrite(str(RAW_PAIRS / f"{sid}_tgt.jpg"), tgt)
        print(f"Wrote {sid}")
    print("Demo pairs ready. Next: python -m src.align_pairs")


if __name__ == "__main__":
    main()
