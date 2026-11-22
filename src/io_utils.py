from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .paths import RAW_PAIRS


def read_bgr(path: Path, max_side: int = 1600) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def write_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def list_pair_ids(folder: Path = RAW_PAIRS) -> list[str]:
    """Find scene IDs that have both *_src and *_tgt files."""
    src = {p.stem.replace("_src", "") for p in folder.glob("*_src.*")}
    tgt = {p.stem.replace("_tgt", "") for p in folder.glob("*_tgt.*")}
    ids = sorted(src & tgt)
    return ids


def pair_paths(scene_id: str, folder: Path = RAW_PAIRS) -> tuple[Path, Path]:
    src_cands = list(folder.glob(f"{scene_id}_src.*"))
    tgt_cands = list(folder.glob(f"{scene_id}_tgt.*"))
    if not src_cands or not tgt_cands:
        raise FileNotFoundError(f"Missing pair for {scene_id}")
    return src_cands[0], tgt_cands[0]


def bgr_to_lab01(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] /= 255.0
    lab[..., 1] = (lab[..., 1] - 128.0) / 127.0
    lab[..., 2] = (lab[..., 2] - 128.0) / 127.0
    return lab


def lab01_to_bgr(lab01: np.ndarray) -> np.ndarray:
    lab = np.empty_like(lab01, dtype=np.float32)
    lab[..., 0] = np.clip(lab01[..., 0] * 255.0, 0, 255)
    lab[..., 1] = np.clip(lab01[..., 1] * 127.0 + 128.0, 0, 255)
    lab[..., 2] = np.clip(lab01[..., 2] * 127.0 + 128.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def sample_pixels(src_lab: np.ndarray, tgt_lab: np.ndarray, mask: np.ndarray, n: int = 40000, seed: int = 0):
    ys, xs = np.where(mask > 0)
    # DPED patches are 100x100 (~10k px); allow smaller floors for tiny crops
    min_need = min(1000, max(100, mask.size // 4))
    if len(ys) < min_need:
        raise RuntimeError("Too few valid pixels after masking/alignment")
    rng = np.random.default_rng(seed)
    take = min(n, len(ys))
    idx = rng.choice(len(ys), size=take, replace=False)
    s = src_lab[ys[idx], xs[idx]]
    t = tgt_lab[ys[idx], xs[idx]]
    return s, t
