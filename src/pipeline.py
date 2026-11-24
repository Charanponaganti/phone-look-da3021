"""
Auto enhancement cascade: course colour + TinyCNN + L-clarity.

Presets tune how aggressive each stage is. Adaptive strength pulls back on
high-chroma / sunset-like images so they don't go neon.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .io_utils import bgr_to_lab01, lab01_to_bgr
from .models import (
    apply_color_matrix,
    apply_tone_curves,
    load_model,
    stabilize_lab,
)
from .paths import MODELS

PRESETS = {
    "Natural": {
        "exposure": 0.35,
        "colour": 0.55,
        "cnn": 0.35,
        "clarity": 0.25,
        "max_da": 0.14,
        "max_db": 0.14,
        "max_dL": 0.22,
    },
    "Balanced": {
        "exposure": 0.45,
        "colour": 0.70,
        "cnn": 0.55,
        "clarity": 0.45,
        "max_da": 0.16,
        "max_db": 0.16,
        "max_dL": 0.26,
    },
    "Punchy": {
        "exposure": 0.55,
        "colour": 0.80,
        "cnn": 0.65,
        "clarity": 0.70,
        "max_da": 0.18,
        "max_db": 0.18,
        "max_dL": 0.30,
    },
}


@lru_cache(maxsize=4)
def _load(name: str) -> dict:
    path = MODELS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return load_model(path)


def adaptive_strength(lab: np.ndarray, strength: float) -> float:
    """Scale strength down for high-chroma / saturated highlight images."""
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    mean_c = float(chroma.mean())
    hi = lab[..., 0] > 0.65
    sat_hi = float(chroma[hi].mean()) if np.any(hi) else mean_c
    factor = 1.0
    if mean_c > 0.22:
        factor *= 0.75
    if sat_hi > 0.35:
        factor *= 0.7
    if mean_c > 0.35:
        factor *= 0.85
    return float(np.clip(strength * factor, 0.25, 1.0))


def _percentile_stretch_L(lab: np.ndarray, amount: float) -> np.ndarray:
    out = lab.copy().astype(np.float64)
    L = out[..., 0]
    lo, hi = np.percentile(L, [2, 98])
    if hi - lo < 1e-4:
        return out
    stretched = np.clip((L - lo) / (hi - lo), 0.0, 1.0)
    # Slight mid contrast (S-curve lite)
    stretched = 0.5 + 1.15 * (stretched - 0.5)
    stretched = np.clip(stretched, 0.0, 1.0)
    out[..., 0] = (1.0 - amount) * L + amount * stretched
    return out


def _blend_colour(lab: np.ndarray, amount: float) -> np.ndarray:
    """L from tone curves + chroma from colour matrix, then blend with input."""
    tone = _load("tone_curves.json")
    mat = _load("color_matrix.json")
    tone_lab = apply_tone_curves(lab, tone["payload"])
    mat_lab = apply_color_matrix(lab, mat["payload"])
    mixed = lab.copy().astype(np.float64)
    mixed[..., 0] = tone_lab[..., 0]
    mixed[..., 1] = mat_lab[..., 1]
    mixed[..., 2] = mat_lab[..., 2]
    return lab.astype(np.float64) * (1.0 - amount) + mixed * amount


def _apply_cnn(lab: np.ndarray, amount: float) -> np.ndarray:
    cnn_path = MODELS / "tiny_cnn.json"
    if not cnn_path.exists() or amount <= 1e-6:
        return lab
    from .tiny_cnn import apply_tiny_cnn_lab

    obj = _load("tiny_cnn.json")
    pred = apply_tiny_cnn_lab(lab.astype(np.float32), obj["payload"])
    return lab.astype(np.float64) * (1.0 - amount) + pred.astype(np.float64) * amount


def _unsharp_L(lab: np.ndarray, amount: float, radius: float = 1.2) -> np.ndarray:
    if amount <= 1e-6:
        return lab
    out = lab.copy().astype(np.float64)
    L = out[..., 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), radius)
    detail = L - blur
    out[..., 0] = np.clip(L + amount * 1.4 * detail, 0.0, 1.0)
    return out


def enhance_auto(
    bgr: np.ndarray,
    strength: float = 0.75,
    preset: str = "Balanced",
) -> np.ndarray:
    """Full Auto cascade → BGR uint8."""
    cfg = PRESETS.get(preset, PRESETS["Balanced"])
    src_lab = bgr_to_lab01(bgr)
    s = adaptive_strength(src_lab, float(strength))

    lab = src_lab.astype(np.float64)
    lab = _percentile_stretch_L(lab, cfg["exposure"] * s)
    lab = _blend_colour(lab, cfg["colour"] * s)
    lab = _apply_cnn(lab, cfg["cnn"] * s)
    lab = _unsharp_L(lab, cfg["clarity"] * s)

    lab = stabilize_lab(
        src_lab,
        lab,
        strength=1.0,
        max_dL=cfg["max_dL"],
        max_da=cfg["max_da"],
        max_db=cfg["max_db"],
    )
    return lab01_to_bgr(lab)


def list_presets() -> list[str]:
    return list(PRESETS.keys())
