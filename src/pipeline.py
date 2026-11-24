"""
Auto enhancement cascade with multi-scenario detection.

Scenarios: daylight, haze/fog, sunset/golden, night/low-light, backlight,
portrait-safe, indoor/LED, vivid/colorful, text/detail, default balanced.
"""

from __future__ import annotations

from functools import lru_cache

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
        "dehaze": 0.45,
        "max_da": 0.14,
        "max_db": 0.14,
        "max_dL": 0.22,
    },
    "Balanced": {
        "exposure": 0.45,
        "colour": 0.70,
        "cnn": 0.55,
        "clarity": 0.45,
        "dehaze": 0.70,
        "max_da": 0.16,
        "max_db": 0.16,
        "max_dL": 0.26,
    },
    "Punchy": {
        "exposure": 0.55,
        "colour": 0.80,
        "cnn": 0.65,
        "clarity": 0.70,
        "dehaze": 0.90,
        "max_da": 0.18,
        "max_db": 0.18,
        "max_dL": 0.30,
    },
}

# Multipliers applied on top of preset knobs (1.0 = unchanged)
SCENARIO_TUNING = {
    "daylight": {
        "label": "Daylight",
        "exposure": 1.0,
        "colour": 1.05,
        "cnn": 1.0,
        "clarity": 1.05,
        "dehaze": 0.35,
        "clahe": 0.15,
        "strength": 1.0,
        "max_dL": 1.0,
        "max_da": 1.0,
        "max_db": 1.0,
    },
    "haze": {
        "label": "Fog / haze / overcast",
        "exposure": 1.35,
        "colour": 0.95,
        "cnn": 0.9,
        "clarity": 1.45,
        "dehaze": 1.0,
        "clahe": 0.85,
        "strength": 1.2,
        "max_dL": 1.55,
        "max_da": 0.9,
        "max_db": 0.9,
    },
    "sunset": {
        "label": "Sunset / golden hour",
        "exposure": 0.85,
        "colour": 0.55,
        "cnn": 0.5,
        "clarity": 0.7,
        "dehaze": 0.2,
        "clahe": 0.1,
        "strength": 0.65,
        "max_dL": 0.85,
        "max_da": 0.55,
        "max_db": 0.55,
    },
    "night": {
        "label": "Night / low light",
        "exposure": 1.4,
        "colour": 0.85,
        "cnn": 0.75,
        "clarity": 0.55,
        "dehaze": 0.15,
        "clahe": 0.65,
        "strength": 1.15,
        "max_dL": 1.45,
        "max_da": 0.75,
        "max_db": 0.75,
    },
    "backlight": {
        "label": "Backlight / silhouette",
        "exposure": 1.35,
        "colour": 0.8,
        "cnn": 0.7,
        "clarity": 0.9,
        "dehaze": 0.25,
        "clahe": 0.75,
        "strength": 1.1,
        "max_dL": 1.4,
        "max_da": 0.8,
        "max_db": 0.8,
    },
    "portrait": {
        "label": "Portrait / skin-safe",
        "exposure": 0.9,
        "colour": 0.65,
        "cnn": 0.55,
        "clarity": 0.5,
        "dehaze": 0.15,
        "clahe": 0.2,
        "strength": 0.8,
        "max_dL": 0.9,
        "max_da": 0.55,
        "max_db": 0.55,
    },
    "indoor": {
        "label": "Indoor / LED",
        "exposure": 1.1,
        "colour": 1.0,
        "cnn": 0.95,
        "clarity": 0.85,
        "dehaze": 0.2,
        "clahe": 0.35,
        "strength": 1.0,
        "max_dL": 1.1,
        "max_da": 0.85,
        "max_db": 0.85,
    },
    "vivid": {
        "label": "Vivid / colourful",
        "exposure": 0.95,
        "colour": 0.7,
        "cnn": 0.65,
        "clarity": 0.85,
        "dehaze": 0.2,
        "clahe": 0.2,
        "strength": 0.75,
        "max_dL": 0.95,
        "max_da": 0.65,
        "max_db": 0.65,
    },
    "detail": {
        "label": "Text / detail",
        "exposure": 1.05,
        "colour": 0.75,
        "cnn": 0.7,
        "clarity": 1.5,
        "dehaze": 0.25,
        "clahe": 0.55,
        "strength": 1.05,
        "max_dL": 1.15,
        "max_da": 0.7,
        "max_db": 0.7,
    },
    "default": {
        "label": "General",
        "exposure": 1.0,
        "colour": 1.0,
        "cnn": 1.0,
        "clarity": 1.0,
        "dehaze": 0.45,
        "clahe": 0.25,
        "strength": 1.0,
        "max_dL": 1.0,
        "max_da": 1.0,
        "max_db": 1.0,
    },
}


@lru_cache(maxsize=4)
def _load(name: str) -> dict:
    path = MODELS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return load_model(path)


def _lab_stats(lab: np.ndarray) -> dict:
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    h, w = L.shape
    # Center vs border for backlight / portrait cues
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center = L[cy0:cy1, cx0:cx1]
    border = np.concatenate([L[: h // 8].ravel(), L[-h // 8 :].ravel(), L[:, : w // 8].ravel(), L[:, -w // 8 :].ravel()])
    # Edge energy (detail / text)
    gx = cv2.Sobel((L * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel((L * 255).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.sqrt(gx * gx + gy * gy)) / 255.0)
    # Skin-ish pixels in Lab (very rough): a>0, b>0, mid L
    skin = (L > 0.35) & (L < 0.85) & (a > 0.02) & (a < 0.35) & (b > 0.0) & (b < 0.35)
    return {
        "mean_L": float(L.mean()),
        "std_L": float(L.std()),
        "p10_L": float(np.percentile(L, 10)),
        "p90_L": float(np.percentile(L, 90)),
        "mean_c": float(chroma.mean()),
        "p90_c": float(np.percentile(chroma, 90)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "center_L": float(center.mean()) if center.size else float(L.mean()),
        "border_L": float(border.mean()) if border.size else float(L.mean()),
        "edge": edge,
        "skin_frac": float(skin.mean()),
    }


def detect_scenario(lab: np.ndarray) -> tuple[str, dict, dict]:
    """Return (scenario_id, tuning, scores)."""
    s = _lab_stats(lab)
    scores: dict[str, float] = {}

    # Haze / fog / washed overcast
    bright = np.clip((s["mean_L"] - 0.45) / 0.35, 0, 1)
    flat = np.clip((0.22 - s["std_L"]) / 0.18, 0, 1)
    muted = np.clip((0.18 - s["mean_c"]) / 0.15, 0, 1)
    scores["haze"] = 0.45 * bright + 0.35 * flat + 0.20 * muted

    # Sunset / golden — warm b+ chroma, orange/red bias
    warm = np.clip((s["mean_b"] - 0.02) / 0.20, 0, 1)
    hi_c = np.clip((s["p90_c"] - 0.22) / 0.35, 0, 1)
    warm_bias = np.clip(s["mean_b"] - max(s["mean_a"], 0.0), 0, 1)
    scores["sunset"] = float(0.4 * warm + 0.3 * hi_c + 0.3 * warm_bias)

    # Night / low light
    dark = np.clip((0.35 - s["mean_L"]) / 0.30, 0, 1)
    crush = np.clip((0.18 - s["p10_L"]) / 0.15, 0, 1)
    scores["night"] = float(0.6 * dark + 0.4 * crush)

    # Backlight — bright border, darker center
    scores["backlight"] = float(
        np.clip((s["border_L"] - s["center_L"] - 0.05) / 0.25, 0, 1)
    )

    # Portrait — skin-ish fraction (+ a little warmth)
    scores["portrait"] = float(
        np.clip((s["skin_frac"] - 0.04) / 0.20, 0, 1) * 0.85
        + 0.15 * np.clip(s["mean_a"] / 0.15, 0, 1)
    )

    # Indoor — mid brightness, modest chroma, not outdoor-vivid / not dark
    mid = 1.0 - abs(s["mean_L"] - 0.48) / 0.40
    scores["indoor"] = float(
        np.clip(0.55 * mid + 0.25 * (1 - hi_c) + 0.20 * (1 - dark) - 0.25 * scores["haze"], 0, 1)
    )

    # Vivid colourful (neutral / mixed hues — not specifically warm sunset)
    scores["vivid"] = float(
        np.clip((s["mean_c"] - 0.22) / 0.25, 0, 1) * (1.0 - 0.5 * warm_bias)
    )

    # Text / detail — high edge energy, not night
    scores["detail"] = float(np.clip((s["edge"] - 0.08) / 0.12, 0, 1) * (1 - 0.5 * dark))

    # Daylight — bright, decent contrast, moderate chroma
    day = np.clip((s["mean_L"] - 0.42) / 0.28, 0, 1) * np.clip(s["std_L"] / 0.18, 0, 1)
    scores["daylight"] = float(day * (1 - 0.6 * scores["haze"]) * (1 - 0.4 * dark))

    scores["default"] = 0.12

    scenario = max(scores, key=scores.get)

    # Priority overrides (course-demo reliability)
    if scores["night"] > 0.55:
        scenario = "night"
    elif scores["backlight"] > 0.55:
        scenario = "backlight"
    elif (
        scores["sunset"] > 0.42
        and s["mean_b"] > 0.08
        and s["mean_c"] > 0.12
        and s["mean_b"] >= s["mean_a"]  # warm/yellow, not green-dominant
        and s["mean_a"] > -0.05
    ):
        scenario = "sunset"
    elif (
        scores["haze"] > 0.55
        and s["mean_c"] < 0.14
        and s["std_L"] < 0.20
        and scores["sunset"] < 0.4
        and scores["backlight"] < 0.45
    ):
        scenario = "haze"
    elif scores["portrait"] > 0.45 and scores["vivid"] < 0.8:
        scenario = "portrait"
    elif scores["detail"] > 0.6 and scores["detail"] >= scores.get(scenario, 0) - 0.05:
        scenario = "detail"
    elif scores["daylight"] > 0.45 and scores["daylight"] >= scores.get(scenario, 0) - 0.05:
        scenario = "daylight"

    # Green-dominant outdoor should not be tagged sunset
    if scenario == "sunset" and s["mean_a"] < -0.08:
        scenario = "daylight" if s["mean_L"] > 0.4 else "vivid"

    return scenario, SCENARIO_TUNING[scenario], scores


def _percentile_stretch_L(lab: np.ndarray, amount: float, haze: float = 0.0) -> np.ndarray:
    out = lab.copy().astype(np.float64)
    L = out[..., 0]
    p_lo, p_hi = (4 - 2 * haze, 96 + 2 * haze)
    lo, hi = np.percentile(L, [p_lo, p_hi])
    if hi - lo < 1e-4:
        return out
    stretched = np.clip((L - lo) / (hi - lo), 0.0, 1.0)
    contrast = 1.15 + 0.35 * haze
    stretched = np.clip(0.5 + contrast * (stretched - 0.5), 0.0, 1.0)
    out[..., 0] = (1.0 - amount) * L + amount * stretched
    return out


def _clahe_L(lab: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return lab
    out = lab.copy().astype(np.float64)
    L8 = np.clip(out[..., 0] * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0 + 2.5 * amount, tileGridSize=(8, 8))
    L2 = clahe.apply(L8).astype(np.float64) / 255.0
    out[..., 0] = (1.0 - amount) * out[..., 0] + amount * L2
    return out


def _soft_dehaze_bgr(bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return bgr
    img = bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(np.min(img, axis=2), kernel)
    flat_dark = dark.reshape(-1)
    n_top = max(int(0.001 * flat_dark.size), 1)
    idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    A = np.clip(img.reshape(-1, 3)[idx].mean(axis=0), 0.6, 0.98)
    omega = 0.75 * amount
    t = 1.0 - omega * (dark / (np.max(A) + 1e-6))
    t = cv2.GaussianBlur(t.astype(np.float32), (0, 0), 3)
    t = np.clip(t, 0.25, 1.0)[..., None]
    J = np.clip((img - A) / t + A, 0.0, 1.0)
    out = (1.0 - amount) * img + amount * J
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _blend_colour(lab: np.ndarray, amount: float) -> np.ndarray:
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
    if not (MODELS / "tiny_cnn.json").exists() or amount <= 1e-6:
        return lab
    try:
        from .tiny_cnn import apply_tiny_cnn_lab
    except ImportError:
        return lab

    obj = _load("tiny_cnn.json")
    pred = apply_tiny_cnn_lab(lab.astype(np.float32), obj["payload"])
    return lab.astype(np.float64) * (1.0 - amount) + pred.astype(np.float64) * amount


def _unsharp_L(lab: np.ndarray, amount: float, radius: float = 1.2) -> np.ndarray:
    if amount <= 1e-6:
        return lab
    out = lab.copy().astype(np.float64)
    L = out[..., 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), radius)
    out[..., 0] = np.clip(L + amount * 1.4 * (L - blur), 0.0, 1.0)
    return out


def enhance_auto(
    bgr: np.ndarray,
    strength: float = 0.75,
    preset: str = "Balanced",
    return_info: bool = False,
):
    """Full Auto cascade → BGR uint8 (optionally with scenario info)."""
    cfg = dict(PRESETS.get(preset, PRESETS["Balanced"]))
    src_lab = bgr_to_lab01(bgr)
    scenario, tune, scores = detect_scenario(src_lab)

    s = float(np.clip(strength * tune["strength"], 0.2, 1.25))
    hz = scores.get("haze", 0.0)

    dehaze_amt = cfg["dehaze"] * s * tune["dehaze"] * (0.15 + 0.85 * max(hz, 0.35 if scenario == "haze" else 0.0))
    if scenario != "haze":
        dehaze_amt *= 0.5
    work_bgr = _soft_dehaze_bgr(bgr, float(np.clip(dehaze_amt, 0, 1)))

    lab = bgr_to_lab01(work_bgr).astype(np.float64)
    lab = _percentile_stretch_L(
        lab,
        cfg["exposure"] * s * tune["exposure"],
        haze=hz if scenario == "haze" else 0.15 * hz,
    )
    lab = _clahe_L(lab, float(np.clip(tune["clahe"] * s, 0, 1)))
    lab = _blend_colour(lab, cfg["colour"] * s * tune["colour"])
    lab = _apply_cnn(lab, cfg["cnn"] * s * tune["cnn"])
    lab = _unsharp_L(
        lab,
        cfg["clarity"] * s * tune["clarity"],
        radius=1.15 + (0.5 if scenario in {"haze", "detail", "backlight"} else 0.0),
    )

    lab = stabilize_lab(
        src_lab,
        lab,
        strength=1.0,
        max_dL=cfg["max_dL"] * tune["max_dL"],
        max_da=cfg["max_da"] * tune["max_da"],
        max_db=cfg["max_db"] * tune["max_db"],
    )
    out = lab01_to_bgr(lab)
    if return_info:
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        info = {
            "scenario": scenario,
            "label": tune["label"],
            "scores": {k: round(float(v), 3) for k, v in top},
        }
        return out, info
    return out


def list_presets() -> list[str]:
    return list(PRESETS.keys())


def list_scenarios() -> list[str]:
    return [SCENARIO_TUNING[k]["label"] for k in SCENARIO_TUNING]
