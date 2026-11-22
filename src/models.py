"""Colour/tone models: tone curves, 3x3 matrix, root-polynomial + ridge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelSpec:
    name: str
    kind: str
    params: dict


def fit_tone_curves(src: np.ndarray, tgt: np.ndarray, degree: int = 3) -> dict:
    """Independent polynomial per Lab channel: tgt_c ≈ poly(src_c)."""
    curves = {}
    for i, ch in enumerate(["L", "a", "b"]):
        x = src[:, i]
        y = tgt[:, i]
        coef = np.polyfit(x, y, degree)
        curves[ch] = coef.tolist()
    return {"degree": degree, "curves": curves}


def apply_tone_curves(lab: np.ndarray, model: dict) -> np.ndarray:
    out = lab.copy()
    for i, ch in enumerate(["L", "a", "b"]):
        coef = np.asarray(model["curves"][ch], dtype=np.float64)
        out[..., i] = np.polyval(coef, lab[..., i])
    return out


def fit_color_matrix(src: np.ndarray, tgt: np.ndarray) -> dict:
    """tgt ≈ src @ M  (3x3), least squares with bias via homogeneous coords."""
    n = src.shape[0]
    X = np.concatenate([src, np.ones((n, 1))], axis=1)  # Nx4
    # Solve for 4x3 mapping
    M, _, _, _ = np.linalg.lstsq(X, tgt, rcond=None)
    return {"M": M.tolist()}  # 4x3


def apply_color_matrix(lab: np.ndarray, model: dict) -> np.ndarray:
    M = np.asarray(model["M"], dtype=np.float64)
    flat = lab.reshape(-1, 3)
    X = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    out = X @ M
    return out.reshape(lab.shape)


def root_poly_features(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Root-polynomial style features on Lab (Finlayson-inspired, simplified)."""
    L, a, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    feats = [L, a, b, np.sqrt(np.clip(L, 0, None)), np.sqrt(np.abs(a)), np.sqrt(np.abs(b))]
    if degree >= 2:
        feats += [L * a, L * b, a * b, L * L, a * a, b * b]
    if degree >= 3:
        feats += [L * a * b, L * L * a, L * L * b]
    return np.concatenate(feats, axis=1)


def fit_root_poly(src: np.ndarray, tgt: np.ndarray, degree: int = 2, alpha: float = 1.0) -> dict:
    from sklearn.linear_model import Ridge

    X = root_poly_features(src, degree=degree)
    # Scale columns for conditioning (course idea)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xs, tgt)
    return {
        "degree": degree,
        "alpha": alpha,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": model.coef_.tolist(),  # 3 x F
        "intercept": model.intercept_.tolist(),
    }


def apply_root_poly(lab: np.ndarray, model: dict) -> np.ndarray:
    flat = lab.reshape(-1, 3)
    X = root_poly_features(flat, degree=int(model["degree"]))
    mean = np.asarray(model["mean"])
    std = np.asarray(model["std"])
    Xs = (X - mean) / std
    coef = np.asarray(model["coef"])
    intercept = np.asarray(model["intercept"])
    out = Xs @ coef.T + intercept
    return out.reshape(lab.shape)


def save_model(path: Path, kind: str, payload: dict, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"kind": kind, "payload": payload, "meta": meta or {}}
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_model_lab(lab: np.ndarray, model_obj: dict) -> np.ndarray:
    kind = model_obj["kind"]
    payload = model_obj["payload"]
    if kind == "tone_curves":
        out = apply_tone_curves(lab, payload)
    elif kind == "color_matrix":
        out = apply_color_matrix(lab, payload)
    elif kind == "root_poly":
        out = apply_root_poly(lab, payload)
    else:
        raise ValueError(kind)
    # Keep L in [0,1], a/b roughly in [-1,1]
    out = out.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], -1.0, 1.0)
    out[..., 2] = np.clip(out[..., 2], -1.0, 1.0)
    return out
