"""Colour/tone models: course transforms + small neural nets (chroma-safe)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelSpec:
    name: str
    kind: str
    params: dict


def fit_tone_curves(src: np.ndarray, tgt: np.ndarray, degree_L: int = 3) -> dict:
    """L: smooth poly. a/b: linear only (avoids magenta explosions on OOD photos)."""
    curves = {}
    ranges = {}
    for i, ch in enumerate(["L", "a", "b"]):
        x = src[:, i]
        y = tgt[:, i]
        deg = degree_L if ch == "L" else 1
        coef = np.polyfit(x, y, deg)
        curves[ch] = coef.tolist()
        # Percentile range for safe eval (ignore outliers)
        ranges[ch] = [float(np.percentile(x, 1)), float(np.percentile(x, 99))]
    return {"degree_L": degree_L, "degree_chroma": 1, "curves": curves, "ranges": ranges}


def apply_tone_curves(lab: np.ndarray, model: dict) -> np.ndarray:
    out = lab.copy().astype(np.float64)
    ranges = model.get("ranges", {})
    for i, ch in enumerate(["L", "a", "b"]):
        coef = np.asarray(model["curves"][ch], dtype=np.float64)
        x = lab[..., i].astype(np.float64)
        if ch in ranges:
            lo, hi = ranges[ch]
            x = np.clip(x, lo, hi)
        out[..., i] = np.polyval(coef, x)
    return out


def fit_color_matrix(src: np.ndarray, tgt: np.ndarray, ridge: float = 1e-2) -> dict:
    """tgt ≈ [src,1] @ M with light ridge (stable 4x3)."""
    n = src.shape[0]
    X = np.concatenate([src, np.ones((n, 1))], axis=1)
    xtx = X.T @ X + ridge * np.eye(4)
    # Don't regularize bias row as hard — mild is fine on all
    M, _, _, _ = np.linalg.lstsq(xtx, X.T @ tgt, rcond=None)
    return {"M": M.tolist(), "ridge": ridge}


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
    """Ridge on root-poly features (closed-form)."""
    X = root_poly_features(src, degree=degree)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std
    n, f = Xs.shape
    Xa = np.concatenate([Xs, np.ones((n, 1))], axis=1)
    xtx = Xa.T @ Xa
    xtx[:f, :f] += alpha * np.eye(f)
    W, _, _, _ = np.linalg.lstsq(xtx, Xa.T @ tgt, rcond=None)
    return {
        "degree": degree,
        "alpha": alpha,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": W[:f].T.tolist(),
        "intercept": W[f].tolist(),
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


def _mlp_forward(x: np.ndarray, coefs: list, intercepts: list, activation: str = "relu") -> np.ndarray:
    a = x
    last = len(coefs) - 1
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        a = a @ np.asarray(W, dtype=np.float64) + np.asarray(b, dtype=np.float64)
        if i < last:
            a = np.tanh(a) if activation == "tanh" else np.maximum(0.0, a)
    return a


def _standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-8
    return (x - mean) / scale, mean, scale


def _init_mlp(sizes: list[int], rng: np.random.Generator):
    coefs, intercepts = [], []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        coefs.append(w)
        intercepts.append(b)
    return coefs, intercepts


def _train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: tuple[int, ...],
    max_iter: int = 300,
    batch_size: int = 2048,
    lr: float = 1e-3,
    seed: int = 0,
    chroma_weight: float = 0.5,
) -> tuple[list, list, int]:
    """Adam MLP. Predicts residual (Y-X_lab) when Y is residual; chroma downweighted."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, Y.shape[1]]
    coefs, intercepts = _init_mlp(sizes, rng)

    m_w = [np.zeros_like(w) for w in coefs]
    v_w = [np.zeros_like(w) for w in coefs]
    m_b = [np.zeros_like(b) for b in intercepts]
    v_b = [np.zeros_like(b) for b in intercepts]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    wchan = np.asarray([1.0, chroma_weight, chroma_weight], dtype=np.float64)

    n = len(X)
    t = 0
    for _epoch in range(max_iter):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            t += 1
            idx = perm[start : start + batch_size]
            xb, yb = X[idx], Y[idx]

            acts = [xb]
            pre = []
            a = xb
            for i, (W, b) in enumerate(zip(coefs, intercepts)):
                z = a @ W + b
                pre.append(z)
                a = np.maximum(0.0, z) if i < len(coefs) - 1 else z
                acts.append(a)

            # Weighted residual MSE
            err = (acts[-1] - yb) * wchan
            delta = (2.0 / len(xb)) * err * wchan
            for i in range(len(coefs) - 1, -1, -1):
                a_prev = acts[i]
                gW = a_prev.T @ delta
                gb = delta.sum(axis=0)
                W_i = coefs[i]

                m_w[i] = beta1 * m_w[i] + (1 - beta1) * gW
                v_w[i] = beta2 * v_w[i] + (1 - beta2) * (gW * gW)
                m_b[i] = beta1 * m_b[i] + (1 - beta1) * gb
                v_b[i] = beta2 * v_b[i] + (1 - beta2) * (gb * gb)
                mw_hat = m_w[i] / (1 - beta1**t)
                vw_hat = v_w[i] / (1 - beta2**t)
                mb_hat = m_b[i] / (1 - beta1**t)
                vb_hat = v_b[i] / (1 - beta2**t)
                coefs[i] = coefs[i] - lr * mw_hat / (np.sqrt(vw_hat) + eps)
                intercepts[i] = intercepts[i] - lr * mb_hat / (np.sqrt(vb_hat) + eps)

                if i > 0:
                    delta = (delta @ W_i.T) * (pre[i - 1] > 0)

    return coefs, intercepts, max_iter


def fit_lab_mlp(
    src: np.ndarray,
    tgt: np.ndarray,
    hidden=(64, 32),
    max_iter: int = 300,
    seed: int = 0,
) -> dict:
    """MLP predicts residual Lab (tgt-src); safer OOD behaviour."""
    residual = tgt - src
    Xs, x_mean, x_scale = _standardize_fit(src)
    Ys, y_mean, y_scale = _standardize_fit(residual)
    coefs, intercepts, n_iter = _train_mlp(
        Xs, Ys, hidden=tuple(hidden), max_iter=max_iter, seed=seed, chroma_weight=0.4
    )
    return {
        "mode": "residual",
        "hidden": list(hidden),
        "activation": "relu",
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "coefs": [c.tolist() for c in coefs],
        "intercepts": [b.tolist() for b in intercepts],
        "n_iter": int(n_iter),
    }


def apply_lab_mlp(lab: np.ndarray, model: dict) -> np.ndarray:
    flat = lab.reshape(-1, 3).astype(np.float64)
    x_mean = np.asarray(model["x_mean"])
    x_scale = np.asarray(model["x_scale"]) + 1e-12
    y_mean = np.asarray(model["y_mean"])
    y_scale = np.asarray(model["y_scale"]) + 1e-12
    Xs = (flat - x_mean) / x_scale
    # Clamp to ~3σ of training domain before MLP
    Xs = np.clip(Xs, -3.0, 3.0)
    Ys = _mlp_forward(Xs, model["coefs"], model["intercepts"], model.get("activation", "relu"))
    pred = Ys * y_scale + y_mean
    if model.get("mode") == "residual":
        out = flat + pred
    else:
        out = pred
    return out.reshape(lab.shape)


def fit_root_poly_mlp(
    src: np.ndarray,
    tgt: np.ndarray,
    degree: int = 2,
    hidden=(64, 32),
    max_iter: int = 300,
    seed: int = 0,
) -> dict:
    """Root-poly features → residual MLP."""
    X = root_poly_features(src, degree=degree)
    residual = tgt - src
    Xs, x_mean, x_scale = _standardize_fit(X)
    Ys, y_mean, y_scale = _standardize_fit(residual)
    coefs, intercepts, n_iter = _train_mlp(
        Xs, Ys, hidden=tuple(hidden), max_iter=max_iter, seed=seed, chroma_weight=0.4
    )
    return {
        "mode": "residual",
        "degree": degree,
        "hidden": list(hidden),
        "activation": "relu",
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "coefs": [c.tolist() for c in coefs],
        "intercepts": [b.tolist() for b in intercepts],
        "n_iter": int(n_iter),
    }


def apply_root_poly_mlp(lab: np.ndarray, model: dict) -> np.ndarray:
    flat = lab.reshape(-1, 3).astype(np.float64)
    X = root_poly_features(flat, degree=int(model["degree"]))
    x_mean = np.asarray(model["x_mean"])
    x_scale = np.asarray(model["x_scale"]) + 1e-12
    y_mean = np.asarray(model["y_mean"])
    y_scale = np.asarray(model["y_scale"]) + 1e-12
    Xs = np.clip((X - x_mean) / x_scale, -3.0, 3.0)
    Ys = _mlp_forward(Xs, model["coefs"], model["intercepts"], model.get("activation", "relu"))
    pred = Ys * y_scale + y_mean
    if model.get("mode") == "residual":
        out = flat + pred
    else:
        out = pred
    return out.reshape(lab.shape)


def stabilize_lab(
    src: np.ndarray,
    pred: np.ndarray,
    strength: float = 1.0,
    max_dL: float = 0.28,
    max_da: float = 0.18,
    max_db: float = 0.18,
) -> np.ndarray:
    """Blend + clamp channel residuals so OOD photos don't go neon."""
    strength = float(np.clip(strength, 0.0, 1.0))
    d = pred.astype(np.float64) - src.astype(np.float64)
    d[..., 0] = np.clip(d[..., 0], -max_dL, max_dL)
    d[..., 1] = np.clip(d[..., 1], -max_da, max_da)
    d[..., 2] = np.clip(d[..., 2], -max_db, max_db)
    out = src.astype(np.float64) + strength * d
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1], -1.0, 1.0)
    out[..., 2] = np.clip(out[..., 2], -1.0, 1.0)
    return out


def save_model(path: Path, kind: str, payload: dict, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"kind": kind, "payload": payload, "meta": meta or {}}
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_model_lab(lab: np.ndarray, model_obj: dict, strength: float = 1.0) -> np.ndarray:
    kind = model_obj["kind"]
    payload = model_obj["payload"]
    if kind == "tone_curves":
        out = apply_tone_curves(lab, payload)
    elif kind == "color_matrix":
        out = apply_color_matrix(lab, payload)
    elif kind == "root_poly":
        out = apply_root_poly(lab, payload)
    elif kind == "lab_mlp":
        out = apply_lab_mlp(lab, payload)
    elif kind == "root_poly_mlp":
        out = apply_root_poly_mlp(lab, payload)
    elif kind == "tiny_cnn":
        from .tiny_cnn import apply_tiny_cnn_lab

        out = apply_tiny_cnn_lab(lab, payload)
    else:
        raise ValueError(kind)
    return stabilize_lab(lab, out, strength=strength)
