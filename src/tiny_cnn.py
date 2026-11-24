"""
Tiny residual CNN — DPED-style local enhancement that still runs on CPU.

v2: trains on full patches + multi-scale crops, slightly wider/deeper net,
chroma-weighted MSE + L-channel structural term.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .io_utils import bgr_to_lab01
from .paths import MODELS

DEVICE = torch.device("cpu")


class TinyResCNN(nn.Module):
    """Residual CNN: out = x + bounded f(x), Lab channels, same spatial size."""

    def __init__(self, channels: int = 32, depth: int = 6, res_scale: float = 0.25):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.res_scale = res_scale
        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * torch.tanh(self.body(x))


def _lab_to_nchw(lab: np.ndarray) -> np.ndarray:
    return np.transpose(lab, (2, 0, 1))[None, ...].astype(np.float32)


def sample_patches(
    scene_dirs: list[Path],
    patch: int = 64,
    n_per_scene: int = 40,
    seed: int = 0,
    include_full: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame patches (resized) + random multi-scale crops."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    crop_sizes = sorted({patch, max(32, patch // 2), min(100, patch + 16)})

    for d in scene_dirs:
        src = cv2.imread(str(d / "src.png"))
        tgt = cv2.imread(str(d / "tgt_warped.png"))
        mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if src is None or tgt is None or mask is None:
            continue
        s_lab = bgr_to_lab01(src)
        t_lab = bgr_to_lab01(tgt)
        h, w = mask.shape

        if include_full:
            s_full = cv2.resize(s_lab, (patch, patch), interpolation=cv2.INTER_AREA)
            t_full = cv2.resize(t_lab, (patch, patch), interpolation=cv2.INTER_AREA)
            xs.append(np.transpose(s_full, (2, 0, 1)))
            ys.append(np.transpose(t_full, (2, 0, 1)))

        for psz in crop_sizes:
            if h < psz or w < psz:
                continue
            got = 0
            attempts = 0
            while got < n_per_scene and attempts < n_per_scene * 25:
                attempts += 1
                y = int(rng.integers(0, h - psz + 1))
                x = int(rng.integers(0, w - psz + 1))
                if float(mask[y : y + psz, x : x + psz].mean()) < 0.85:
                    continue
                s_c = s_lab[y : y + psz, x : x + psz]
                t_c = t_lab[y : y + psz, x : x + psz]
                if psz != patch:
                    s_c = cv2.resize(s_c, (patch, patch), interpolation=cv2.INTER_AREA)
                    t_c = cv2.resize(t_c, (patch, patch), interpolation=cv2.INTER_AREA)
                xs.append(np.transpose(s_c, (2, 0, 1)))
                ys.append(np.transpose(t_c, (2, 0, 1)))
                got += 1

    if not xs:
        raise RuntimeError("No patches sampled for TinyResCNN")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def _ssim_l_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Cheap structural term on L channel only."""
    p = pred[:, 0:1]
    t = tgt[:, 0:1]
    mu_p = p.mean(dim=(2, 3), keepdim=True)
    mu_t = t.mean(dim=(2, 3), keepdim=True)
    sig_p = ((p - mu_p) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_t = ((t - mu_t) ** 2).mean(dim=(2, 3), keepdim=True)
    sig_pt = ((p - mu_p) * (t - mu_t)).mean(dim=(2, 3), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / ((mu_p**2 + mu_t**2 + c1) * (sig_p + sig_t + c2) + 1e-8)
    return 1.0 - ssim.mean()


def fit_tiny_cnn(
    scene_dirs: list[Path],
    out_weights: Path,
    channels: int = 32,
    depth: int = 6,
    patch: int = 64,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    n_per_scene: int = 24,
) -> dict:
    torch.manual_seed(seed)
    X, Y = sample_patches(
        scene_dirs, patch=patch, n_per_scene=n_per_scene, seed=seed, include_full=True
    )
    print(f"  TinyResCNN patches: {len(X)} @ {patch}x{patch}")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    res_scale = 0.25
    model = TinyResCNN(channels=channels, depth=depth, res_scale=res_scale).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    chan_w = torch.tensor([1.0, 0.4, 0.4], device=DEVICE).view(1, 3, 1, 1)

    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            err = (pred - yb) * chan_w
            mse = (err * err).mean()
            loss = mse + 0.15 * _ssim_l_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={total / max(n, 1):.5f}")

    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": channels,
            "depth": depth,
            "res_scale": res_scale,
            "version": 2,
        },
        out_weights,
    )
    _CNN_CACHE.clear()
    return {
        "weights": out_weights.name,
        "channels": channels,
        "depth": depth,
        "res_scale": res_scale,
        "patch": patch,
        "epochs": epochs,
        "n_patches": int(len(X)),
        "residual": True,
        "version": 2,
    }


_CNN_CACHE: dict[str, TinyResCNN] = {}


def load_tiny_cnn(payload: dict, models_dir: Path = MODELS) -> TinyResCNN:
    key = str(models_dir / payload["weights"])
    if key not in _CNN_CACHE:
        ckpt = torch.load(key, map_location=DEVICE, weights_only=True)
        net = TinyResCNN(
            channels=int(ckpt["channels"]),
            depth=int(ckpt["depth"]),
            res_scale=float(ckpt.get("res_scale", payload.get("res_scale", 0.25))),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        _CNN_CACHE[key] = net
    return _CNN_CACHE[key]


@torch.no_grad()
def apply_tiny_cnn_lab(lab: np.ndarray, payload: dict) -> np.ndarray:
    """Full-image forward (pad-preserving convs). Caps long side for CPU."""
    net = load_tiny_cnn(payload)
    h, w = lab.shape[:2]
    max_side = 768
    scale = min(1.0, max_side / max(h, w))
    work = lab
    if scale < 1.0:
        work = cv2.resize(lab, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(_lab_to_nchw(work)).to(DEVICE)
    y = net(x).cpu().numpy()[0]
    out = np.transpose(y, (1, 2, 0))
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)
