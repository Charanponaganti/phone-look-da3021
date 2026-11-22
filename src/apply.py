"""Apply a trained model to any photo."""

from __future__ import annotations

from pathlib import Path

import cv2

from .io_utils import bgr_to_lab01, lab01_to_bgr, read_bgr, write_bgr
from .models import apply_model_lab, load_model
from .paths import ENHANCED, MODELS


def get_best_model_path() -> Path:
    marker = MODELS / "best_model.txt"
    if marker.exists():
        name = marker.read_text(encoding="utf-8").strip()
        path = MODELS / name
        if path.exists():
            return path
    # fallback
    cands = sorted(MODELS.glob("root_poly*.json"))
    if cands:
        return cands[0]
    tone = MODELS / "tone_curves.json"
    if tone.exists():
        return tone
    raise FileNotFoundError("No trained model found. Run fit_models + evaluate first.")


def enhance_bgr(img_bgr, model_obj=None):
    if model_obj is None:
        model_obj = load_model(get_best_model_path())
    lab = bgr_to_lab01(img_bgr)
    out_lab = apply_model_lab(lab, model_obj)
    return lab01_to_bgr(out_lab)


def enhance_file(in_path: Path, out_path: Path | None = None, model_path: Path | None = None) -> Path:
    img = read_bgr(in_path, max_side=2400)
    model_obj = load_model(model_path) if model_path else load_model(get_best_model_path())
    out = enhance_bgr(img, model_obj)
    if out_path is None:
        out_path = ENHANCED / f"{in_path.stem}_enhanced.png"
    write_bgr(out_path, out)
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="Enhance a photo with the trained look model")
    p.add_argument("image", type=Path, help="Input image path")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("-m", "--model", type=Path, default=None)
    args = p.parse_args()
    out = enhance_file(args.image, args.out, args.model)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
