"""Apply a trained model or the Auto cascade to any photo."""

from __future__ import annotations

from pathlib import Path

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
    for name in ("color_matrix.json", "tone_curves.json", "tiny_cnn.json"):
        path = MODELS / name
        if path.exists():
            return path
    cands = sorted(MODELS.glob("*.json"))
    cands = [p for p in cands if p.name not in {"split.json", "eval_test.json"}]
    if cands:
        return cands[0]
    raise FileNotFoundError("No trained model found. Run fit_models + evaluate first.")


def enhance_bgr(
    img_bgr,
    model_obj=None,
    strength: float = 0.75,
    mode: str = "auto",
    preset: str = "Balanced",
    return_info: bool = False,
):
    """
    mode='auto'  → cascade (default, best for demos)
    mode='model' → single trained JSON model
    """
    if mode == "auto":
        from .pipeline import enhance_auto

        return enhance_auto(img_bgr, strength=strength, preset=preset, return_info=return_info)

    if model_obj is None:
        model_obj = load_model(get_best_model_path())
    lab = bgr_to_lab01(img_bgr)
    out_lab = apply_model_lab(lab, model_obj, strength=strength)
    out = lab01_to_bgr(out_lab)
    if return_info:
        return out, {"scenario": "model", "label": "Single model", "scores": {}}
    return out


def enhance_file(
    in_path: Path,
    out_path: Path | None = None,
    model_path: Path | None = None,
    strength: float = 0.75,
    mode: str = "auto",
    preset: str = "Balanced",
) -> Path:
    img = read_bgr(in_path, max_side=2400)
    model_obj = None
    if mode == "model":
        model_obj = load_model(model_path) if model_path else load_model(get_best_model_path())
    out = enhance_bgr(img, model_obj, strength=strength, mode=mode, preset=preset)
    if out_path is None:
        out_path = ENHANCED / f"{in_path.stem}_enhanced.png"
    write_bgr(out_path, out)
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="Enhance a photo with Auto or a trained model")
    p.add_argument("image", type=Path, help="Input image path")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("-m", "--model", type=Path, default=None)
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--mode", choices=["auto", "model"], default="auto")
    p.add_argument("--preset", default="Balanced", choices=["Natural", "Balanced", "Punchy"])
    args = p.parse_args()
    out = enhance_file(
        args.image,
        args.out,
        args.model,
        strength=args.strength,
        mode=args.mode,
        preset=args.preset,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
