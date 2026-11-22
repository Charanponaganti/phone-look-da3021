"""
Gradio upload app: drop a photo → get the learned look.

Run after training (or with shipped models):
  python -m src.app
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .apply import enhance_bgr, get_best_model_path
from .models import load_model
from .paths import MODELS


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def _list_models() -> list[str]:
    names = sorted(p.name for p in MODELS.glob("*.json") if p.name != "split.json" and p.name != "eval_test.json")
    return names


def enhance_pil(img: Image.Image, model_obj, max_side: int = 1600) -> Image.Image:
    bgr = _pil_to_bgr(img)
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = enhance_bgr(bgr, model_obj)
    return _bgr_to_pil(out)


def main():
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit("Install gradio first: pip install gradio") from e

    model_names = _list_models()
    try:
        default_name = get_best_model_path().name
    except FileNotFoundError:
        default_name = model_names[0] if model_names else None

    cache: dict[str, dict] = {}

    def _get_model(name: str):
        if name not in cache:
            path = MODELS / name
            if not path.exists():
                raise FileNotFoundError(name)
            cache[name] = load_model(path)
        return cache[name]

    def run(photo, model_name: str):
        if photo is None:
            return None
        if not model_names:
            raise gr.Error("No trained models in outputs/models/. Run fit_models + evaluate first.")
        name = model_name or default_name
        if not isinstance(photo, Image.Image):
            photo = Image.fromarray(photo)
        return enhance_pil(photo, _get_model(name))

    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Image(type="pil", label="Upload phone photo"),
            gr.Dropdown(
                choices=model_names or ["(none)"],
                value=default_name,
                label="Model",
            ),
        ],
        outputs=gr.Image(type="pil", label="Enhanced look"),
        title="Phone Look — DA3021",
        description=(
            "CPU colour/tone transform learned from paired iPhone→Canon (DPED) patches. "
            f"Default model: **{default_name or 'none'}**. "
            "Upload any phone photo to try the look."
        ),
        examples=None,
        allow_flagging="never",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
