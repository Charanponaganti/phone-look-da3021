"""
Phone Look Studio — Gradio Blocks UI.

Run:
  python -m src.app
  python app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .apply import enhance_bgr, get_best_model_path
from .models import load_model
from .paths import MODELS, ROOT
from .pipeline import list_presets

EXAMPLES_DIR = ROOT / "assets" / "examples"


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


def _list_models() -> list[str]:
    return sorted(
        p.name
        for p in MODELS.glob("*.json")
        if p.name not in {"split.json", "eval_test.json"}
    )


def _resize_max(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _example_paths() -> list[str]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(str(p) for p in EXAMPLES_DIR.glob("*.png")) + sorted(
        str(p) for p in EXAMPLES_DIR.glob("*.jpg")
    )


def build_demo():
    import gradio as gr

    model_names = _list_models()
    try:
        fallback_model = get_best_model_path().name
    except FileNotFoundError:
        fallback_model = model_names[0] if model_names else None

    cache: dict[str, dict] = {}

    def _get_model(name: str):
        if name not in cache:
            cache[name] = load_model(MODELS / name)
        return cache[name]

    def run(photo, mode, model_name, preset, strength):
        if photo is None:
            return None, None, ""
        if not isinstance(photo, Image.Image):
            photo = Image.fromarray(photo)
        bgr = _resize_max(_pil_to_bgr(photo))
        info = {"label": "Single model", "scores": {}}
        if mode == "Auto":
            out, info = enhance_bgr(
                bgr, mode="auto", strength=float(strength), preset=preset, return_info=True
            )
        else:
            if not model_names:
                raise gr.Error("No trained models found.")
            name = model_name or fallback_model
            out = enhance_bgr(
                bgr,
                model_obj=_get_model(name),
                strength=float(strength),
                mode="model",
            )
            info = {"label": f"Single model · {name}", "scores": {}}
        scores = info.get("scores") or {}
        score_txt = ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else ""
        status = f"Detected: **{info.get('label', '?')}**"
        if score_txt:
            status += f"  \nTop scores: `{score_txt}`"
        return _bgr_to_pil(bgr), _bgr_to_pil(out), status

    def compare(photo, strength):
        if photo is None:
            return []
        if not isinstance(photo, Image.Image):
            photo = Image.fromarray(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=1000)
        gallery = []
        for preset in ("Natural", "Balanced", "Punchy"):
            out = enhance_bgr(bgr, mode="auto", strength=float(strength), preset=preset)
            gallery.append((_bgr_to_pil(out), preset))
        if fallback_model and (MODELS / fallback_model).exists():
            out = enhance_bgr(
                bgr,
                model_obj=_get_model(fallback_model),
                strength=float(strength),
                mode="model",
            )
            gallery.append((_bgr_to_pil(out), f"Single: {fallback_model}"))
        return gallery

    examples = _example_paths()

    with gr.Blocks(title="Phone Look Studio") as demo:
        gr.Markdown(
            """
# Phone Look Studio — DA3021
**Auto** detects the scene (fog, night, sunset, portrait, backlight, indoor, vivid, detail, daylight)
and tunes dehaze / exposure / colour / CNN / clarity. Or pick a single ladder model.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(type="pil", label="Upload phone photo")
                mode = gr.Radio(["Auto", "Single model"], value="Auto", label="Mode")
                preset = gr.Dropdown(list_presets(), value="Balanced", label="Auto preset")
                model_name = gr.Dropdown(
                    choices=model_names or ["(none)"],
                    value=fallback_model,
                    label="Single model",
                )
                strength = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Strength")
                run_btn = gr.Button("Enhance", variant="primary")
                cmp_btn = gr.Button("Compare presets")
                scenario_md = gr.Markdown("Detected: —")
            with gr.Column(scale=2):
                with gr.Row():
                    before = gr.Image(type="pil", label="Before")
                    after = gr.Image(type="pil", label="After")
                gallery = gr.Gallery(label="Compare strip", columns=4, height=280)

        if examples:
            gr.Examples(examples=examples, inputs=photo, label="Example DPED patches")

        outs = [before, after, scenario_md]
        run_btn.click(run, [photo, mode, model_name, preset, strength], outs)
        photo.change(run, [photo, mode, model_name, preset, strength], outs)
        cmp_btn.click(compare, [photo, strength], [gallery])

    return demo


def _studio_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
    ).set(
        body_background_fill="#1a1a1a",
        block_background_fill="#242424",
        block_border_color="#3a3a3a",
        body_text_color="#f2efe8",
        button_primary_background_fill="#d4a017",
        button_primary_text_color="#1a1a1a",
    )


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=_studio_theme())


if __name__ == "__main__":
    main()
