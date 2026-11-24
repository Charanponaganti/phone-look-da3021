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
from .pipeline import list_looks, list_presets

EXAMPLES_DIR = ROOT / "assets" / "examples"

STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Sora:wght@400;500;600&display=swap');

:root {
  --pl-bg0: #14110e;
  --pl-bg1: #1c1814;
  --pl-ink: #f3eee6;
  --pl-muted: #a89f93;
  --pl-line: rgba(243, 238, 230, 0.10);
  --pl-accent: #e0b35a;
  --pl-accent-ink: #1a140c;
}

.gradio-container {
  max-width: 1180px !important;
  font-family: "Sora", system-ui, sans-serif !important;
  color: var(--pl-ink) !important;
}

/* Atmospheric backdrop instead of flat black */
.gradio-container,
.main,
footer {
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(224, 179, 90, 0.16), transparent 55%),
    radial-gradient(700px 380px at 92% 8%, rgba(90, 140, 160, 0.10), transparent 50%),
    linear-gradient(180deg, #1a1612 0%, #12100e 55%, #0e0c0a 100%) !important;
}

.pl-hero {
  padding: 1.4rem 0 0.4rem;
  border-bottom: 1px solid var(--pl-line);
  margin-bottom: 1.1rem;
}

.pl-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--pl-accent);
  margin: 0 0 0.45rem;
  font-weight: 600;
}

.pl-title {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: clamp(2.1rem, 4.5vw, 3.1rem);
  font-weight: 700;
  line-height: 1.05;
  margin: 0 0 0.55rem;
  color: var(--pl-ink);
  letter-spacing: -0.02em;
}

.pl-sub {
  max-width: 42rem;
  margin: 0;
  color: var(--pl-muted);
  font-size: 0.98rem;
  line-height: 1.55;
}

.pl-status {
  margin-top: 0.65rem;
  padding: 0.7rem 0.85rem;
  border-left: 3px solid var(--pl-accent);
  background: rgba(224, 179, 90, 0.08);
  color: var(--pl-ink);
  font-size: 0.92rem;
}

button.primary,
.gr-button-primary {
  background: var(--pl-accent) !important;
  color: var(--pl-accent-ink) !important;
  border: none !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
}

button.primary:hover,
.gr-button-primary:hover {
  filter: brightness(1.06);
}

footer { opacity: 0.55; }
"""


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


def _gradio_major() -> int:
    import gradio as gr

    return int(gr.__version__.split(".", 1)[0])


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

    def run(photo, mode, look, model_name, preset, strength):
        if photo is None:
            return None, None, '<div class="pl-status">Drop a phone photo to begin.</div>'
        if not isinstance(photo, Image.Image):
            photo = Image.fromarray(photo)
        bgr = _resize_max(_pil_to_bgr(photo))
        info = {"label": "Single model", "scores": {}, "blurb": ""}
        if mode == "Auto":
            out, info = enhance_bgr(
                bgr,
                mode="auto",
                strength=float(strength),
                preset=preset,
                look=look or "Premium camera",
                return_info=True,
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
            info = {"label": f"Single model · {name}", "scores": {}, "blurb": ""}
        scores = info.get("scores") or {}
        score_txt = " · ".join(f"{k} {v}" for k, v in scores.items()) if scores else ""
        blurb = info.get("blurb") or ""
        status = f'<div class="pl-status"><strong>{info.get("label", "?")}</strong>'
        if blurb:
            status += f" — {blurb}"
        if score_txt:
            status += f"<br/><span style='color:#a89f93'>Scene cues: {score_txt}</span>"
        status += "</div>"
        return _bgr_to_pil(bgr), _bgr_to_pil(out), status

    def compare(photo, strength):
        if photo is None:
            return []
        if not isinstance(photo, Image.Image):
            photo = Image.fromarray(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=900)
        gallery = []
        for look_name in list_looks():
            out = enhance_bgr(
                bgr, mode="auto", strength=float(strength), look=look_name
            )
            gallery.append((_bgr_to_pil(out), look_name))
        return gallery

    examples = _example_paths()
    theme = _studio_theme()
    major = _gradio_major()
    blocks_kw: dict = {"title": "Phone Look Studio"}
    if major < 6:
        blocks_kw["theme"] = theme
        blocks_kw["css"] = STUDIO_CSS

    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(
            """
            <div class="pl-hero">
              <p class="pl-kicker">DA3021 · iQOO Neo 9 Pro → premium-camera look</p>
              <h1 class="pl-title">Phone Look</h1>
              <p class="pl-sub">
                Good cameras cost money. Learns colour &amp; tone from phone↔DSLR
                pairs (DPED) plus your Neo 9 Pro Photo→Night/HDR captures — so
                everyday shots get a cleaner, richer finish without new glass.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=280):
                photo = gr.Image(
                    type="pil",
                    label="Phone photo",
                    height=360,
                    sources=["upload", "clipboard"],
                )
                look = gr.Dropdown(
                    list_looks(),
                    value="Premium camera",
                    label="Camera look",
                )
                with gr.Row():
                    mode = gr.Radio(
                        ["Auto", "Single model"],
                        value="Auto",
                        label="Engine",
                        scale=2,
                    )
                    strength = gr.Slider(
                        0.0, 1.0, value=0.75, step=0.05, label="Strength", scale=3
                    )
                with gr.Row():
                    preset = gr.Dropdown(
                        list_presets(), value="Balanced", label="Base preset", scale=1
                    )
                    model_name = gr.Dropdown(
                        choices=model_names or ["(none)"],
                        value=fallback_model,
                        label="Single model",
                        scale=1,
                    )
                with gr.Row():
                    run_btn = gr.Button("Enhance", variant="primary", scale=2)
                    cmp_btn = gr.Button("Compare all looks", scale=1)
                scenario_md = gr.HTML(
                    '<div class="pl-status">Drop a phone photo to begin.</div>'
                )

            with gr.Column(scale=7, min_width=320):
                with gr.Row():
                    before = gr.Image(type="pil", label="Before", height=320)
                    after = gr.Image(type="pil", label="After", height=320)
                gallery = gr.Gallery(
                    label="Look strip",
                    columns=4,
                    height=240,
                    object_fit="contain",
                )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=photo,
                label="Try a DPED patch",
                examples_per_page=8,
            )

        outs = [before, after, scenario_md]
        inputs = [photo, mode, look, model_name, preset, strength]
        run_btn.click(run, inputs, outs)
        photo.change(run, inputs, outs)
        look.change(run, inputs, outs)
        cmp_btn.click(compare, [photo, strength], [gallery])

    return demo


def _studio_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Sora"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size="md",
    ).set(
        body_background_fill="#14110e",
        body_background_fill_dark="#14110e",
        block_background_fill="#1c1814",
        block_background_fill_dark="#1c1814",
        block_border_color="#3a342c",
        block_border_color_dark="#3a342c",
        block_label_text_color="#f3eee6",
        block_title_text_color="#f3eee6",
        body_text_color="#f3eee6",
        body_text_color_dark="#f3eee6",
        body_text_color_subdued="#a89f93",
        button_primary_background_fill="#e0b35a",
        button_primary_background_fill_hover="#ebc474",
        button_primary_text_color="#1a140c",
        button_secondary_background_fill="#2a241c",
        button_secondary_text_color="#f3eee6",
        border_color_primary="#3a342c",
        shadow_drop="none",
        shadow_drop_lg="none",
    )


def launch_kwargs(port: int) -> dict:
    import gradio as gr

    kw: dict = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
    }
    if _gradio_major() >= 6:
        kw["theme"] = _studio_theme()
        kw["css"] = STUDIO_CSS
    return kw


def main():
    import os

    demo = build_demo()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(**launch_kwargs(port))


if __name__ == "__main__":
    main()
