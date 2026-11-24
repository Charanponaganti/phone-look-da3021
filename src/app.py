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
from .io_utils import bgr_to_lab01
from .models import load_model
from .paths import MODELS, ROOT
from .pipeline import list_looks, list_presets

EXAMPLES_DIR = ROOT / "assets" / "examples"

STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
  --pl-bg: #0f0e0c;
  --pl-panel: #1a1714;
  --pl-ink: #f7f2ea;
  --pl-muted: #d2c8b8;
  --pl-line: rgba(247, 242, 234, 0.16);
  --pl-accent: #f0c14d;
  --pl-accent-ink: #1a1408;
}

html, body, .gradio-container, .main, footer, .wrap {
  background: var(--pl-bg) !important;
  color: var(--pl-ink) !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
  font-size: 17px !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  padding: 1.25rem 1.75rem 2.5rem !important;
  margin: 0 !important;
}

/* Kill Gradio's narrow centered column */
.gradio-container .main,
.gradio-container > .main,
div.main {
  max-width: 100% !important;
  width: 100% !important;
}

/* Labels: large + high contrast */
.block .label-wrap,
.block .label-wrap span,
.block-label,
.block > label,
label,
span[data-testid="block-info"],
.svelte-1gfkn6j {
  color: var(--pl-ink) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  opacity: 1 !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
}

.block {
  background: var(--pl-panel) !important;
  border: 1px solid var(--pl-line) !important;
  border-radius: 14px !important;
  padding-top: 0.35rem !important;
}

/* Inputs / numbers readable */
input, textarea, select,
input[type="number"],
input[type="text"],
.secondary-wrap input {
  color: var(--pl-ink) !important;
  background: #0c0b09 !important;
  border: 1px solid #5a5248 !important;
  border-radius: 10px !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--pl-ink) !important;
  min-height: 2.4rem !important;
}

.slider input[type="number"] {
  width: 4rem !important;
  min-width: 4rem !important;
  font-size: 1.1rem !important;
}

/* Radio options */
[role="radiogroup"] label,
fieldset label,
.wrap label span {
  color: var(--pl-ink) !important;
  background: transparent !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
}

[role="radiogroup"] {
  gap: 0.5rem 1.1rem !important;
  flex-wrap: wrap !important;
}

button {
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  min-height: 2.75rem !important;
  border-radius: 10px !important;
}

button.primary, .gr-button-primary {
  background: var(--pl-accent) !important;
  color: var(--pl-accent-ink) !important;
  border: none !important;
}

button.secondary, .gr-button-secondary {
  background: #2b261f !important;
  color: var(--pl-ink) !important;
  border: 1px solid #5a5248 !important;
}

.pl-hero {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 0.75rem 1.5rem;
  padding: 0.25rem 0 1rem;
  margin-bottom: 0.85rem;
  border-bottom: 1px solid var(--pl-line);
}

.pl-hero-copy { max-width: 52rem; }

.pl-kicker {
  margin: 0 0 0.35rem;
  font-size: 0.8rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--pl-accent);
  font-weight: 700;
}

.pl-title {
  margin: 0 0 0.4rem;
  font-family: "Fraunces", Georgia, serif !important;
  font-size: clamp(2rem, 4vw, 2.85rem);
  font-weight: 700;
  line-height: 1.05;
  color: var(--pl-ink);
  letter-spacing: -0.02em;
}

.pl-sub {
  margin: 0;
  color: var(--pl-muted);
  font-size: 1.08rem;
  line-height: 1.45;
  font-weight: 500;
}

.pl-status {
  margin-top: 0.65rem;
  padding: 0.95rem 1.1rem;
  border-radius: 12px;
  border: 1px solid rgba(240, 193, 77, 0.35);
  background: #241f18;
  color: var(--pl-ink);
  font-size: 1.05rem;
  line-height: 1.5;
  font-weight: 500;
}

.pl-status strong {
  color: var(--pl-accent);
  font-size: 1.12rem;
}

.pl-status .pl-delta {
  display: block;
  margin-top: 0.35rem;
  color: #efe6d6;
  font-size: 1.02rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}

.pl-status .pl-cues {
  display: block;
  margin-top: 0.25rem;
  color: #cfc4b4;
  font-size: 0.95rem;
}

/* Bigger image stage */
.pl-stage .image-container,
.pl-stage img,
.pl-slider img {
  max-height: min(68vh, 720px) !important;
}

footer { display: none !important; }

@media (max-width: 900px) {
  .gradio-container { padding: 0.85rem !important; }
  .pl-title { font-size: 1.75rem; }
}
"""


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


def _as_pil(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def _list_models() -> list[str]:
    return sorted(
        p.name
        for p in MODELS.glob("*.json")
        if p.name not in {"split.json", "eval_test.json"}
    )


def _resize_max(bgr: np.ndarray, max_side: int = 1100) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _example_paths() -> list[str]:
    if not EXAMPLES_DIR.exists():
        return []
    paths = sorted(EXAMPLES_DIR.glob("*.png")) + sorted(EXAMPLES_DIR.glob("*.jpg"))
    rank = {"haze": 0, "night": 1, "low": 2, "fog": 3, "dark": 4}

    def key(p: Path):
        name = p.stem.lower()
        for token, r in rank.items():
            if token in name:
                return (r, name)
        return (9, name)

    return [str(p) for p in sorted(paths, key=key)]


def _gradio_major() -> int:
    import gradio as gr

    return int(gr.__version__.split(".", 1)[0])


def _has_image_slider() -> bool:
    import gradio as gr

    return hasattr(gr, "ImageSlider")


def _lab_delta(before_bgr: np.ndarray, after_bgr: np.ndarray) -> dict[str, float]:
    a = bgr_to_lab01(before_bgr)
    b = bgr_to_lab01(after_bgr)
    dL = float(np.mean(np.abs(b[..., 0] - a[..., 0])))
    da = float(np.mean(np.abs(b[..., 1] - a[..., 1])))
    db = float(np.mean(np.abs(b[..., 2] - a[..., 2])))
    chroma = float(np.mean(np.sqrt(da * da + db * db)))
    return {
        "tone_pct": 100.0 * dL,
        "colour_pct": 100.0 * chroma,
        "mean_dE": float(np.mean(np.sqrt(((b - a) ** 2).sum(-1)))),
    }


def _diff_heatmap(before_bgr: np.ndarray, after_bgr: np.ndarray, gain: float = 5.0) -> np.ndarray:
    diff = np.abs(after_bgr.astype(np.float32) - before_bgr.astype(np.float32))
    mag = np.clip(diff.max(axis=2) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def _empty_status(msg: str = "Upload a phone photo — result appears automatically.") -> str:
    return f'<div class="pl-status">{msg}</div>'


def _status_html(info: dict, delta: dict | None = None) -> str:
    scores = info.get("scores") or {}
    score_txt = " · ".join(f"{k} {v}" for k, v in scores.items()) if scores else ""
    blurb = info.get("blurb") or ""
    html = f'<div class="pl-status"><strong>{info.get("label", "?")}</strong>'
    if blurb:
        html += f" — {blurb}"
    if delta:
        html += (
            f'<span class="pl-delta">Δ tone {delta["tone_pct"]:.1f}%'
            f' · colour {delta["colour_pct"]:.1f}%'
            f' · mean ΔE {delta["mean_dE"]:.3f}</span>'
        )
    if score_txt:
        html += f'<span class="pl-cues">Scene cues: {score_txt}</span>'
    html += "</div>"
    return html


def build_demo():
    import gradio as gr

    model_names = _list_models()
    try:
        fallback_model = get_best_model_path().name
    except FileNotFoundError:
        fallback_model = model_names[0] if model_names else None

    cache: dict[str, dict] = {}
    use_slider = _has_image_slider()

    def _get_model(name: str):
        if name not in cache:
            cache[name] = load_model(MODELS / name)
        return cache[name]

    def run(photo, mode, look, model_name, preset, strength, view_mode):
        view_mode = view_mode or ("Slider" if use_slider else "Side-by-side")
        show_slider = use_slider and view_mode == "Slider"
        show_side = view_mode == "Side-by-side"
        show_diff = view_mode == "Difference"

        if photo is None:
            return (
                gr.update(value=None, visible=show_slider),
                None,
                None,
                gr.update(value=None, visible=show_diff),
                _empty_status(),
                None,
                None,
                gr.update(visible=show_side),
            )

        photo = _as_pil(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=1000)

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

        before_pil = _bgr_to_pil(bgr)
        after_pil = _bgr_to_pil(out)
        delta = _lab_delta(bgr, out)
        heat = _bgr_to_pil(_diff_heatmap(bgr, out))
        status = _status_html(info, delta)
        slider_val = (before_pil, after_pil) if use_slider else None

        return (
            gr.update(value=slider_val, visible=show_slider),
            before_pil,
            after_pil,
            gr.update(value=heat, visible=show_diff),
            status,
            before_pil,
            after_pil,
            gr.update(visible=show_side),
        )

    def switch_view(view_mode, before_pil, after_pil):
        view_mode = view_mode or ("Slider" if use_slider else "Side-by-side")
        before_pil = _as_pil(before_pil)
        after_pil = _as_pil(after_pil)
        heat = None
        if (
            view_mode == "Difference"
            and before_pil is not None
            and after_pil is not None
        ):
            heat = _bgr_to_pil(
                _diff_heatmap(_pil_to_bgr(before_pil), _pil_to_bgr(after_pil))
            )
        return (
            gr.update(visible=use_slider and view_mode == "Slider"),
            gr.update(visible=view_mode == "Side-by-side"),
            gr.update(visible=view_mode == "Difference", value=heat),
        )

    def compare(photo, strength):
        if photo is None:
            return []
        photo = _as_pil(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=720)
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
    blocks_kw: dict = {
        "title": "Phone Look Studio",
        "fill_height": False,
        "fill_width": True,
    }
    if major < 6:
        blocks_kw["theme"] = theme
        blocks_kw["css"] = STUDIO_CSS

    looks = list_looks()
    default_view = "Slider" if use_slider else "Side-by-side"

    with gr.Blocks(**blocks_kw) as demo:
        st_before = gr.State(None)
        st_after = gr.State(None)

        gr.HTML(
            """
            <div class="pl-hero">
              <div class="pl-hero-copy">
                <p class="pl-kicker">DA3021 · iQOO Neo 9 Pro → premium-camera look</p>
                <h1 class="pl-title">Phone Look Studio</h1>
                <p class="pl-sub">Upload a phone photo. Drag the seam to compare before and after.</p>
              </div>
            </div>
            """
        )

        # Controls strip — full width, readable
        with gr.Row(equal_height=True):
            photo = gr.Image(
                type="pil",
                label="1 · Phone photo",
                height=180,
                sources=["upload", "clipboard"],
                scale=3,
            )
            with gr.Column(scale=4):
                look = gr.Dropdown(
                    looks,
                    value="Premium camera",
                    label="2 · Camera look",
                )
                strength = gr.Slider(
                    0.0, 1.0, value=0.9, step=0.05, label="3 · Strength"
                )
                with gr.Row():
                    run_btn = gr.Button("Re-run", variant="secondary", scale=1)
                    cmp_btn = gr.Button("Compare all looks", variant="primary", scale=2)
            with gr.Column(scale=3):
                view_mode = gr.Radio(
                    ["Slider", "Side-by-side", "Difference"]
                    if use_slider
                    else ["Side-by-side", "Difference"],
                    value=default_view,
                    label="Compare view",
                )
                with gr.Accordion("Advanced", open=False):
                    mode = gr.Radio(
                        ["Auto", "Single model"],
                        value="Auto",
                        label="Engine",
                    )
                    preset = gr.Dropdown(
                        list_presets(), value="Balanced", label="Base preset"
                    )
                    model_name = gr.Dropdown(
                        choices=model_names or ["(none)"],
                        value=fallback_model,
                        label="Single model",
                    )

        # Dominant stage
        with gr.Column(elem_classes=["pl-stage"]):
            slider = gr.ImageSlider(
                label="Before → After · drag the seam",
                type="pil",
                height=560,
                max_height=680,
                slider_position=45,
                interactive=False,
                elem_classes=["pl-slider"],
                visible=use_slider,
            )
            with gr.Row(visible=not use_slider) as side_row:
                before = gr.Image(type="pil", label="Before", height=480, scale=1)
                after = gr.Image(type="pil", label="After", height=480, scale=1)
            diff_img = gr.Image(
                type="pil",
                label="Difference map · brighter = larger edit",
                height=520,
                visible=False,
            )
            scenario_md = gr.HTML(_empty_status())

        with gr.Accordion("Look strip (all camera looks)", open=False):
            gallery = gr.Gallery(
                label="Look strip",
                columns=3,
                height=240,
                object_fit="contain",
            )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=photo,
                label="Try an example",
                examples_per_page=8,
            )

        run_outs = [
            slider,
            before,
            after,
            diff_img,
            scenario_md,
            st_before,
            st_after,
            side_row,
        ]
        inputs = [photo, mode, look, model_name, preset, strength, view_mode]
        run_btn.click(run, inputs, run_outs)
        photo.change(run, inputs, run_outs)
        look.change(run, inputs, run_outs)
        strength.release(run, inputs, run_outs)
        view_mode.change(
            switch_view,
            [view_mode, st_before, st_after],
            [slider, side_row, diff_img],
        )
        cmp_btn.click(compare, [photo, strength], [gallery])

    return demo


def _studio_theme():
    import gradio as gr

    return gr.themes.Default(
        primary_hue="yellow",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size="lg",
    ).set(
        body_background_fill="#0f0e0c",
        body_background_fill_dark="#0f0e0c",
        body_text_color="#f7f2ea",
        body_text_color_dark="#f7f2ea",
        body_text_color_subdued="#d2c8b8",
        block_background_fill="#1a1714",
        block_background_fill_dark="#1a1714",
        block_border_color="#3f382f",
        block_border_color_dark="#3f382f",
        block_label_background_fill="#1a1714",
        block_label_background_fill_dark="#1a1714",
        block_label_text_color="#f7f2ea",
        block_label_text_color_dark="#f7f2ea",
        block_title_text_color="#f7f2ea",
        block_title_text_color_dark="#f7f2ea",
        input_background_fill="#0c0b09",
        input_background_fill_dark="#0c0b09",
        input_border_color="#5a5248",
        input_border_color_dark="#5a5248",
        input_placeholder_color="#9a9082",
        input_placeholder_color_dark="#9a9082",
        button_primary_background_fill="#f0c14d",
        button_primary_background_fill_hover="#f5cf6e",
        button_primary_text_color="#1a1408",
        button_secondary_background_fill="#2b261f",
        button_secondary_background_fill_hover="#3a3329",
        button_secondary_text_color="#f7f2ea",
        border_color_primary="#3f382f",
        color_accent="#f0c14d",
        shadow_drop="none",
        shadow_drop_lg="none",
    )


def launch_kwargs(port: int) -> dict:
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
