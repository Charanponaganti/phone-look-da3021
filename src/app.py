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

# Aggressive Gradio 6 overrides: Soft+amber paints labels yellow on cream text.
STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Sora:wght@400;500;600&display=swap');

:root {
  --pl-bg0: #14110e;
  --pl-bg1: #1c1814;
  --pl-panel: #221e19;
  --pl-ink: #f3eee6;
  --pl-muted: #b7aea2;
  --pl-line: rgba(243, 238, 230, 0.14);
  --pl-accent: #e0b35a;
  --pl-accent-ink: #1a140c;
}

html, body {
  background: #12100e !important;
}

.gradio-container {
  max-width: min(96vw, 1280px) !important;
  margin: 0 auto !important;
  font-family: "Sora", system-ui, sans-serif !important;
  color: var(--pl-ink) !important;
  padding-bottom: 2rem !important;
}

.gradio-container,
.main,
footer,
.wrap {
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(224, 179, 90, 0.14), transparent 55%),
    radial-gradient(700px 380px at 92% 8%, rgba(90, 140, 160, 0.08), transparent 50%),
    linear-gradient(180deg, #1a1612 0%, #12100e 55%, #0e0c0a 100%) !important;
}

/* ---- Readable labels / inputs (Gradio 6 Soft theme fix) ---- */
.block .label-wrap,
.block .label-wrap span,
span.svelte-1gfkn6j,
label.svelte-1b6s6s,
.block-label,
.block > label,
.form label,
.container > label {
  color: var(--pl-ink) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  opacity: 1 !important;
}

.block {
  background: var(--pl-panel) !important;
  border: 1px solid var(--pl-line) !important;
  border-radius: 12px !important;
}

input[type="number"],
input[type="text"],
textarea,
select,
.secondary-wrap input,
input.svelte-1f354aw,
.number-input input {
  color: var(--pl-ink) !important;
  background: #12100e !important;
  border: 1px solid #4a4339 !important;
  border-radius: 8px !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--pl-ink) !important;
}

.slider input[type="number"] {
  min-width: 3.2rem !important;
  color: var(--pl-ink) !important;
  background: #12100e !important;
}

/* Radio option text readable */
.wrap label span,
fieldset label,
.radio-row label,
[role="radiogroup"] label {
  color: var(--pl-ink) !important;
  background: transparent !important;
}

[role="radiogroup"] {
  gap: 0.35rem 0.85rem !important;
}

button {
  color: inherit;
}

button.primary,
.gr-button-primary {
  background: var(--pl-accent) !important;
  color: var(--pl-accent-ink) !important;
  border: none !important;
  font-weight: 600 !important;
}

button.secondary,
.gr-button-secondary {
  background: #2a241c !important;
  color: var(--pl-ink) !important;
  border: 1px solid #4a4339 !important;
}

.pl-hero {
  padding: 0.55rem 0 0.45rem;
  border-bottom: 1px solid var(--pl-line);
  margin-bottom: 0.75rem;
}

.pl-kicker {
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--pl-accent);
  margin: 0 0 0.25rem;
  font-weight: 600;
}

.pl-title {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: clamp(1.55rem, 3vw, 2.15rem);
  font-weight: 700;
  line-height: 1.08;
  margin: 0 0 0.3rem;
  color: var(--pl-ink);
  letter-spacing: -0.02em;
}

.pl-sub {
  max-width: 46rem;
  margin: 0;
  color: var(--pl-muted);
  font-size: 0.88rem;
  line-height: 1.4;
}

.pl-status {
  margin-top: 0.4rem;
  padding: 0.6rem 0.75rem;
  border-left: 3px solid var(--pl-accent);
  background: rgba(224, 179, 90, 0.10);
  color: var(--pl-ink);
  font-size: 0.86rem;
  line-height: 1.4;
  border-radius: 0 8px 8px 0;
}

.pl-status .pl-delta {
  color: var(--pl-muted);
  font-size: 0.82rem;
}

.pl-slider {
  border-radius: 12px;
  overflow: hidden;
}

footer { opacity: 0.5; }

@media (max-width: 900px) {
  .gradio-container { max-width: 100% !important; }
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


def _empty_status(msg: str = "Drop a phone photo to begin.") -> str:
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
            f'<br/><span class="pl-delta">Δ tone {delta["tone_pct"]:.1f}%'
            f' · colour {delta["colour_pct"]:.1f}%'
            f' · mean ΔE {delta["mean_dE"]:.3f}</span>'
        )
    if score_txt:
        html += f"<br/><span class='pl-delta'>Scene cues: {score_txt}</span>"
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
    blocks_kw: dict = {"title": "Phone Look Studio", "fill_height": False}
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
              <p class="pl-kicker">DA3021 · iQOO Neo 9 Pro → premium-camera look</p>
              <h1 class="pl-title">Phone Look Studio</h1>
              <p class="pl-sub">
                Drag the seam to compare before → after. Upload a phone photo,
                pick a camera look, then slide.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=260):
                photo = gr.Image(
                    type="pil",
                    label="Phone photo",
                    height=200,
                    sources=["upload", "clipboard"],
                )
                look = gr.Dropdown(
                    looks,
                    value="Premium camera",
                    label="Camera look",
                )
                strength = gr.Slider(
                    0.0, 1.0, value=0.85, step=0.05, label="Strength"
                )
                with gr.Row():
                    run_btn = gr.Button("Re-run", variant="secondary", scale=1)
                    cmp_btn = gr.Button("Compare all looks", scale=2)

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

            with gr.Column(scale=8, min_width=360):
                view_mode = gr.Radio(
                    ["Slider", "Side-by-side", "Difference"]
                    if use_slider
                    else ["Side-by-side", "Difference"],
                    value=default_view,
                    label="Compare view",
                )

                slider = gr.ImageSlider(
                    label="Before → After (drag the seam)",
                    type="pil",
                    height=380,
                    max_height=420,
                    slider_position=50,
                    interactive=False,
                    elem_classes=["pl-slider"],
                    visible=use_slider,
                )

                with gr.Row(visible=not use_slider) as side_row:
                    before = gr.Image(type="pil", label="Before", height=320)
                    after = gr.Image(type="pil", label="After", height=320)

                diff_img = gr.Image(
                    type="pil",
                    label="Difference (brighter = larger edit)",
                    height=360,
                    visible=False,
                )
                scenario_md = gr.HTML(_empty_status())

        with gr.Accordion("Look strip (all camera looks)", open=False):
            gallery = gr.Gallery(
                label="Look strip",
                columns=3,
                height=200,
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

    # Base (not Soft): Soft+amber paints yellow label chips that hide light text.
    return gr.themes.Base(
        primary_hue="stone",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Sora"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size="md",
    ).set(
        body_background_fill="#12100e",
        body_background_fill_dark="#12100e",
        body_text_color="#f3eee6",
        body_text_color_dark="#f3eee6",
        body_text_color_subdued="#b7aea2",
        block_background_fill="#221e19",
        block_background_fill_dark="#221e19",
        block_border_color="#3a342c",
        block_border_color_dark="#3a342c",
        block_label_background_fill="#221e19",
        block_label_background_fill_dark="#221e19",
        block_label_text_color="#f3eee6",
        block_label_text_color_dark="#f3eee6",
        block_title_text_color="#f3eee6",
        block_title_text_color_dark="#f3eee6",
        input_background_fill="#12100e",
        input_background_fill_dark="#12100e",
        input_border_color="#4a4339",
        input_border_color_dark="#4a4339",
        input_placeholder_color="#8a8074",
        input_placeholder_color_dark="#8a8074",
        button_primary_background_fill="#e0b35a",
        button_primary_background_fill_hover="#ebc474",
        button_primary_text_color="#1a140c",
        button_secondary_background_fill="#2a241c",
        button_secondary_background_fill_hover="#342c22",
        button_secondary_text_color="#f3eee6",
        border_color_primary="#3a342c",
        border_color_accent="#e0b35a",
        color_accent="#e0b35a",
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
