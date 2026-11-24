"""
Phone Look — Gradio UI ("Contact Sheet" lab layout).

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

# Contact Sheet: paper / press / darkroom proofing — not dark-gold "AI studio".
STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
  --bg: #F4F2ED;
  --ink: #17181A;
  --muted: #6B6E73;
  --line: #DAD7D0;
  --stage: #E8E6E1;
  --accent: #1F4FD8;
  --accent-ink: #FFFFFF;
}

* { border-radius: 2px !important; box-shadow: none !important; }

html, body, .gradio-container, .main, footer, .wrap {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 13px !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

.gradio-container .main,
div.main {
  max-width: 100% !important;
  width: 100% !important;
  padding: 0 !important;
}

/* ---- masthead ---- */
.pl-mast {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  padding: 14px 18px 12px;
  border-bottom: 1px solid var(--line);
  background: var(--bg);
}

.pl-mark {
  font-family: "Newsreader", Georgia, serif !important;
  font-size: 1.55rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
  line-height: 1;
}

.pl-mark span {
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-left: 0.65rem;
  vertical-align: middle;
}

.pl-line {
  margin: 0;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 400;
  letter-spacing: 0.02em;
}

/* ---- toolbar / stage / status strips ---- */
.pl-toolbar {
  padding: 8px 14px !important;
  border-bottom: 1px solid var(--line) !important;
  background: var(--bg) !important;
}

.pl-stage {
  background: var(--stage) !important;
  padding: 12px 14px 8px !important;
  border-bottom: 1px solid var(--line) !important;
  min-height: 62vh;
}

.pl-foot {
  padding: 10px 18px 18px !important;
  background: var(--bg) !important;
}

/* Labels: mono uppercase, quiet */
.block .label-wrap,
.block .label-wrap span,
.block-label,
.block > label,
label,
span[data-testid="block-info"] {
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  opacity: 1 !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

.block {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Inputs */
input, textarea, select,
input[type="number"],
input[type="text"] {
  color: var(--ink) !important;
  background: #FFFFFF !important;
  border: 1px solid var(--line) !important;
  border-radius: 2px !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--ink) !important;
  min-height: 2.15rem !important;
}

.slider input[type="number"] {
  width: 3.6rem !important;
  min-width: 3.6rem !important;
}

[role="radiogroup"] {
  gap: 0 !important;
  border: 1px solid var(--line) !important;
  display: inline-flex !important;
  flex-wrap: wrap !important;
  background: #fff !important;
}

[role="radiogroup"] label,
fieldset label,
.wrap label span {
  color: var(--ink) !important;
  background: transparent !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  padding: 0.45rem 0.75rem !important;
  border-right: 1px solid var(--line) !important;
  margin: 0 !important;
}

[role="radiogroup"] label:last-child {
  border-right: none !important;
}

[role="radiogroup"] label:has(input:checked),
[role="radiogroup"] input:checked + span {
  background: var(--ink) !important;
  color: #fff !important;
}

button {
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  min-height: 2.25rem !important;
  border-radius: 2px !important;
  box-shadow: none !important;
}

button.primary, .gr-button-primary {
  background: var(--accent) !important;
  color: var(--accent-ink) !important;
  border: 1px solid var(--accent) !important;
}

button.secondary, .gr-button-secondary {
  background: transparent !important;
  color: var(--ink) !important;
  border: 1px solid var(--line) !important;
}

/* Stage images */
.pl-stage .image-container,
.pl-stage img {
  background: var(--stage) !important;
  max-height: min(68vh, 740px) !important;
}

.pl-status {
  margin: 0;
  padding: 0;
  color: var(--ink);
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  line-height: 1.55;
  border: none;
  background: transparent;
}

.pl-status strong {
  color: var(--ink);
  font-weight: 600;
}

.pl-status .pl-delta {
  color: var(--muted);
  display: inline;
  margin-left: 0.65rem;
}

.pl-status .pl-cues {
  display: block;
  margin-top: 0.2rem;
  color: var(--muted);
  font-size: 11px;
}

footer { display: none !important; }

@media (max-width: 900px) {
  .pl-mast { flex-direction: column; align-items: flex-start; gap: 0.35rem; }
  .pl-stage { min-height: 48vh; }
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


def _empty_status(msg: str = "ready — upload a phone photo") -> str:
    return f'<p class="pl-status">{msg}</p>'


def _status_html(info: dict, delta: dict | None = None) -> str:
    scores = info.get("scores") or {}
    score_txt = " · ".join(f"{k} {v}" for k, v in scores.items()) if scores else ""
    blurb = info.get("blurb") or ""
    html = f'<p class="pl-status"><strong>{info.get("label", "?")}</strong>'
    if blurb:
        html += f" — {blurb}"
    if delta:
        html += (
            f'<span class="pl-delta">Δ tone {delta["tone_pct"]:.1f}%'
            f' · colour {delta["colour_pct"]:.1f}%'
            f' · ΔE {delta["mean_dE"]:.3f}</span>'
        )
    if score_txt:
        html += f'<span class="pl-cues">{score_txt}</span>'
    html += "</p>"
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
        "title": "retouch · lab",
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
            <header class="pl-mast">
              <h1 class="pl-mark">retouch · lab<span>DA3021</span></h1>
              <p class="pl-line">Upload a phone photo. Compare before and after.</p>
            </header>
            """
        )

        with gr.Row(elem_classes=["pl-toolbar"], equal_height=True):
            photo = gr.Image(
                type="pil",
                label="Upload",
                height=120,
                sources=["upload", "clipboard"],
                scale=2,
            )
            look = gr.Dropdown(
                looks,
                value="Premium camera",
                label="Look",
                scale=2,
            )
            strength = gr.Slider(
                0.0, 1.0, value=0.9, step=0.05, label="Strength", scale=3
            )
            with gr.Column(scale=2, min_width=160):
                run_btn = gr.Button("Enhance", variant="primary")
                cmp_btn = gr.Button("All looks", variant="secondary")

        with gr.Column(elem_classes=["pl-stage"]):
            slider = gr.ImageSlider(
                label="Before → After",
                type="pil",
                height=580,
                max_height=720,
                slider_position=45,
                interactive=False,
                visible=use_slider,
            )
            with gr.Row(visible=not use_slider) as side_row:
                before = gr.Image(type="pil", label="Before", height=480)
                after = gr.Image(type="pil", label="After", height=480)
            diff_img = gr.Image(
                type="pil",
                label="Difference",
                height=500,
                visible=False,
            )
            view_mode = gr.Radio(
                ["Slider", "Side-by-side", "Difference"]
                if use_slider
                else ["Side-by-side", "Difference"],
                value=default_view,
                label="Compare",
            )

        with gr.Column(elem_classes=["pl-foot"]):
            scenario_md = gr.HTML(_empty_status())
            with gr.Accordion("Advanced", open=False):
                mode = gr.Radio(
                    ["Auto", "Single model"],
                    value="Auto",
                    label="Engine",
                )
                preset = gr.Dropdown(
                    list_presets(), value="Balanced", label="Preset"
                )
                model_name = gr.Dropdown(
                    choices=model_names or ["(none)"],
                    value=fallback_model,
                    label="Model",
                )
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
                label="Examples",
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

    return gr.themes.Base(
        primary_hue="blue",
        secondary_hue="zinc",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size="md",
        radius_size="none",
    ).set(
        body_background_fill="#F4F2ED",
        body_background_fill_dark="#F4F2ED",
        body_text_color="#17181A",
        body_text_color_dark="#17181A",
        body_text_color_subdued="#6B6E73",
        block_background_fill="#F4F2ED",
        block_background_fill_dark="#F4F2ED",
        block_border_color="#DAD7D0",
        block_border_color_dark="#DAD7D0",
        block_label_background_fill="#F4F2ED",
        block_label_background_fill_dark="#F4F2ED",
        block_label_text_color="#6B6E73",
        block_label_text_color_dark="#6B6E73",
        block_title_text_color="#17181A",
        block_title_text_color_dark="#17181A",
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_border_color="#DAD7D0",
        input_border_color_dark="#DAD7D0",
        input_placeholder_color="#6B6E73",
        input_placeholder_color_dark="#6B6E73",
        button_primary_background_fill="#1F4FD8",
        button_primary_background_fill_hover="#163DB0",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#F4F2ED",
        button_secondary_background_fill_hover="#E8E6E1",
        button_secondary_text_color="#17181A",
        border_color_primary="#DAD7D0",
        color_accent="#1F4FD8",
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
