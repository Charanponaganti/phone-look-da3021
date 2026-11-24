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
  max-width: 1120px !important;
  font-family: "Sora", system-ui, sans-serif !important;
  color: var(--pl-ink) !important;
}

.gradio-container,
.main,
footer {
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(224, 179, 90, 0.16), transparent 55%),
    radial-gradient(700px 380px at 92% 8%, rgba(90, 140, 160, 0.10), transparent 50%),
    linear-gradient(180deg, #1a1612 0%, #12100e 55%, #0e0c0a 100%) !important;
}

.pl-hero {
  padding: 0.85rem 0 0.35rem;
  border-bottom: 1px solid var(--pl-line);
  margin-bottom: 0.85rem;
}

.pl-kicker {
  font-size: 0.68rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--pl-accent);
  margin: 0 0 0.3rem;
  font-weight: 600;
}

.pl-title {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: clamp(1.7rem, 3.6vw, 2.45rem);
  font-weight: 700;
  line-height: 1.05;
  margin: 0 0 0.35rem;
  color: var(--pl-ink);
  letter-spacing: -0.02em;
}

.pl-sub {
  max-width: 40rem;
  margin: 0;
  color: var(--pl-muted);
  font-size: 0.9rem;
  line-height: 1.45;
}

.pl-status {
  margin-top: 0.45rem;
  padding: 0.65rem 0.8rem;
  border-left: 3px solid var(--pl-accent);
  background: rgba(224, 179, 90, 0.08);
  color: var(--pl-ink);
  font-size: 0.88rem;
  line-height: 1.45;
}

.pl-status .pl-delta {
  color: var(--pl-muted);
  font-size: 0.84rem;
}

.pl-pills label,
.pl-view label {
  border-radius: 999px !important;
  border: 1px solid var(--pl-line) !important;
  background: var(--pl-bg1) !important;
  color: var(--pl-muted) !important;
  padding: 0.35rem 0.75rem !important;
  font-size: 0.78rem !important;
  white-space: nowrap;
}

.pl-pills input:checked + label,
.pl-pills label:has(input:checked),
.pl-view input:checked + label,
.pl-view label:has(input:checked) {
  background: var(--pl-accent) !important;
  color: var(--pl-accent-ink) !important;
  border-color: var(--pl-accent) !important;
  font-weight: 600 !important;
}

.pl-pills .wrap,
.pl-view .wrap {
  gap: 0.35rem !important;
  flex-wrap: wrap !important;
}

@media (max-width: 860px) {
  .pl-pills .wrap {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    padding-bottom: 0.25rem;
  }
}

.pl-slider {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px var(--pl-line);
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


def _resize_max(bgr: np.ndarray, max_side: int = 1400) -> np.ndarray:
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
        bgr = _resize_max(_pil_to_bgr(photo))

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
                Drag the seam to see before → after. Learns colour &amp; tone from
                DPED phone↔DSLR pairs plus your Neo Photo→Night/HDR captures.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=240):
                photo = gr.Image(
                    type="pil",
                    label="Phone photo",
                    height=220,
                    sources=["upload", "clipboard"],
                )
                look = gr.Radio(
                    looks,
                    value="Premium camera",
                    label="Camera look",
                    elem_classes=["pl-pills"],
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

            with gr.Column(scale=7, min_width=320):
                view_mode = gr.Radio(
                    ["Slider", "Side-by-side", "Difference"]
                    if use_slider
                    else ["Side-by-side", "Difference"],
                    value=default_view,
                    label="Compare view",
                    elem_classes=["pl-view"],
                )

                slider = gr.ImageSlider(
                    label="Before → After  (drag the seam)",
                    type="pil",
                    height=460,
                    max_height=520,
                    slider_position=50,
                    interactive=False,
                    elem_classes=["pl-slider"],
                    visible=use_slider,
                )

                with gr.Row(visible=not use_slider) as side_row:
                    before = gr.Image(type="pil", label="Before", height=360)
                    after = gr.Image(type="pil", label="After", height=360)

                diff_img = gr.Image(
                    type="pil",
                    label="Difference (brighter = larger edit)",
                    height=420,
                    visible=False,
                )
                scenario_md = gr.HTML(_empty_status())

        with gr.Accordion("Look strip (all camera looks)", open=False):
            gallery = gr.Gallery(
                label="Look strip",
                columns=3,
                height=220,
                object_fit="contain",
            )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=photo,
                label="Try an example (hazy / low-light first)",
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
