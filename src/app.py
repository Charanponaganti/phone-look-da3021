"""
Phone Look Studio — Gradio UI matching the dark optical-LUT mock.

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

# Mock palette: charcoal + amber, instrument UI
PROFILE_TILES = {
    "Flagship": ("Premium camera", "Balanced"),
    "Summicron": ("Natural DSLR", "Micro-Con"),
    "Positive": ("Vivid DSLR", "Rich Warm"),
}

STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0e0e0e;
  --panel: #141414;
  --panel-2: #1a1a1a;
  --ink: #f5f5f5;
  --muted: #8a8a8a;
  --line: #2a2a2a;
  --amber: #e8a54b;
  --amber-dim: #c4893a;
  --amber-ink: #111111;
}

* { box-shadow: none !important; }

html, body, .gradio-container, .main, footer, .wrap {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: Inter, system-ui, sans-serif !important;
  font-size: 14px !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

.gradio-container .main, div.main {
  max-width: 100% !important;
  width: 100% !important;
  padding: 0 !important;
}

/* ---- header ---- */
.pl-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: #0c0c0c;
}

.pl-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.pl-hex {
  width: 22px; height: 22px;
  background: var(--amber);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  flex-shrink: 0;
}

.pl-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.01em;
}

.pl-crumb {
  margin: 0;
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--muted);
}
.pl-crumb b { color: var(--amber); font-weight: 600; }

.pl-top-actions { display: flex; gap: 8px; }

/* ---- toolbar ---- */
.pl-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--bg);
}

.pl-src-flag {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

/* ---- stage: dot grid ---- */
.pl-stage {
  position: relative;
  min-height: 52vh;
  padding: 18px 20px 8px !important;
  background-color: #101010 !important;
  background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px) !important;
  background-size: 18px 18px !important;
  border-bottom: 1px solid var(--line);
}

.pl-stage .block {
  background: transparent !important;
  border: none !important;
}

/* upload drop look */
.pl-stage .image-container,
.pl-drop .image-container {
  border: 1px dashed #3a3a3a !important;
  border-radius: 14px !important;
  background: rgba(0,0,0,0.25) !important;
  min-height: 360px !important;
}

/* ---- bottom sheet ---- */
.pl-sheet {
  background: var(--panel) !important;
  border-top: 1px solid var(--line);
  padding: 6px 20px 20px !important;
}

.pl-step {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 14px 0 8px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.pl-step .tag { color: var(--amber); font-weight: 600; }

/* labels */
.block .label-wrap,
.block .label-wrap span,
.block-label,
.block > label,
label {
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  opacity: 1 !important;
}

.block {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

input, textarea, select,
input[type="number"], input[type="text"] {
  color: var(--ink) !important;
  background: #121212 !important;
  border: 1px solid #333 !important;
  border-radius: 8px !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  -webkit-text-fill-color: var(--ink) !important;
  opacity: 1 !important;
  min-height: 2.4rem !important;
}

.slider input[type="number"] {
  width: 3.8rem !important;
  min-width: 3.8rem !important;
  background: #1a1a1a !important;
  border: 1px solid #333 !important;
  text-align: center !important;
}

/* segmented view */
[role="radiogroup"] {
  display: inline-flex !important;
  gap: 0 !important;
  background: #161616 !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 999px !important;
  padding: 3px !important;
}

[role="radiogroup"] label {
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 999px !important;
  padding: 0.4rem 0.85rem !important;
  margin: 0 !important;
  font-family: Inter, sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}

[role="radiogroup"] label:has(input:checked) {
  background: #2a2a2a !important;
  color: #fff !important;
}

/* profile tiles */
.pl-tiles [role="radiogroup"] {
  display: grid !important;
  grid-template-columns: repeat(3, 1fr) !important;
  width: 100% !important;
  border-radius: 10px !important;
  padding: 0 !important;
  gap: 8px !important;
  background: transparent !important;
  border: none !important;
}

.pl-tiles [role="radiogroup"] label {
  border: 1px solid #2e2e2e !important;
  border-radius: 10px !important;
  background: #151515 !important;
  color: #9a9a9a !important;
  padding: 12px 14px !important;
  text-align: left !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-family: Inter, sans-serif !important;
}

.pl-tiles [role="radiogroup"] label:has(input:checked) {
  border-color: var(--amber) !important;
  color: #fff !important;
  background: #1a1610 !important;
}

button {
  font-family: Inter, sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  min-height: 2.55rem !important;
  border-radius: 10px !important;
}

button.primary, .gr-button-primary {
  background: var(--amber) !important;
  color: var(--amber-ink) !important;
  border: 1px solid var(--amber) !important;
}

button.secondary, .gr-button-secondary {
  background: #171717 !important;
  color: #d0d0d0 !important;
  border: 1px solid #333 !important;
}

.pl-status {
  margin: 8px 0 0;
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.5;
}
.pl-status strong { color: var(--amber); font-weight: 600; }
.pl-status .pl-delta { color: #bdbdbd; margin-left: 0.5rem; }

footer { display: none !important; }

@media (max-width: 900px) {
  .pl-top { flex-direction: column; align-items: flex-start; }
  .pl-tiles [role="radiogroup"] { grid-template-columns: 1fr !important; }
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
    return [str(p) for p in paths]


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


def _empty_status() -> str:
    return '<p class="pl-status">ready — drop a phone photo above</p>'


def _status_html(info: dict, delta: dict | None = None) -> str:
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
    html += "</p>"
    return html


def _src_flag(photo) -> str:
    if photo is None:
        return "NO SOURCE LOADED"
    return "SOURCE LOADED"


def build_demo():
    import gradio as gr

    model_names = _list_models()
    try:
        fallback_model = get_best_model_path().name
    except FileNotFoundError:
        fallback_model = model_names[0] if model_names else None

    cache: dict[str, dict] = {}
    use_slider = _has_image_slider()
    examples = _example_paths()
    sample_path = examples[0] if examples else None

    def _get_model(name: str):
        if name not in cache:
            cache[name] = load_model(MODELS / name)
        return cache[name]

    def _resolve_look(profile: str, look: str) -> str:
        if profile in PROFILE_TILES:
            return PROFILE_TILES[profile][0]
        return look or "Premium camera"

    def run(photo, mode, look, profile, model_name, preset, strength, view_mode):
        view_mode = view_mode or ("Split Seam" if use_slider else "Side-by-Side")
        # map UI labels → internal
        view_key = {
            "Split Seam": "Slider",
            "Side-by-Side": "Side-by-side",
            "Difference": "Difference",
        }.get(view_mode, view_mode)

        show_slider = use_slider and view_key == "Slider"
        show_side = view_key == "Side-by-side"
        show_diff = view_key == "Difference"
        flag = _src_flag(photo)

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
                flag,
                None,
            )

        photo = _as_pil(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=1000)
        look_name = _resolve_look(profile, look)

        if mode == "Auto":
            out, info = enhance_bgr(
                bgr,
                mode="auto",
                strength=float(strength),
                preset=preset,
                look=look_name,
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
            flag,
            after_pil,  # export
        )

    def switch_view(view_mode, before_pil, after_pil):
        view_key = {
            "Split Seam": "Slider",
            "Side-by-Side": "Side-by-side",
            "Difference": "Difference",
        }.get(view_mode or "Split Seam", "Slider")
        before_pil = _as_pil(before_pil)
        after_pil = _as_pil(after_pil)
        heat = None
        if view_key == "Difference" and before_pil and after_pil:
            heat = _bgr_to_pil(
                _diff_heatmap(_pil_to_bgr(before_pil), _pil_to_bgr(after_pil))
            )
        return (
            gr.update(visible=use_slider and view_key == "Slider"),
            gr.update(visible=view_key == "Side-by-side"),
            gr.update(visible=view_key == "Difference", value=heat),
        )

    def load_sample():
        if not sample_path:
            raise gr.Error("No sample images in assets/examples.")
        return Image.open(sample_path).convert("RGB")

    def on_profile(profile, look_dd):
        if profile in PROFILE_TILES:
            return PROFILE_TILES[profile][0]
        return look_dd

    def compare(photo, strength, profile, look):
        if photo is None:
            return []
        photo = _as_pil(photo)
        bgr = _resize_max(_pil_to_bgr(photo), max_side=720)
        look_name = _resolve_look(profile, look)
        gallery = []
        for name in list_looks():
            out = enhance_bgr(
                bgr, mode="auto", strength=float(strength), look=name
            )
            gallery.append((_bgr_to_pil(out), name))
        # put active look first visually by order — fine
        _ = look_name
        return gallery

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
    default_view = "Split Seam" if use_slider else "Side-by-Side"

    with gr.Blocks(**blocks_kw) as demo:
        st_before = gr.State(None)
        st_after = gr.State(None)

        gr.HTML(
            """
            <div class="pl-top">
              <div class="pl-brand">
                <div class="pl-hex"></div>
                <h1 class="pl-title">Phone Look Studio</h1>
                <p class="pl-crumb"><b>DA3021</b> / IQOO NEO 9 PRO → OPTICAL LUT</p>
              </div>
            </div>
            """
        )

        with gr.Row(elem_classes=["pl-bar"]):
            with gr.Column(scale=5):
                view_mode = gr.Radio(
                    ["Split Seam", "Side-by-Side", "Difference"]
                    if use_slider
                    else ["Side-by-Side", "Difference"],
                    value=default_view,
                    label="View",
                    show_label=False,
                )
            with gr.Column(scale=3):
                src_flag = gr.Textbox(
                    value="NO SOURCE LOADED",
                    label="Source",
                    show_label=False,
                    interactive=False,
                    elem_classes=["pl-src-flag"],
                )
            with gr.Column(scale=2):
                sample_btn = gr.Button("Load Sample RAW", variant="secondary")
                # Export uses download button on after image; keep a downloadable file
                export_btn = gr.DownloadButton("Export", variant="primary")

        # STAGE
        with gr.Column(elem_classes=["pl-stage", "pl-drop"]):
            photo = gr.Image(
                type="pil",
                label="Drop a phone photo here, or click to browse",
                height=380,
                sources=["upload", "clipboard"],
            )
            slider = gr.ImageSlider(
                label="Split seam · drag to compare",
                type="pil",
                height=480,
                max_height=560,
                slider_position=45,
                interactive=False,
                visible=False,
            )
            with gr.Row(visible=False) as side_row:
                before = gr.Image(type="pil", label="Before", height=400)
                after = gr.Image(type="pil", label="After", height=400)
            diff_img = gr.Image(
                type="pil",
                label="Difference",
                height=420,
                visible=False,
            )

        # BOTTOM SHEET
        with gr.Column(elem_classes=["pl-sheet"]):
            gr.HTML('<div class="pl-step"><span>01 · Input source</span><span>None</span></div>')
            with gr.Row():
                # Gradio can't easily "paste" via button; clipboard is on Image sources
                replace_hint = gr.Markdown(
                    "Use the stage dropzone · **Replace** by uploading again · **Paste** with Ctrl+V on the image"
                )

            gr.HTML(
                '<div class="pl-step"><span>02 · Optical profile</span>'
                '<span class="tag">LUT-3D</span></div>'
            )
            look = gr.Dropdown(
                looks,
                value="Premium camera",
                label="Profile",
                show_label=False,
            )
            profile = gr.Radio(
                ["Flagship", "Summicron", "Positive"],
                value="Flagship",
                label="Tiles",
                show_label=False,
                elem_classes=["pl-tiles"],
            )

            gr.HTML('<div class="pl-step"><span>03 · Blend strength</span></div>')
            strength = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Strength", show_label=False)

            with gr.Accordion("04 · Optical & tone fine-tuning", open=False):
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

            scenario_md = gr.HTML(_empty_status())

            with gr.Row():
                run_btn = gr.Button("Re-run", variant="secondary", scale=1)
                cmp_btn = gr.Button("Compare All Looks", variant="primary", scale=3)

            gallery = gr.Gallery(
                label="Look strip",
                columns=3,
                height=200,
                object_fit="contain",
            )

            # hidden file for export button
            export_file = gr.File(label="export", visible=False)

        if examples:
            gr.Examples(
                examples=examples[:4],
                inputs=photo,
                label="Samples",
                examples_per_page=4,
            )

        def _export_path(after_pil):
            if after_pil is None:
                return None
            after_pil = _as_pil(after_pil)
            out = ROOT / "outputs" / "enhanced" / "_export.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            after_pil.save(out)
            return str(out)

        def run_and_export(*args):
            result = list(run(*args))
            after_pil = result[2]  # after image
            path = _export_path(after_pil)
            # DownloadButton wants a file path update
            result.append(gr.update(value=path) if path else gr.update())
            return result

        run_outs = [
            slider,
            before,
            after,
            diff_img,
            scenario_md,
            st_before,
            st_after,
            side_row,
            src_flag,
            after,
            export_btn,
        ]
        inputs = [photo, mode, look, profile, model_name, preset, strength, view_mode]

        def run_wrapped(photo, mode, look, profile, model_name, preset, strength, view_mode):
            r = run(photo, mode, look, profile, model_name, preset, strength, view_mode)
            # r: slider, before, after, diff, status, st_b, st_a, side, flag, export_pil
            path = _export_path(r[9])
            return (*r[:10], gr.update(value=path) if path else gr.update(value=None))

        run_btn.click(run_wrapped, inputs, run_outs)
        photo.change(run_wrapped, inputs, run_outs)
        look.change(run_wrapped, inputs, run_outs)
        strength.release(run_wrapped, inputs, run_outs)
        profile.change(
            lambda p, l: on_profile(p, l),
            [profile, look],
            [look],
        ).then(run_wrapped, inputs, run_outs)

        view_mode.change(
            switch_view,
            [view_mode, st_before, st_after],
            [slider, side_row, diff_img],
        )
        sample_btn.click(load_sample, outputs=[photo])
        cmp_btn.click(compare, [photo, strength, profile, look], [gallery])

        # silence unused
        _ = (replace_hint, export_file)

    return demo


def _studio_theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue="amber",
        secondary_hue="zinc",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size="md",
    ).set(
        body_background_fill="#0e0e0e",
        body_background_fill_dark="#0e0e0e",
        body_text_color="#f5f5f5",
        body_text_color_dark="#f5f5f5",
        body_text_color_subdued="#8a8a8a",
        block_background_fill="#141414",
        block_background_fill_dark="#141414",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        block_label_background_fill="#141414",
        block_label_background_fill_dark="#141414",
        block_label_text_color="#8a8a8a",
        block_label_text_color_dark="#8a8a8a",
        block_title_text_color="#f5f5f5",
        input_background_fill="#121212",
        input_background_fill_dark="#121212",
        input_border_color="#333333",
        input_border_color_dark="#333333",
        button_primary_background_fill="#e8a54b",
        button_primary_background_fill_hover="#f0b45c",
        button_primary_text_color="#111111",
        button_secondary_background_fill="#171717",
        button_secondary_background_fill_hover="#222222",
        button_secondary_text_color="#d0d0d0",
        border_color_primary="#2a2a2a",
        color_accent="#e8a54b",
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
