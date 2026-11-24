"""
Phone Look Studio — Gradio 6 UI (dark optical-LUT mock, Render-safe).

Run:
  python -m src.app
  python app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .apply import enhance_bgr
from .io_utils import bgr_to_lab01
from .paths import ROOT
from .pipeline import list_looks, list_presets

EXAMPLES_DIR = ROOT / "assets" / "examples"

# Quick tiles → look name (dropdown remains source of truth)
PROFILE_TILES = {
    "Flagship": "Premium camera",
    "Summicron": "Natural DSLR",
    "Positive": "Vivid DSLR",
}

VIEW_SPLIT = "Split Seam"
VIEW_SIDE = "Side-by-Side"
VIEW_DIFF = "Difference"

STUDIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0e0e0e;
  --panel: #141414;
  --ink: #f5f5f5;
  --muted: #8a8a8a;
  --line: #2a2a2a;
  --amber: #e8a54b;
  --amber-ink: #111111;
}

#phone-look,
#phone-look .main {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: Inter, system-ui, sans-serif !important;
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

#phone-look footer { display: none !important; }

/* ---- header ---- */
#phone-look .pl-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: #0c0c0c;
}

#phone-look .pl-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

#phone-look .pl-hex {
  width: 22px;
  height: 22px;
  background: var(--amber);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  flex-shrink: 0;
}

#phone-look .pl-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.01em;
}

#phone-look .pl-crumb {
  margin: 0;
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--muted);
}
#phone-look .pl-crumb b { color: var(--amber); font-weight: 600; }

/* ---- bar / sheet ---- */
#phone-look .pl-bar {
  padding: 10px 20px !important;
  border-bottom: 1px solid var(--line);
  background: var(--bg);
}

#phone-look .pl-stage {
  min-height: 48vh;
  padding: 16px 20px 10px !important;
  background-color: #101010 !important;
  background-image: radial-gradient(rgba(255,255,255,0.06) 1px, transparent 1px) !important;
  background-size: 18px 18px !important;
  border-bottom: 1px solid var(--line);
}

#phone-look .pl-sheet {
  background: var(--panel) !important;
  border-top: 1px solid var(--line);
  padding: 8px 20px 22px !important;
}

#phone-look .pl-step {
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
#phone-look .pl-step .tag { color: var(--amber); font-weight: 600; }

/* Gradio "processing | 12.3s" chip — keep off the toolbar, readable on dark */
#phone-look .progress-text,
#phone-look .meta-text,
#phone-look .eta-bar,
#phone-look .pending,
.progress-text,
.meta-text,
.eta-bar {
  color: #111111 !important;
  -webkit-text-fill-color: #111111 !important;
  background: var(--amber) !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 6px 12px !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
  z-index: 50 !important;
}
#phone-look .progress-level,
.progress-level {
  background: rgba(232, 165, 75, 0.25) !important;
}

#phone-look .pl-status {
  margin: 10px 0 0;
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  color: var(--muted);
  line-height: 1.5;
}
#phone-look .pl-status strong { color: var(--amber); font-weight: 600; }
#phone-look .pl-status .pl-delta { color: #bdbdbd; margin-left: 0.5rem; }

/* Dropzone vs compact replace strip */
#phone-look .pl-drop .image-container {
  border: 1px dashed #3a3a3a !important;
  border-radius: 14px !important;
  background: rgba(0,0,0,0.25) !important;
  min-height: 340px !important;
}
#phone-look .pl-compact .image-container {
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  min-height: 88px !important;
  max-height: 110px !important;
}

/* Scoped labels / inputs — not global */
#phone-look .label-wrap span,
#phone-look .block-label {
  color: var(--muted) !important;
  background: transparent !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  opacity: 1 !important;
}

#phone-look input[type="number"],
#phone-look input[type="text"],
#phone-look textarea,
#phone-look select {
  color: var(--ink) !important;
  background: #121212 !important;
  border: 1px solid #333 !important;
  border-radius: 8px !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  -webkit-text-fill-color: var(--ink) !important;
  opacity: 1 !important;
}

#phone-look .slider input[type="number"] {
  width: 3.8rem !important;
  min-width: 3.8rem !important;
  background: #1a1a1a !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  text-align: center !important;
}

/* View pills — Gradio 6: target by elem_id, not role=radiogroup */
#view-mode {
  background: #161616 !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 999px !important;
  padding: 3px !important;
}
#view-mode label {
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 999px !important;
  padding: 0.4rem 0.85rem !important;
  margin: 0 !important;
  font-family: Inter, sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
#view-mode label:has(input:checked) {
  background: #2a2a2a !important;
  color: #fff !important;
}

/* Profile tiles */
#look-tiles .wrap {
  display: grid !important;
  grid-template-columns: repeat(3, 1fr) !important;
  gap: 8px !important;
}
#look-tiles label {
  border: 1px solid #2e2e2e !important;
  border-radius: 10px !important;
  background: #151515 !important;
  color: #9a9a9a !important;
  padding: 12px 14px !important;
  text-align: left !important;
  font-family: Inter, sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
#look-tiles label:has(input:checked) {
  border-color: var(--amber) !important;
  color: #fff !important;
  background: #1a1610 !important;
}

#phone-look button.primary,
#phone-look .primary {
  background: var(--amber) !important;
  color: var(--amber-ink) !important;
  border: 1px solid var(--amber) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}
#phone-look button.secondary,
#phone-look .secondary {
  background: #171717 !important;
  color: #d0d0d0 !important;
  border: 1px solid #333 !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

/* Gradio error/info toasts: dark theme made light text on white unreadable */
.toast-wrap,
.toast-body,
div.toast,
[class*="toast"] {
  color: #111111 !important;
  -webkit-text-fill-color: #111111 !important;
}
.toast-wrap .toast-title,
.toast-body .toast-title,
.toast .toast-title {
  color: #111111 !important;
  font-weight: 700 !important;
}
.toast-wrap .toast-text,
.toast-body span,
.toast p,
.toast li {
  color: #222222 !important;
  opacity: 1 !important;
}

/* Look strip: tight square tiles, no tall empty black frames */
#look-strip {
  margin-top: 0.5rem;
}
#look-strip .grid-wrap,
#look-strip .thumbnails,
#look-strip .gallery-item,
#look-strip [class*="thumbnail"] {
  gap: 10px !important;
}
#look-strip button.thumbnail-item,
#look-strip .thumbnail-item,
#look-strip .gallery-item {
  background: #121212 !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  aspect-ratio: 1 / 1 !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 0 !important;
}
#look-strip img {
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
  display: block !important;
}
#look-strip .caption,
#look-strip figcaption,
#look-strip .thumbnail-item > div:last-child {
  position: absolute !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  margin: 0 !important;
  padding: 6px 8px !important;
  background: linear-gradient(transparent, rgba(0,0,0,0.82)) !important;
  color: #f5f5f5 !important;
  font-family: "IBM Plex Mono", monospace !important;
  font-size: 11px !important;
  text-align: left !important;
}

@media (max-width: 900px) {
  #look-tiles .wrap { grid-template-columns: 1fr !important; }
  #phone-look .pl-top { flex-direction: column; align-items: flex-start; }
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


def _downscale_pil(img: Image.Image, max_side: int = 1000) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def _example_paths() -> list[str]:
    if not EXAMPLES_DIR.exists():
        return []
    paths = list(EXAMPLES_DIR.glob("*.png")) + list(EXAMPLES_DIR.glob("*.jpg"))
    return [str(p) for p in sorted(paths, key=lambda p: p.stat().st_size, reverse=True)]


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
    return (
        '<p class="pl-status"><strong>ready</strong>'
        '<span class="pl-delta"> — drop a phone photo above</span></p>'
    )


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


def _export_path(after_pil: Image.Image | None) -> str | None:
    if after_pil is None:
        return None
    import os

    out_dir = ROOT / "outputs" / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".jpg", prefix="export_", dir=str(out_dir))
    os.close(fd)
    path = Path(name)
    after_pil.convert("RGB").save(path, format="JPEG", quality=92)
    return str(path)


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
        background_fill_primary="#0e0e0e",
        background_fill_primary_dark="#0e0e0e",
        background_fill_secondary="#141414",
        background_fill_secondary_dark="#141414",
        block_background_fill="#141414",
        block_background_fill_dark="#141414",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        block_label_background_fill="#141414",
        block_label_background_fill_dark="#141414",
        block_label_text_color="#8a8a8a",
        block_label_text_color_dark="#8a8a8a",
        block_title_text_color="#f5f5f5",
        block_title_text_color_dark="#f5f5f5",
        input_background_fill="#121212",
        input_background_fill_dark="#121212",
        input_border_color="#333333",
        input_border_color_dark="#333333",
        input_placeholder_color="#8a8a8a",
        input_placeholder_color_dark="#8a8a8a",
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


def build_demo():
    import gradio as gr

    use_slider = hasattr(gr, "ImageSlider")
    examples = _example_paths()
    sample_path = examples[0] if examples else None
    looks = list_looks()
    presets = list_presets()
    default_view = VIEW_SPLIT if use_slider else VIEW_SIDE
    view_choices = (
        [VIEW_SPLIT, VIEW_SIDE, VIEW_DIFF]
        if use_slider
        else [VIEW_SIDE, VIEW_DIFF]
    )

    def _view_flags(view: str, has_result: bool):
        view = view or default_view
        show_slider = bool(use_slider and has_result and view == VIEW_SPLIT)
        show_side = bool(has_result and view == VIEW_SIDE)
        show_diff = bool(has_result and view == VIEW_DIFF)
        return show_slider, show_side, show_diff

    def _safe_look(look: str | None) -> str:
        if look in looks:
            return look  # type: ignore[return-value]
        return "Premium camera"

    def _safe_preset(preset: str | None) -> str:
        if preset in presets:
            return preset  # type: ignore[return-value]
        return "Balanced"

    def process(photo, look, strength, preset, view):
        """Single enhance path — Auto cascade only."""
        look_name = _safe_look(look)
        preset_name = _safe_preset(preset)

        if photo is None:
            show_slider, show_side, show_diff = _view_flags(view, False)
            return (
                gr.update(elem_classes=["pl-drop"]),
                gr.update(value=None, visible=show_slider),
                None,
                None,
                gr.update(value=None, visible=show_diff),
                gr.update(visible=show_side),
                _empty_status(),
                gr.update(value=None, interactive=False),
                None,
                None,
            )

        pil = _downscale_pil(_as_pil(photo), 1000)
        bgr = _pil_to_bgr(pil)
        out, info = enhance_bgr(
            bgr,
            mode="auto",
            strength=float(strength),
            preset=preset_name,
            look=look_name,
            return_info=True,
        )
        before_pil = _bgr_to_pil(bgr)
        after_pil = _bgr_to_pil(out)
        delta = _lab_delta(bgr, out)
        status = _status_html(info, delta)
        show_slider, show_side, show_diff = _view_flags(view, True)

        heat = None
        if show_diff:
            heat = _bgr_to_pil(_diff_heatmap(bgr, out))

        slider_val = (before_pil, after_pil) if use_slider else None
        export = _export_path(after_pil)

        return (
            gr.update(elem_classes=["pl-compact"]),
            gr.update(value=slider_val, visible=show_slider),
            before_pil,
            after_pil,
            gr.update(value=heat, visible=show_diff),
            gr.update(visible=show_side),
            status,
            gr.update(value=export, interactive=bool(export)),
            before_pil,
            after_pil,
        )

    def switch_view(view, before_pil, after_pil):
        has = before_pil is not None and after_pil is not None
        show_slider, show_side, show_diff = _view_flags(view, has)
        heat = None
        if show_diff and has:
            b = _as_pil(before_pil)
            a = _as_pil(after_pil)
            heat = _bgr_to_pil(_diff_heatmap(_pil_to_bgr(b), _pil_to_bgr(a)))
        return (
            gr.update(visible=show_slider),
            gr.update(visible=show_side),
            gr.update(visible=show_diff, value=heat),
        )

    def load_sample():
        if not sample_path:
            raise gr.Error("No sample images in assets/examples.")
        return Image.open(sample_path).convert("RGB")

    def tile_to_look(profile: str) -> str:
        return PROFILE_TILES.get(profile, "Premium camera")

    def compare_all(photo, strength, look, preset):
        if photo is None:
            return []
        pil = _downscale_pil(_as_pil(photo), 720)
        bgr = _pil_to_bgr(pil)
        gallery = []
        for name in looks:
            out = enhance_bgr(
                bgr,
                mode="auto",
                strength=float(strength),
                preset=preset or "Balanced",
                look=name,
            )
            gallery.append((_bgr_to_pil(out), name))
        return gallery

    with gr.Blocks(
        title="Phone Look Studio",
        fill_height=False,
        fill_width=True,
        elem_id="phone-look",
    ) as demo:
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
            view_mode = gr.Radio(
                view_choices,
                value=default_view,
                label="View",
                show_label=False,
                elem_id="view-mode",
                scale=7,
            )
            with gr.Column(scale=3, min_width=200):
                sample_btn = gr.Button("Load sample", variant="secondary")
                export_btn = gr.DownloadButton(
                    "Export", variant="primary", interactive=False
                )

        with gr.Column(elem_classes=["pl-stage"]):
            # Wrapper gets compact/drop classes — never put `photo` in process outputs
            # (Gradio input+output on same Image can scramble sibling Dropdown values).
            with gr.Column(elem_classes=["pl-drop"]) as upload_col:
                photo = gr.Image(
                    type="pil",
                    label="Drop a phone photo · or click to browse",
                    height=360,
                    sources=["upload", "clipboard"],
                    format="jpeg",
                )
            if use_slider:
                slider = gr.ImageSlider(
                    label="Split seam · drag to compare",
                    type="pil",
                    height=460,
                    max_height=540,
                    slider_position=45,
                    interactive=False,
                    visible=False,
                    format="jpeg",
                )
            else:
                slider = gr.Image(
                    label="Result",
                    type="pil",
                    height=460,
                    interactive=False,
                    visible=False,
                    format="jpeg",
                )

            with gr.Row(visible=False) as side_row:
                before = gr.Image(
                    type="pil",
                    label="Before",
                    height=400,
                    interactive=False,
                    buttons=[],
                    format="jpeg",
                )
                after = gr.Image(
                    type="pil",
                    label="After",
                    height=400,
                    interactive=False,
                    buttons=["download"],
                    format="jpeg",
                )
            diff_img = gr.Image(
                type="pil",
                label="Difference",
                height=420,
                interactive=False,
                buttons=[],
                visible=False,
                format="jpeg",
            )

        with gr.Column(elem_classes=["pl-sheet"]):
            gr.HTML(
                '<div class="pl-step"><span>01 · Input source</span>'
                "<span>Replace via stage · Paste Ctrl+V</span></div>"
            )

            gr.HTML(
                '<div class="pl-step"><span>02 · Optical profile</span>'
                '<span class="tag">LOOK</span></div>'
            )
            look = gr.Dropdown(
                looks,
                value="Premium camera",
                label="Look",
                show_label=False,
                elem_id="look-select",
                # Accept stale client values (e.g. legacy "Auto") then sanitize in process.
                allow_custom_value=True,
            )
            profile = gr.Radio(
                list(PROFILE_TILES.keys()),
                value="Flagship",
                label="Tiles",
                show_label=False,
                elem_id="look-tiles",
            )

            gr.HTML('<div class="pl-step"><span>03 · Blend strength</span></div>')
            strength = gr.Slider(
                0.0, 1.0, value=0.9, step=0.05, label="Strength", show_label=False
            )

            with gr.Accordion("04 · Optical & tone fine-tuning", open=False):
                preset = gr.Dropdown(
                    presets, value="Balanced", label="Base preset"
                )

            status = gr.HTML(_empty_status())

            with gr.Row():
                run_btn = gr.Button("Re-run", variant="secondary", scale=1)
                cmp_btn = gr.Button("Compare All Looks", variant="primary", scale=3)

            with gr.Accordion("Look comparisons", open=False) as look_acc:
                gallery = gr.Gallery(
                    label="All looks",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="cover",
                    preview=False,
                    selected_index=None,
                    format="jpeg",
                    elem_id="look-strip",
                    show_label=False,
                )

        if examples:
            gr.Examples(
                examples=examples[:4],
                inputs=photo,
                label="Samples",
                examples_per_page=4,
            )

        outs = [
            upload_col,
            slider,
            before,
            after,
            diff_img,
            side_row,
            status,
            export_btn,
            st_before,
            st_after,
        ]
        inputs = [photo, look, strength, preset, view_mode]

        # Minimal progress: amber chip via CSS, not a ghost "NO SOURCE LOADED" overlay
        kw = {"show_progress": "minimal"}
        run_btn.click(process, inputs, outs, **kw)
        photo.change(process, inputs, outs, **kw)
        look.change(process, inputs, outs, **kw)
        strength.release(process, inputs, outs, **kw)
        preset.change(process, inputs, outs, **kw)

        # Tiles only set look — look.change runs process once (no .then)
        profile.change(tile_to_look, [profile], [look])

        view_mode.change(
            switch_view,
            [view_mode, st_before, st_after],
            [slider, side_row, diff_img],
        )
        sample_btn.click(load_sample, outputs=[photo])

        def compare_and_open(photo, strength, look, preset):
            import gradio as gr

            return compare_all(photo, strength, look, preset), gr.update(open=True)

        cmp_btn.click(
            compare_and_open,
            [photo, strength, look, preset],
            [gallery, look_acc],
        )

        demo.queue(default_concurrency_limit=1, max_size=8)

    return demo


def launch_kwargs(port: int) -> dict:
    return {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        "ssr_mode": False,
        "max_file_size": "15mb",
        "theme": _studio_theme(),
        "css": STUDIO_CSS,
        "js": "() => { document.body.classList.add('dark'); }",
    }


def main():
    import os

    demo = build_demo()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(**launch_kwargs(port))


if __name__ == "__main__":
    main()
