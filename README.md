---
title: Phone Look DA3021
emoji: 📷
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: CPU phone→Canon colour/tone look from DPED pairs
---

# Phone Look — DA3021

CPU pipeline that learns a “better camera” colour/tone look from paired photos (phone → reference), then serves it in a Gradio upload app.

Trained on **DPED iPhone→Canon** sample patches. Swap in your own Night/HDR pairs anytime.

## Live demo

Deployed as a Hugging Face Space (see repo / Space link after deploy).

## Quick start (local)

```bash
git clone https://github.com/Charanponaganti/phone-look-da3021.git
cd phone-look-da3021
pip install -r requirements.txt
python app.py
```

Open the Gradio URL, upload a photo, pick a model.

Shipped default model: `tone_curves.json`.

## Full pipeline

```bash
python -m src.download_dped_sample   # optional
python -m src.align_pairs
python -m src.fit_models
python -m src.evaluate
python -m src.make_figures
python app.py
```

Own pairs go in `data/raw/pairs/` as `sceneXX_src.jpg` + `sceneXX_tgt.jpg`. See `Capture_Checklist.pdf`.

## Layout

| Path | Role |
|------|------|
| `app.py` | Gradio / Spaces entrypoint |
| `src/` | align, fit, evaluate, apply, app logic |
| `data/raw/pairs/` | training pairs |
| `outputs/models/` | trained JSON models |

## CLI enhance

```bash
python -m src.apply path/to/photo.jpg
```
