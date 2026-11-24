---
title: Phone Look DA3021
emoji: 📷
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Auto phone→Canon look — course transforms + TinyResCNN
suggested_hardware: cpu-basic
---

# Phone Look Studio — DA3021

CPU pipeline that learns a better-camera colour/tone look from paired photos, then serves it in a **Gradio Studio** app.

## Live demo

Deploy on **Render** (free): connect this GitHub repo as a Docker web service (`render.yaml` + `Dockerfile`).  
Hugging Face Gradio Spaces now require PRO — Render is the free permanent host for this app.

## What you get

**Auto cascade (default)** — detects fog / night / sunset / portrait / backlight / indoor / vivid / detail / daylight and tunes:
1. Soft dehaze (when hazy)  
2. Exposure / CLAHE  
3. Course colour (tone L + colour matrix)  
4. TinyResCNN local residual  
5. L-clarity + chroma clamps  

**Single-model ladder** for the course story: tone → matrix → root-poly → MLPs → TinyCNN  

## Local run

```bash
git clone https://github.com/Charanponaganti/phone-look-da3021.git
cd phone-look-da3021
pip install -r requirements.txt
python app.py
```

## Retrain

```bash
python -m src.align_pairs
python -m src.fit_models
python -m src.evaluate
python -m src.make_figures
```

Own pairs: `data/raw/pairs/sceneXX_src.jpg` + `sceneXX_tgt.jpg` (see `Capture_Checklist.pdf`).
