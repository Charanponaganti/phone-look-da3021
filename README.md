---
title: Phone Look DA3021
emoji: 📷
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Auto phone→Canon look — course transforms + TinyResCNN
---

# Phone Look Studio — DA3021

CPU pipeline that learns a better-camera colour/tone look from paired photos, then serves it in a **Gradio Studio** app.

## What you get

**Auto cascade (default in the app)**
1. Mild L exposure stretch  
2. Course colour (tone curves on L + colour matrix chroma)  
3. TinyResCNN local residual (DPED-style, CPU)  
4. L-channel clarity (unsharp)  
5. Chroma/luma residual clamps + adaptive strength on high-saturation images  

**Single-model ladder** (for the course story): tone → matrix → root-poly → Lab MLP → root-poly MLP → TinyCNN  

## Quick start

```bash
git clone https://github.com/Charanponaganti/phone-look-da3021.git
cd phone-look-da3021
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL. Use **Auto + Balanced** for demos; drop Strength if a sunset still looks strong.

## Retrain

```bash
python -m src.download_dped_sample   # optional refresh
python -m src.align_pairs
python -m src.fit_models
python -m src.evaluate               # composite = ΔE + chroma penalty; ranks Auto too
python -m src.make_figures
python app.py
```

Own pairs: `data/raw/pairs/sceneXX_src.jpg` + `sceneXX_tgt.jpg` (see `Capture_Checklist.pdf`).

## Layout

| Path | Role |
|------|------|
| `app.py` / `src/app.py` | Gradio Studio |
| `src/pipeline.py` | Auto cascade |
| `src/tiny_cnn.py` | TinyResCNN v2 |
| `src/models.py` | Course transforms + MLPs |
| `outputs/models/` | Trained JSON + `tiny_cnn.pt` |
| `assets/examples/` | Demo thumbs |

## CLI

```bash
python -m src.apply path/to/photo.jpg --mode auto --preset Balanced
python -m src.apply path/to/photo.jpg --mode model -m outputs/models/tone_curves.json
```
