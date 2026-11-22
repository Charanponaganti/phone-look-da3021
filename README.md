# Phone Look — DA3021

CPU pipeline that learns a “better camera” colour/tone look from paired photos (phone → reference), then serves it in a Gradio upload app.

Trained here on **DPED iPhone→Canon** sample patches. Swap in your own Night/HDR pairs anytime.

## Quick start (app)

```bash
git clone https://github.com/Charanponaganti/phone-look-da3021.git
cd phone-look-da3021
pip install -r requirements.txt
python -m src.app
```

Open the local Gradio URL (default `http://127.0.0.1:7860`), upload a photo, pick a model.

Shipped default model: `tone_curves.json` (best held-out ΔE on the DPED sample split).

## Full pipeline

```bash
# optional: refresh DPED sample pairs
python -m src.download_dped_sample

python -m src.align_pairs
python -m src.fit_models
python -m src.evaluate
python -m src.make_figures
python -m src.app
```

Or shoot your own pairs into `data/raw/pairs/`:

```
scene01_src.jpg   # normal phone
scene01_tgt.jpg   # Night / HDR / better camera
```

See `Capture_Checklist.pdf` for a capture guide.

## Layout

| Path | Role |
|------|------|
| `src/` | align, fit, evaluate, apply, Gradio app |
| `data/raw/pairs/` | `*_src` / `*_tgt` training pairs |
| `outputs/models/` | trained JSON models + eval |
| `outputs/figures/` | triptychs / ranking plots (local) |

## Models

Ladder fitted in Lab: tone curves → 3×3 (+bias) matrix → root-polynomial + ridge.
Apply any photo with:

```bash
python -m src.apply path/to/photo.jpg
```
