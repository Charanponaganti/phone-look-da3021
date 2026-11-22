from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PAIRS = ROOT / "data" / "raw" / "pairs"
RAW_BURSTS = ROOT / "data" / "raw" / "bursts"
ALIGNED = ROOT / "data" / "aligned"
MODELS = ROOT / "outputs" / "models"
FIGURES = ROOT / "outputs" / "figures"
ENHANCED = ROOT / "outputs" / "enhanced"

for p in (RAW_PAIRS, RAW_BURSTS, ALIGNED, MODELS, FIGURES, ENHANCED):
    p.mkdir(parents=True, exist_ok=True)
