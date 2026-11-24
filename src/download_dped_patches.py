"""
Download full DPED training patches and install a deadline-friendly subset.

Default: 8000 random iPhone↔Canon training pairs (~smart path for DA3021).

  python -m src.download_dped_patches
  python -m src.download_dped_patches --n 10000
  python -m src.download_dped_patches --skip-download   # if dped.gz already present
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from .download_dped_sample import download, extract
from .paths import RAW_PAIRS, ROOT

EXTERNAL = ROOT / "data" / "external" / "dped_patches"
ARCHIVE = EXTERNAL / "dped.gz"
EXTRACTED = EXTERNAL / "extracted"
PATCHES_URL = "https://download.ai-benchmark.com/s/dWZz5RD9kHpedix/download/dped.gz"


def _find_iphone_train(root: Path) -> tuple[Path, Path] | None:
    """Return (iphone_dir, canon_dir) for training patches."""
    cands: list[tuple[Path, Path]] = []
    for phone in root.rglob("*"):
        if not phone.is_dir():
            continue
        if phone.name.lower() != "iphone":
            continue
        # Prefer .../training_data/iphone + sibling canon
        parent = phone.parent
        canon = parent / "canon"
        if canon.is_dir() and "training" in str(parent).lower().replace("\\", "/"):
            cands.append((phone, canon))
    if cands:
        # Prefer deepest / training_data path
        cands.sort(key=lambda p: ("training" not in str(p[0]).lower(), -len(str(p[0]))))
        return cands[0]
    return None


def _pair_files(phone_dir: Path, canon_dir: Path) -> list[tuple[Path, Path]]:
    phones = {p.stem: p for p in phone_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    canons = {p.stem: p for p in canon_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    keys = sorted(set(phones) & set(canons), key=lambda k: int(k) if k.isdigit() else k)
    return [(phones[k], canons[k]) for k in keys]


def install_subset(pairs: list[tuple[Path, Path]], n: int, seed: int = 42) -> int:
    RAW_PAIRS.mkdir(parents=True, exist_ok=True)
    # Keep existing non-patch own captures (scene*) if present
    rng = random.Random(seed)
    chosen = pairs if n <= 0 or n >= len(pairs) else rng.sample(pairs, n)
    # Wipe old dped* patch installs so we don't mix sizes inconsistently
    for old in RAW_PAIRS.glob("dped*_src.*"):
        sid = old.name.split("_src")[0]
        for p in RAW_PAIRS.glob(f"{sid}_*"):
            p.unlink(missing_ok=True)

    for i, (src, tgt) in enumerate(chosen, start=1):
        sid = f"dped{i:04d}"
        shutil.copy2(src, RAW_PAIRS / f"{sid}_src{src.suffix.lower()}")
        shutil.copy2(tgt, RAW_PAIRS / f"{sid}_tgt{tgt.suffix.lower()}")
        if i % 500 == 0 or i == len(chosen):
            print(f"  installed {i}/{len(chosen)}")
    return len(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="Number of iPhone↔Canon pairs to install (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download(PATCHES_URL, ARCHIVE)
    elif not ARCHIVE.exists():
        raise SystemExit(f"Missing {ARCHIVE}; run without --skip-download")

    if not args.skip_extract or not EXTRACTED.exists():
        print("Extracting (this can take a while)…")
        extract(ARCHIVE, EXTRACTED)

    found = _find_iphone_train(EXTRACTED)
    if not found:
        raise SystemExit(f"Could not find iphone/canon training folders under {EXTRACTED}")
    phone_dir, canon_dir = found
    print(f"Using phone={phone_dir}")
    print(f"Using canon={canon_dir}")
    pairs = _pair_files(phone_dir, canon_dir)
    print(f"Found {len(pairs)} aligned iPhone↔Canon training patches")
    if not pairs:
        raise SystemExit("No paired patch filenames matched")

    n = install_subset(pairs, args.n, seed=args.seed)
    print(f"Installed {n} pairs into {RAW_PAIRS}")
    print("Next:")
    print("  python -m src.align_pairs")
    print("  python -m src.fit_models")
    print("  python -m src.evaluate")


if __name__ == "__main__":
    main()
