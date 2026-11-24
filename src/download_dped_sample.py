"""Download DPED sample photos and install as src/tgt pairs for this project."""

from __future__ import annotations

import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .paths import RAW_PAIRS, ROOT

EXTERNAL = ROOT / "data" / "external" / "dped_sample"
ARCHIVE = EXTERNAL / "sample_images.gz"
EXTRACTED = EXTERNAL / "extracted"

SAMPLE_URL = "https://download.ai-benchmark.com/s/9RqJ5JZPDggMoyo/download/sample_images.gz"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already downloaded: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"Downloading {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        total = r.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total_i:
                print(f"\r  {done/1e6:.1f}/{total_i/1e6:.1f} MB ({100*done/total_i:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {done/1e6:.1f} MB", end="", flush=True)
    print(f"\nSaved {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def extract(archive: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Detect format from magic bytes
    with open(archive, "rb") as f:
        magic = f.read(8)

    # ZIP
    if magic[:2] == b"PK":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest)
        print("Extracted as ZIP")
        return

    # gzip / tar.gz
    if magic[:2] == b"\x1f\x8b":
        # try tar.gz first
        try:
            with tarfile.open(archive, "r:gz") as t:
                t.extractall(dest)
            print("Extracted as tar.gz")
            return
        except tarfile.ReadError:
            out = dest / "sample_images.bin"
            with gzip.open(archive, "rb") as g, open(out, "wb") as f:
                shutil.copyfileobj(g, f)
            # maybe the gunzipped file is a tar/zip
            with open(out, "rb") as f:
                m2 = f.read(8)
            if m2[:2] == b"PK":
                with zipfile.ZipFile(out, "r") as z:
                    z.extractall(dest)
                print("Gunzipped then extracted ZIP")
                return
            try:
                with tarfile.open(out, "r:") as t:
                    t.extractall(dest)
                print("Gunzipped then extracted TAR")
                return
            except tarfile.ReadError:
                print(f"Gunzipped to {out}; unknown inner format")
                return

    # plain tar
    try:
        with tarfile.open(archive, "r:") as t:
            t.extractall(dest)
        print("Extracted as TAR")
        return
    except tarfile.ReadError:
        pass

    raise RuntimeError(f"Unknown archive format, magic={magic!r}")


def collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _is_patch(path: Path) -> bool:
    return "patches" in [x.lower() for x in path.parts]


def _is_original(path: Path) -> bool:
    parts = [x.lower() for x in path.parts]
    return "original_images" in parts or "original" in parts


def pair_from_folder_structure(files: list[Path], prefer: str = "patches") -> list[tuple[Path, Path]]:
    """Pair by shared stem across phone/canon folders.

    prefer='patches' uses already-aligned 100x100 DPED patches (best for tone fitting).
    prefer='originals' uses full-resolution photos (need SIFT alignment).
    """
    phone_names = ("iphone", "sony", "blackberry")

    def build(subset: list[Path]) -> list[tuple[Path, Path]]:
        canon_map: dict[str, Path] = {}
        phone_maps: dict[str, dict[str, Path]] = {p: {} for p in phone_names}
        for p in subset:
            parts = [x.lower() for x in p.parts]
            stem = p.stem
            if "canon" in parts or "dslr" in parts:
                # Patch trees like patches/iphone/canon stay device-scoped
                phone_scope = next((ph for ph in phone_names if ph in parts), "_")
                canon_map[f"{phone_scope}:{stem}"] = p
                canon_map.setdefault(stem, p)
                continue
            for phone in phone_names:
                if phone in parts:
                    phone_maps[phone][stem] = p
                    break
        pairs: list[tuple[Path, Path]] = []
        for phone in phone_names:
            items = sorted(
                phone_maps[phone].items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0],
            )
            for stem, src in items:
                tgt = canon_map.get(f"{phone}:{stem}") or canon_map.get(stem)
                if tgt is not None:
                    pairs.append((src, tgt))
        return pairs

    patch_files = [p for p in files if _is_patch(p)]
    orig_files = [p for p in files if _is_original(p)]
    other = [p for p in files if not _is_patch(p) and not _is_original(p)]

    ordered = [patch_files, orig_files, other] if prefer == "patches" else [orig_files, patch_files, other]
    for subset in ordered:
        pairs = build(subset)
        if pairs:
            return pairs
    return []


def pair_by_name_tokens(files: list[Path]) -> list[tuple[Path, Path]]:
    import re

    buckets: dict[str, dict[str, Path]] = {}
    num = re.compile(r"(\d+)")
    for p in files:
        name = p.stem.lower()
        m = num.findall(name)
        key = m[-1] if m else name
        if any(k in name for k in ("iphone", "blackberry", "sony", "phone", "orig", "original", "input")) and "canon" not in name:
            buckets.setdefault(key, {})["src"] = p
        elif any(k in name for k in ("canon", "dslr", "target", "ref")):
            buckets.setdefault(key, {})["tgt"] = p
    pairs = []
    for d in buckets.values():
        if "src" in d and "tgt" in d:
            pairs.append((d["src"], d["tgt"]))
    return pairs


def install_pairs(pairs: list[tuple[Path, Path]], limit: int = 48) -> None:
    RAW_PAIRS.mkdir(parents=True, exist_ok=True)

    # Clear previous demo/synthetic pairs so DPED is clean
    for pat in ("scene*_src.*", "scene*_tgt.*", "dped*_src.*", "dped*_tgt.*"):
        for p in RAW_PAIRS.glob(pat):
            p.unlink(missing_ok=True)

    n = min(limit, len(pairs))
    for i, (src, tgt) in enumerate(pairs[:n], start=1):
        sid = f"dped{i:02d}"
        src_out = RAW_PAIRS / f"{sid}_src{src.suffix.lower()}"
        tgt_out = RAW_PAIRS / f"{sid}_tgt{tgt.suffix.lower()}"
        shutil.copy2(src, src_out)
        shutil.copy2(tgt, tgt_out)
        kind = "patch" if _is_patch(src) else "original"
        print(f"{sid} [{kind}]: {src.parent.name}/{src.name}  ->  {tgt.parent.name}/{tgt.name}")
    print(f"Installed {n} DPED pairs into {RAW_PAIRS}")


def main():
    download(SAMPLE_URL, ARCHIVE)
    extract(ARCHIVE, EXTRACTED)
    files = collect_images(EXTRACTED)
    print(f"Found {len(files)} images under {EXTRACTED}")
    for p in files[:25]:
        print(" ", p.relative_to(EXTRACTED))
    if len(files) > 25:
        print(f"  ... +{len(files)-25} more")

    pairs = pair_from_folder_structure(files, prefer="patches")
    if not pairs:
        pairs = pair_by_name_tokens(files)
    # Prefer iPhone patches for the project story (phone → Canon look)
    iphone_pairs = [
        (a, b)
        for a, b in pairs
        if _is_patch(a) and "iphone" in [x.lower() for x in a.parts]
    ]
    if iphone_pairs:
        pairs = iphone_pairs
    print(f"Paired {len(pairs)} phone↔Canon samples (iPhone patches preferred)")
    if not pairs:
        raise SystemExit("Could not pair images automatically. Inspect data/external/dped_sample/extracted/")

    # Install every iPhone patch available (better TinyCNN / ladder training)
    install_pairs(pairs, limit=max(len(pairs), 1))
    print("\nNext commands:")
    print("  python -m src.align_pairs")
    print("  python -m src.fit_models")
    print("  python -m src.evaluate")
    print("  python -m src.make_figures")


if __name__ == "__main__":
    main()
