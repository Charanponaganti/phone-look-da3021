"""Align src→tgt pairs with SIFT + RANSAC homography (works for different FOVs)."""

from __future__ import annotations

import json

import cv2
import numpy as np

from .io_utils import list_pair_ids, pair_paths, read_bgr, write_bgr
from .paths import ALIGNED


def _gray_for_features(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Local contrast helps phone/DSLR pairs in flat lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def _clip_mask(src_bgr: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    src_clip = (np.min(src_bgr, axis=2) <= 2) | (np.max(src_bgr, axis=2) >= 253)
    tgt_clip = (np.min(warped, axis=2) <= 2) | (np.max(warped, axis=2) >= 253)
    out = mask.copy()
    out[src_clip | tgt_clip] = 0
    return out


def align_identity(src_bgr: np.ndarray, tgt_bgr: np.ndarray):
    """Same-size pairs (e.g. DPED patches) — no geometric warp needed."""
    if src_bgr.shape[:2] != tgt_bgr.shape[:2]:
        # Resize tgt to src canvas for already-corresponding crops of different encoding sizes
        tgt_bgr = cv2.resize(tgt_bgr, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    mask = np.ones(src_bgr.shape[:2], np.uint8)
    mask = _clip_mask(src_bgr, tgt_bgr, mask)
    min_pix = max(500, int(0.2 * mask.size))
    if int(mask.sum()) < min_pix:
        raise RuntimeError("Identity overlap mask too small")
    return tgt_bgr, mask, 1.0, 0


def align_pair(src_bgr: np.ndarray, tgt_bgr: np.ndarray):
    """Warp tgt onto src geometry. Uses identity when shapes already match."""
    # DPED patches and equal-size captures: skip feature matching
    if src_bgr.shape[:2] == tgt_bgr.shape[:2]:
        return align_identity(src_bgr, tgt_bgr)

    src_g = _gray_for_features(src_bgr)
    tgt_g = _gray_for_features(tgt_bgr)

    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(src_g, None)
    k2, d2 = sift.detectAndCompute(tgt_g, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        raise RuntimeError("Not enough SIFT features")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    matches = flann.knnMatch(d1, d2, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) < 10:
        raise RuntimeError(f"Only {len(good)} good matches")

    pts_src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_tgt = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography maps tgt points -> src points
    H, inl = cv2.findHomography(pts_tgt, pts_src, cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)
    if H is None:
        raise RuntimeError("Homography failed")

    h, w = src_bgr.shape[:2]
    warped = cv2.warpPerspective(tgt_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
    ones = np.ones(tgt_bgr.shape[:2], np.uint8) * 255
    mask = cv2.warpPerspective(ones, H, (w, h))
    mask = (mask > 200).astype(np.uint8)

    # Erode a little to drop border interpolation junk
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = _clip_mask(src_bgr, warped, mask)

    if int(mask.sum()) < 5000:
        raise RuntimeError("Overlap mask too small after warp")

    inlier_ratio = float(inl.sum() / max(len(good), 1)) if inl is not None else 0.0
    return warped, mask, inlier_ratio, len(good)


def main():
    ids = list_pair_ids()
    if not ids:
        print("No pairs found in data/raw/pairs/")
        print("Add files like scene01_src.jpg and scene01_tgt.jpg then rerun.")
        return

    report = {}
    print(f"Found {len(ids)} pairs")
    for sid in ids:
        src_path, tgt_path = pair_paths(sid)
        # Higher res helps SIFT on DPED originals
        src = read_bgr(src_path, max_side=2000)
        tgt = read_bgr(tgt_path, max_side=2000)
        try:
            warped, mask, inl, n_good = align_pair(src, tgt)
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            report[sid] = {"ok": False, "error": str(e)}
            continue

        out = ALIGNED / sid
        out.mkdir(parents=True, exist_ok=True)
        write_bgr(out / "src.png", src)
        write_bgr(out / "tgt_warped.png", warped)
        cv2.imwrite(str(out / "mask.png"), mask * 255)
        report[sid] = {
            "ok": True,
            "inlier_ratio": round(inl, 3),
            "good_matches": n_good,
            "valid_pixels": int(mask.sum()),
        }
        print(f"[OK] {sid}: inliers={inl:.2f}, matches={n_good}, pixels={int(mask.sum())}")

    with open(ALIGNED / "align_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    ok_n = sum(1 for v in report.values() if v.get("ok"))
    print(f"Aligned {ok_n}/{len(ids)} pairs")
    print(f"Wrote {ALIGNED / 'align_report.json'}")


if __name__ == "__main__":
    main()
