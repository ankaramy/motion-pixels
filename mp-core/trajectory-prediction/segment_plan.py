"""
segment_plan.py
---------------
Auto-segment a top-down plan image into walkable / obstacle regions
and compute distance maps for spatial encoding.

Two modes:
  threshold    (default) — Otsu threshold on grayscale.  Good for clean
                           plan drawings.
  google_maps             — Grayscale + Otsu + large morphological closing.
                           Detects the main open plaza by spatial continuity,
                           not pixel colour.  Robust against shadows, textures
                           and Google Maps labels.

Usage
-----
  python segment_plan.py --input_image PLAN_PATH
                         [--output_dir OUT_DIR]
                         [--scale METERS_PER_PIXEL]
                         [--mode {threshold,google_maps}]
                         [--ignore_textures true]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE        = Path(__file__).resolve().parent
MP_ROOT     = HERE.parent.parent
DEFAULT_OUT = MP_ROOT / "mp-data" / "processed" / "spatial"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true",  "yes", "1", "y", "t"): return True
    if s in ("false", "no",  "0", "n", "f"): return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {v!r}")


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill from corners and OR the inverted result back."""
    h, w   = mask.shape
    flood  = mask.copy()
    ff_msk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_msk, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def keep_top_components(mask: np.ndarray, min_ratio: float = 0.25,
                       min_area: int = 1000) -> np.ndarray:
    """Keep all components whose area >= min_ratio × largest, and >= min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(int(areas.max() * min_ratio), min_area)
    keep      = 1 + np.where(areas >= threshold)[0]
    out       = np.zeros_like(mask)
    for lbl in keep:
        out[labels == lbl] = 255
    return out


def outer_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def distance_maps(walkable: np.ndarray, contour, scale=None):
    dist_obs = cv2.distanceTransform(walkable, cv2.DIST_L2, 5)
    if contour is not None:
        bnd = np.full_like(walkable, 255)
        cv2.drawContours(bnd, [contour], -1, 0, 1)
        dist_bnd = cv2.distanceTransform(bnd, cv2.DIST_L2, 5)
    else:
        dist_bnd = np.zeros_like(dist_obs)
    if scale is not None:
        dist_obs = dist_obs * float(scale)
        dist_bnd = dist_bnd * float(scale)
    return dist_obs.astype(np.float32), dist_bnd.astype(np.float32)


def make_overlay(img, walkable, obstacles, contour):
    overlay = img.copy()
    g = np.zeros_like(img); g[walkable  > 0] = (0, 200, 0)
    r = np.zeros_like(img); r[obstacles > 0] = (0,   0, 200)
    overlay = cv2.addWeighted(overlay, 0.65, g, 0.35, 0)
    overlay = cv2.addWeighted(overlay, 1.00, r, 0.30, 0)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (255, 80, 0), 2)
    return overlay


def save_heatmap(dist, walkable, out_path: Path, unit: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    masked = np.where(walkable > 0, dist, np.nan)
    im = ax.imshow(masked, cmap="inferno")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"dist_to_obstacle ({unit})")
    ax.set_title("Distance to nearest obstacle")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: threshold (original pipeline)
# ---------------------------------------------------------------------------

def threshold_walkable(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold + flip so walkable = white (255)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def segment_threshold(img: np.ndarray):
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    walkable  = threshold_walkable(gray)
    walkable  = clean_mask(walkable)
    walkable  = keep_largest_component(walkable)
    obstacles = cv2.bitwise_not(walkable)
    contour   = outer_contour(walkable)
    return walkable, obstacles, contour, None


# ---------------------------------------------------------------------------
# Mode: google_maps
# ---------------------------------------------------------------------------

def remove_elongated_components(mask: np.ndarray,
                                max_aspect: float = 8.0) -> np.ndarray:
    """Drop connected components whose bounding-box aspect ratio exceeds max_aspect.
    Removes thin road markings, label underlines, and similar artefacts."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for lbl in range(1, n):
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect <= max_aspect:
            out[labels == lbl] = 255
    return out


def save_component_debug(img: np.ndarray, before: np.ndarray,
                         after: np.ndarray, out_path: Path):
    """Debug image: components present before the largest-keep step (cyan),
    and the final walkable mask (green outline) overlaid on the original."""
    dbg = img.copy()
    # Shade all surviving-before-filter components in translucent cyan
    cyan_layer = np.zeros_like(img)
    cyan_layer[before > 0] = (180, 180, 0)
    dbg = cv2.addWeighted(dbg, 0.75, cyan_layer, 0.25, 0)
    # Final walkable boundary in bright green
    contours, _ = cv2.findContours(after, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dbg, contours, -1, (0, 220, 0), 2)
    n_comp_before = cv2.connectedComponentsWithStats(before)[0] - 1
    label = (f"components after close/clean: {n_comp_before}   "
             f"kept (walkable): 1  |  green = final boundary")
    cv2.rectangle(dbg, (0, 0), (dbg.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(dbg, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), dbg)


def segment_google_maps(img: np.ndarray,
                        ignore_textures: bool,
                        debug_path: Path):
    # 1. Grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur to suppress fine texture and noise.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Otsu threshold → binary mask.
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Invert if needed so the dominant open space is white (walkable).
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # 5. Large morphological closing merges plaza pixels separated by shadows,
    #    grid lines, or thin markings into one solid region.
    big_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, big_k, iterations=2)

    # 6. Remove small components (noise, text blobs, isolated specks).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cleaned = np.zeros_like(closed)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= 2000:
            cleaned[labels == lbl] = 255

    # 7. Keep the single largest connected component as the walkable plaza.
    walkable = keep_largest_component(cleaned)

    # 8. Optional: remove any remaining elongated thin strips (road markings,
    #    label underlines) whose bounding-box aspect ratio exceeds 8.
    if ignore_textures:
        walkable = remove_elongated_components(walkable, max_aspect=8.0)

    save_component_debug(img, cleaned, walkable, debug_path)

    # 9. obstacle_mask = everything outside the walkable area.
    obstacles = cv2.bitwise_not(walkable)
    contour   = outer_contour(walkable)

    return walkable, obstacles, contour, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment a plan image into walkable/obstacle and "
                    "compute distance maps.")
    parser.add_argument("--input_image", type=Path, required=True,
                        help="path to plan/top-down image")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: mp-data/processed/spatial/)")
    parser.add_argument("--scale", type=float, default=None,
                        help="meters per pixel (optional, default = pixels)")
    parser.add_argument("--mode", choices=("threshold", "google_maps"),
                        default="threshold",
                        help="segmentation pipeline (default: threshold)")
    parser.add_argument("--ignore_textures", type=str2bool, default=True,
                        help="[google_maps] remove elongated thin regions "
                             "such as road markings or label strokes "
                             "(default true)")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = "m" if args.scale else "px"

    print(f"[INFO]  Plan  : {args.input_image}")
    print(f"[INFO]  Out   : {out_dir}")
    print(f"[INFO]  Mode  : {args.mode}")
    if args.scale:
        print(f"[INFO]  Scale : {args.scale} m/px")
    else:
        print(f"[INFO]  Scale : pixels (no scale provided)")
    if args.mode == "google_maps":
        print(f"[INFO]  Ignore textures: {args.ignore_textures}")

    img = load_image(args.input_image)

    debug_path = out_dir / "obstacle_candidates.png"
    if args.mode == "google_maps":
        walkable, obstacles, contour, info = segment_google_maps(
            img,
            ignore_textures = args.ignore_textures,
            debug_path      = debug_path,
        )
    else:
        walkable, obstacles, contour, info = segment_threshold(img)

    dist_obs, dist_bnd = distance_maps(walkable, contour, args.scale)

    cv2.imwrite(str(out_dir / "walkable_mask.png"), walkable)
    cv2.imwrite(str(out_dir / "obstacle_mask.png"), obstacles)
    np.save(out_dir / "dist_to_obstacle.npy", dist_obs)
    np.save(out_dir / "dist_to_boundary.npy", dist_bnd)

    overlay = make_overlay(img, walkable, obstacles, contour)
    cv2.imwrite(str(out_dir / "segmentation_overlay.png"), overlay)

    walk_rgb = cv2.cvtColor(walkable,  cv2.COLOR_GRAY2BGR)
    obs_rgb  = cv2.cvtColor(obstacles, cv2.COLOR_GRAY2BGR)
    side     = np.hstack([img, walk_rgb, obs_rgb, overlay])
    cv2.imwrite(str(out_dir / "side_by_side.png"), side)

    save_heatmap(dist_obs, walkable, out_dir / "dist_heatmap.png", unit)

    print(f"\n[OK]   Walkable pixels      : {(walkable  > 0).sum():,}")
    print(f"[OK]   Obstacle pixels      : {(obstacles > 0).sum():,}")
    print(f"[OK]   Boundary contour pts : "
          f"{len(contour) if contour is not None else 0:,}")
    print(f"[OK]   Max dist_to_obstacle : {dist_obs.max():.3f} {unit}")
    print(f"[OK]   Max dist_to_boundary : {dist_bnd.max():.3f} {unit}")
    if args.mode == "google_maps":
        print(f"[OK]   Debug image          : {debug_path}")
    print(f"[OK]   Outputs              : {out_dir}")


if __name__ == "__main__":
    main()
