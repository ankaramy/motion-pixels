"""
encode_space.py
---------------
Enriches a trajectory CSV with spatial-context columns by letting you click
three sets of landmarks on the floor-plan image.

Coordinate conversion
---------------------
After the user clicks on top-down.png, the clicked plan-image pixel
coordinates are converted to real-world metric coordinates using the linear
scale-and-origin formula from calib.json:

    world_x = (click_x - plan_origin_pixel[0]) * meters_per_plan_pixel
    world_y = (click_y - plan_origin_pixel[1]) * meters_per_plan_pixel
    if invert_plan_y: world_y = -world_y

NOTE: calib.json also contains a homography_matrix, but that matrix maps
*video-frame pixels* to world metres — it must NOT be applied to plan-image
clicks.  The plan image uses a simple orthographic (linear) mapping defined
by meters_per_plan_pixel and plan_origin_pixel.

Dimension check
---------------
Before any conversion the script checks that top-down.png has the same pixel
dimensions as the image that was used during calibration.  If they differ,
encoding is aborted with a clear error so you can supply the correct image.

Auto-seeding
------------
If bottleneck_cells.csv is found (trajectory-extraction output), the top-N slowest
cells are pre-loaded as obstacle points so you do not have to click them all.

Controls
--------
  Left-click  — add a point           Right-click — remove last point (undo)
  N           — confirm phase, next   Q           — finish (entrance phase)

Output
------
  outputs/trajectories_encoded.csv
    original columns + dist_to_obstacle, dist_to_boundary, dist_to_entrance
    all distances in metres

Usage
-----
  python encode_space.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
MP_DATA      = MP_ROOT / "mp-data"
TRAJ_OUT     = MP_DATA / "outputs" / "tracking"
BEHAVIOR_OUT = MP_DATA / "outputs" / "behavior"
PROCESSED    = MP_DATA / "processed" / "encoded"

_CANDIDATES = [
    TRAJ_OUT / "trajectories_world_space.csv",
    TRAJ_OUT / "trajectories_world.csv",
    PROCESSED / "trajectories_world.csv",
    HERE / "outputs" / "trajectories_output.csv",  # local fallback
]
CSV_IN = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])

_BN_CANDIDATES = [
    BEHAVIOR_OUT / "bottlenecks" / "bottleneck_cells.csv",
    HERE / "outputs" / "bottleneck_cells.csv",  # local fallback
]
BOTTLENECK_CSV = next((p for p in _BN_CANDIDATES if p.exists()), None)

_CALIB_CANDIDATES = [
    MP_DATA / "raw" / "calibration" / "calib_skate1.json",
    HERE / "calib.json",  # local fallback
]
CALIB_JSON = next((p for p in _CALIB_CANDIDATES if p.exists()), None)

IMAGE_PATH = MP_DATA / "raw" / "images" / "top-down.png"
CSV_OUT    = PROCESSED / "trajectories_encoded.csv"

AUTO_SEED_TOP_N = 15

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
COLOUR_OBSTACLE  = (0,   80, 255)
COLOUR_BOUNDARY  = (0,  200,   0)
COLOUR_ENTRANCE  = (255, 180,   0)
COLOUR_AUTOSEED  = (0,  200, 255)
COLOUR_TRAJ      = (180, 255, 180)   # faint green — trajectory reference dots
COLOUR_CALIB_PT  = (255, 255,   0)   # yellow — calibration reference crosses
POINT_RADIUS     = 6
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Image-dimension check
# ---------------------------------------------------------------------------

def check_image_dimensions(calib: dict, current_image: np.ndarray) -> None:
    """
    Verify that the current floor-plan image (top-down.png) has the same pixel
    dimensions as the image that was used during calibration.

    calib.json stores the plan image filename under 'plan_image'.  We search
    for that file in a few expected locations.  If found, dimensions are
    compared exactly.  If the file cannot be found we print a warning and
    continue — the plan_points_px coordinates still act as a sanity bound.

    Raises SystemExit if a size mismatch is detected.
    """
    cur_h, cur_w = current_image.shape[:2]
    print(f"\n[CHECK] Current image  : {IMAGE_PATH.name}  {cur_w} × {cur_h} px")

    plan_image_name = calib.get("plan_image", "")
    search_dirs = [HERE, MP_DATA / "raw" / "images", MP_ROOT]
    ref_path = None
    for d in search_dirs:
        candidate = d / plan_image_name
        if candidate.exists():
            ref_path = candidate
            break

    if ref_path is not None:
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"[WARN]  Could not read reference image: {ref_path}")
        else:
            ref_h, ref_w = ref_img.shape[:2]
            print(f"[CHECK] Calibration image: {ref_path.name}  {ref_w} × {ref_h} px")
            if cur_w != ref_w or cur_h != ref_h:
                sys.exit(
                    f"\n[ERROR] Image dimension mismatch!\n"
                    f"        top-down.png used now :  {cur_w} × {cur_h} px\n"
                    f"        Calibration image     :  {ref_w} × {ref_h} px  "
                    f"({ref_path.name})\n\n"
                    f"        top-down.png is NOT in the same pixel space as the "
                    f"calibration.\n"
                    f"        Replace top-down.png with the image that was used "
                    f"during calibration, or re-run calibrate_homography_interactive.py "
                    f"with the current image."
                )
            print("[CHECK] Dimensions match — proceeding.\n")
    else:
        # Can't find the reference file — use plan_points_px as a soft check
        max_plan_x = max(p[0] for p in calib.get("plan_points_px", [[0, 0]]))
        max_plan_y = max(p[1] for p in calib.get("plan_points_px", [[0, 0]]))
        print(f"[WARN]  Reference image '{plan_image_name}' not found — "
              f"cannot verify dimensions.")
        print(f"        Calibration plan_points_px reach up to "
              f"({max_plan_x:.0f}, {max_plan_y:.0f}) px.")
        if cur_w <= max_plan_x or cur_h <= max_plan_y:
            sys.exit(
                f"\n[ERROR] top-down.png ({cur_w}×{cur_h}) is smaller than the "
                f"calibration point extents ({max_plan_x:.0f}×{max_plan_y:.0f}).\n"
                f"        This image cannot be the one used during calibration."
            )
        print(f"        Current image is large enough to contain all calibration "
              f"points — continuing with caution.\n")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def world_to_plan_px(wx, wy, mpp, origin, invert_y):
    ox, oy = origin
    px_x = (np.asarray(wx, dtype=np.float64) / mpp + ox).astype(int)
    if invert_y:
        px_y = (-np.asarray(wy, dtype=np.float64) / mpp + oy).astype(int)
    else:
        px_y = (np.asarray(wy, dtype=np.float64) / mpp + oy).astype(int)
    return px_x, px_y


def build_reference_layer(image: np.ndarray,
                           df: pd.DataFrame,
                           calib: dict,
                           mpp: float,
                           origin: list,
                           invert_y: bool) -> np.ndarray:
    """
    Burn a faint reference layer onto a copy of the plan image showing:
      - Tiny faint-green dots at every trajectory world position, so the user
        can see exactly where pedestrians walked and orient their clicks.
      - Small yellow crosses at the 6 calibration correspondence points
        (plan_points_px from calib.json), as an alignment sanity-check.

    Returns a new image (the original is not modified).
    """
    h, w  = image.shape[:2]
    layer = image.copy()

    # --- Trajectory dots ---
    if "world_x" in df.columns and "world_y" in df.columns:
        wx = df["world_x"].to_numpy(dtype=np.float64)
        wy = df["world_y"].to_numpy(dtype=np.float64)
        px_x, px_y = world_to_plan_px(wx, wy, mpp, origin, invert_y)
        for x, y in zip(px_x.tolist(), px_y.tolist()):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(layer, (x, y), 1, COLOUR_TRAJ, -1)

    # --- Calibration crosses ---
    for pp in calib.get("plan_points_px", []):
        cx, cy = int(round(pp[0])), int(round(pp[1]))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.drawMarker(layer, (cx, cy), COLOUR_CALIB_PT,
                           cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

    # Blend: 35% reference layer, 65% original image so background stays clear
    return cv2.addWeighted(layer, 0.35, image, 0.65, 0)


# ---------------------------------------------------------------------------
# Nearest-neighbour distances (vectorised, world metres)
# ---------------------------------------------------------------------------

def nearest_distances(query_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """
    For every point in query_xy return its Euclidean distance (in metres) to
    the nearest point in ref_xy.  Returns NaN if ref_xy is empty.
    """
    if len(ref_xy) == 0:
        return np.full(len(query_xy), np.nan)
    diff = query_xy[:, np.newaxis, :] - ref_xy[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)


# ---------------------------------------------------------------------------
# Interactive point collection
# ---------------------------------------------------------------------------

def collect_points_interactive(
    image: np.ndarray,
    ref_image: np.ndarray,
    auto_obstacle_px: list,
    auto_obstacle_world: np.ndarray,
    mpp: float,
    origin: list,
    invert_y: bool,
) -> tuple:
    """
    Opens the floor-plan in an OpenCV window and collects three sets of
    landmark clicks.

    The window shows a reference layer (trajectory dots + calibration crosses)
    under the click markers so the user can orient themselves correctly.

    Auto-seeded obstacle points are displayed as cyan rings showing where
    pedestrian bottlenecks occurred.  They are NOT pre-included as obstacles;
    press A to accept them into the obstacle set or ignore them and click your
    own physical barriers.  C clears all manually clicked points in the current
    phase.

    Conversion formula for user clicks (from calib.json):
        world_x = (px - origin[0]) * mpp
        world_y = (py - origin[1]) * mpp   (negated if invert_y is True)

    Parameters
    ----------
    image               : the floor-plan image (top-down.png)
    ref_image           : image with trajectory + calibration reference overlay
    auto_obstacle_px    : list of (x, y) plan-pixel positions — bottleneck rings
    auto_obstacle_world : (N, 2) ndarray of world-metre coords for the above;
                          included in obs_world only if user presses A
    mpp / origin / invert_y : calibration parameters from calib.json

    Returns
    -------
    obs_world, bnd_world, ent_world : each an ndarray of shape (M, 2) in
                                      world metres, or (0, 2) if skipped
    """
    all_points_px      = [[], [], []]
    phase              = [0]
    use_auto_seeds     = [False]   # toggled by pressing A in obstacle phase
    show_ref           = [True]    # toggle reference overlay with R

    colours = [COLOUR_OBSTACLE, COLOUR_BOUNDARY, COLOUR_ENTRANCE]
    labels  = ["OBSTACLE", "BOUNDARY", "ENTRANCE"]
    tips    = [
        "Left=add  Right=undo  C=clear  A=accept auto-seeds  R=ref  N=next",
        "Left=add  Right=undo  C=clear  R=ref  N=next",
        "Left=add  Right=undo  C=clear  R=ref  Q=done",
    ]

    WIN = "Space Encoder"

    def redraw():
        base = ref_image if show_ref[0] else image
        img  = base.copy()
        # Bottleneck rings (cyan) — always shown as reference regardless of A
        for (px, py) in auto_obstacle_px:
            cv2.circle(img, (px, py), POINT_RADIUS + 4, COLOUR_AUTOSEED, 2)
        # Accepted auto-seeds shown with filled orange dot
        if use_auto_seeds[0] and phase[0] == 0:
            for (px, py) in auto_obstacle_px:
                cv2.circle(img, (px, py), POINT_RADIUS, COLOUR_ENTRANCE, -1)
        # User-clicked points
        for p_idx, pts in enumerate(all_points_px):
            for (px, py) in pts:
                cv2.circle(img, (px, py), POINT_RADIUS,     colours[p_idx], -1)
                cv2.circle(img, (px, py), POINT_RADIUS + 2, (255, 255, 255), 1)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 36), (w, h), (30, 30, 30), -1)
        n_auto = len(auto_obstacle_px)
        auto_str = ""
        if phase[0] == 0 and n_auto:
            auto_str = (f"  [A=ACCEPTED {n_auto} auto-seeds]"
                        if use_auto_seeds[0]
                        else f"  [A=accept {n_auto} auto-seeds]")
        label = f"[{labels[phase[0]]}]  {tips[phase[0]]}{auto_str}"
        cv2.putText(img, label, (10, h - 10), FONT, 0.46,
                    colours[phase[0]], 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            all_points_px[phase[0]].append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if all_points_px[phase[0]]:
                all_points_px[phase[0]].pop()
                redraw()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    img_h, img_w = image.shape[:2]
    scale = min(1400 / img_w, 900 / img_h, 1.0)
    cv2.resizeWindow(WIN, int(img_w * scale), int(img_h * scale))
    cv2.setMouseCallback(WIN, on_mouse)
    redraw()

    print("[INFO] Phase 1/3 — OBSTACLE points")
    print(f"       Cyan rings = {len(auto_obstacle_px)} auto-seeded bottleneck locations "
          f"(press A to accept as obstacles)")
    print(f"       Yellow crosses = calibration reference points")
    print(f"       Faint green dots = trajectory paths")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('n'), ord('N')):
            if phase[0] < 2:
                phase[0] += 1
                print(f"[INFO] Phase {phase[0]+1}/3 — {labels[phase[0]]} points  "
                      f"({'N = next' if phase[0] < 2 else 'Q = done'})")
                redraw()
        elif key in (ord('q'), ord('Q')):
            if phase[0] == 2:
                break
            else:
                print("[WARN] Q pressed early — skipping remaining phases.")
                phase[0] = 2
                break
        elif key in (ord('a'), ord('A')):
            if phase[0] == 0 and auto_obstacle_px:
                use_auto_seeds[0] = not use_auto_seeds[0]
                state = "ACCEPTED" if use_auto_seeds[0] else "cleared"
                print(f"[INFO] Auto-seeds {state}.")
                redraw()
        elif key in (ord('c'), ord('C')):
            all_points_px[phase[0]].clear()
            print(f"[INFO] Cleared all clicked {labels[phase[0]]} points.")
            redraw()
        elif key in (ord('r'), ord('R')):
            show_ref[0] = not show_ref[0]
            redraw()
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    # Whether to include auto-seeded obstacles
    accepted_auto_world = auto_obstacle_world if use_auto_seeds[0] else np.empty((0, 2))

    # ── Convert user-clicked plan-pixel coords → world metres ─────────────────
    ox, oy = origin[0], origin[1]

    def px_list_to_world(pts: list) -> np.ndarray:
        """Linear plan-pixel → world-metres using mpp and plan_origin_pixel."""
        if not pts:
            return np.empty((0, 2))
        arr = np.array(pts, dtype=np.float64)
        wx = (arr[:, 0] - ox) * mpp
        wy = (arr[:, 1] - oy) * mpp
        if invert_y:
            wy = -wy
        return np.column_stack([wx, wy])

    # Obstacles = accepted auto-seeds (world coords, no round-trip) +
    #             any obstacles the user manually clicked
    clicked_obs = px_list_to_world(all_points_px[0])
    parts = [p for p in (accepted_auto_world, clicked_obs) if len(p) > 0]
    obs_world = np.vstack(parts) if parts else np.empty((0, 2))

    return (
        obs_world,
        px_list_to_world(all_points_px[1]),
        px_list_to_world(all_points_px[2]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Validate inputs ---
    if not CSV_IN.exists():
        sys.exit(
            "[ERROR] No trajectory CSV found.\n  Looked for:\n"
            + "".join(f"    {p}\n" for p in _CANDIDATES)
            + "  Run trajectory-extraction/run_pipeline.py first."
        )
    if not IMAGE_PATH.exists():
        sys.exit(f"[ERROR] Floor-plan image not found: {IMAGE_PATH}")
    if CALIB_JSON is None:
        sys.exit("[ERROR] calib.json not found.")

    # --- Load calibration ---
    with open(CALIB_JSON) as f:
        calib = json.load(f)

    mpp      = float(calib["meters_per_plan_pixel"])
    origin   = calib["plan_origin_pixel"]
    invert_y = bool(calib.get("invert_plan_y", False))

    # --- Load floor-plan image ---
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        sys.exit(f"[ERROR] Could not read image: {IMAGE_PATH}")

    # --- Dimension check before doing anything with coordinates ---
    check_image_dimensions(calib, image)

    print(f"[INFO] Coordinate transform: mpp={mpp:.6f} m/px  "
          f"origin=({origin[0]:.1f}, {origin[1]:.1f})  "
          f"invert_y={invert_y}")

    # --- Load trajectory CSV and normalise column names ---
    df = pd.read_csv(CSV_IN)
    print(f"[INFO] Loaded {len(df):,} rows from {CSV_IN.name}")

    rename_map = {}
    if "track_id" in df.columns and "person_id" not in df.columns:
        rename_map["track_id"] = "person_id"
    if "frame_idx" in df.columns and "frame_number" not in df.columns:
        rename_map["frame_idx"] = "frame_number"
    elif "frame" in df.columns and "frame_number" not in df.columns:
        rename_map["frame"] = "frame_number"
    if "wx" in df.columns and "world_x" not in df.columns:
        rename_map["wx"] = "world_x"
    if "wy" in df.columns and "world_y" not in df.columns:
        rename_map["wy"] = "world_y"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[INFO] Normalised column names: {rename_map}")

    if "world_x" not in df.columns or "world_y" not in df.columns:
        sys.exit("[ERROR] CSV must contain world_x and world_y columns.")

    n_before = len(df)
    df = df.dropna(subset=["world_x", "world_y", "person_id", "frame_number"])
    if len(df) < n_before:
        print(f"[WARN] Dropped {n_before - len(df)} rows with NaN positions.")

    print(f"[INFO] Floor plan: {IMAGE_PATH.name}  "
          f"({image.shape[1]} × {image.shape[0]} px)")
    print(f"[INFO] Trajectory world extent: "
          f"X=[{df['world_x'].min():.1f}, {df['world_x'].max():.1f}] m  "
          f"Y=[{df['world_y'].min():.1f}, {df['world_y'].max():.1f}] m")

    # --- Build reference overlay (trajectory dots + calibration crosses) ---
    # This is burned into a separate image used as the window background so
    # the user can see exactly where trajectories are and verify calibration.
    ref_image = build_reference_layer(image, df, calib, mpp, origin, invert_y)

    # --- Auto-seed obstacles from bottleneck data ---
    # auto_obs_world : original world-metre coords  → used for distance calc
    # auto_obs_px    : plan-pixel coords derived from the above → display only
    auto_obs_world = np.empty((0, 2), dtype=np.float64)
    auto_obs_px    = []
    if BOTTLENECK_CSV and BOTTLENECK_CSV.exists():
        bn     = pd.read_csv(BOTTLENECK_CSV)
        top_bn = bn.nlargest(AUTO_SEED_TOP_N, "bottleneck_score")
        wx_all = top_bn["cell_x"].to_numpy(dtype=np.float64)
        wy_all = top_bn["cell_y"].to_numpy(dtype=np.float64)

        # Convert world → plan pixels for display; keep only in-bounds points
        px_x, px_y = world_to_plan_px(wx_all, wy_all, mpp, origin, invert_y)
        h_img, w_img = image.shape[:2]
        keep = [
            i for i, (x, y) in enumerate(zip(px_x.tolist(), px_y.tolist()))
            if 0 <= x < w_img and 0 <= y < h_img
        ]
        for i in keep:
            auto_obs_px.append((int(px_x[i]), int(px_y[i])))
        auto_obs_world = np.column_stack([wx_all[keep], wy_all[keep]])

        print(f"[INFO] {len(auto_obs_px)} bottleneck locations loaded "
              f"from {BOTTLENECK_CSV.name}  (press A in window to accept as obstacles)")
    else:
        print("[INFO] No bottleneck_cells.csv — click obstacles manually.")

    # --- Interactive landmark collection ---
    print()
    obs_world, bnd_world, ent_world = collect_points_interactive(
        image, ref_image, auto_obs_px, auto_obs_world, mpp, origin, invert_y
    )

    print(f"\n[INFO] Points collected and transformed to world metres:")
    print(f"       Obstacles  : {len(obs_world)}")
    print(f"       Boundaries : {len(bnd_world)}")
    print(f"       Entrances  : {len(ent_world)}")

    if len(obs_world) == 0 and len(bnd_world) == 0 and len(ent_world) == 0:
        sys.exit("[ERROR] No points were defined — nothing to encode.")

    # --- Query array: trajectory world positions ---
    query = df[["world_x", "world_y"]].to_numpy(dtype=np.float64)

    # --- Compute nearest-neighbour distances in world metres ---
    print("[INFO] Computing nearest-neighbour distances (metres) …")
    df["dist_to_obstacle"] = np.round(nearest_distances(query, obs_world), 4)
    df["dist_to_boundary"] = np.round(nearest_distances(query, bnd_world), 4)
    df["dist_to_entrance"] = np.round(nearest_distances(query, ent_world), 4)

    # --- Save ---
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"[INFO] Encoded CSV saved: {CSV_OUT}")

    # --- Summary ---
    print("\n" + "=" * 56)
    print("  SUMMARY  (distances in metres)")
    print("=" * 56)
    for col in ("dist_to_obstacle", "dist_to_boundary", "dist_to_entrance"):
        s = df[col].dropna()
        if s.empty:
            print(f"  {col:22s}  —  no landmarks defined")
        else:
            print(f"  {col:22s}  "
                  f"min={s.min():6.2f} m   "
                  f"max={s.max():6.2f} m   "
                  f"mean={s.mean():5.2f} m")
    print("=" * 56)
    print(f"\n  Rows written : {len(df):,}")
    print(f"  Columns      : {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
