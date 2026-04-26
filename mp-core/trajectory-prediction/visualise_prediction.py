"""
visualise_prediction.py
-----------------------
Loads the trained LSTM and runs an autoregressive prediction for one real
pedestrian, then draws three paths on the floor-plan image:

  GREEN  — the person's full real trajectory (all tracked frames)
  BLUE   — the 10-frame seed given to the model as input
  RED    — the 30 future positions predicted by the model

World coordinates (metres) are converted back to floor-plan pixels using the
scale and origin stored in calib.json.

Output
------
  prediction_visual.png  — saved next to this script
  OpenCV window          — opens automatically; press any key to close

Usage
-----
  python visualise_prediction.py
"""

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
MP_ROOT      = HERE.parent.parent
MP_DATA      = MP_ROOT / "mp-data"
BEHAVIOR_OUT = MP_DATA / "outputs" / "behavior"
PROCESSED    = MP_DATA / "processed" / "encoded"
PRED_OUT     = MP_DATA / "outputs" / "prediction"

CSV_PATH    = PROCESSED / "trajectories_encoded.csv"
MODEL_PATH  = PRED_OUT  / "trajectory_model.pth"
SCALER_PATH = PRED_OUT  / "scaler.pkl"
CALIB_PATH  = MP_DATA   / "raw" / "calibration" / "calib_skate1.json"
IMAGE_PATH  = MP_DATA   / "raw" / "images" / "top-down.png"
OUT_IMAGE   = PRED_OUT  / "prediction_visual.png"

# Optional trajectory-extraction analysis overlays — used if found, silently skipped if not
_BN_CANDIDATES = [
    BEHAVIOR_OUT / "bottlenecks" / "bottleneck_cells.csv",
    HERE / "outputs" / "bottleneck_cells.csv",  # local fallback
]
_FL_CANDIDATES = [
    BEHAVIOR_OUT / "flow_fields" / "flow_field_cells.csv",
    HERE / "outputs" / "flow_field_cells.csv",  # local fallback
]
BOTTLENECK_CSV = next((p for p in _BN_CANDIDATES if p.exists()), None)
FLOW_CSV       = next((p for p in _FL_CANDIDATES if p.exists()), None)

# ---------------------------------------------------------------------------
# How many future steps to predict
# ---------------------------------------------------------------------------
PREDICT_STEPS = 30

# ---------------------------------------------------------------------------
# Drawing colours  (BGR)
# ---------------------------------------------------------------------------
COL_REAL    = (0,  200,   0)    # green  — full real path
COL_SEED    = (220, 120,  0)    # blue   — seed window
COL_PRED    = (0,   0,  220)    # red    — predicted path
COL_TEXT    = (255, 255, 255)   # white  — legend text


# ---------------------------------------------------------------------------
# LSTM definition  (must match train_model.py exactly)
# ---------------------------------------------------------------------------

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# Coordinate conversion: world metres → floor-plan pixel
# ---------------------------------------------------------------------------

def world_to_plan_px(world_x: np.ndarray,
                     world_y: np.ndarray,
                     mpp: float,
                     origin: list,
                     invert_y: bool) -> tuple:
    """
    Convert world coordinates (metres) to pixel positions on the floor-plan
    image.  The relationship is stored in calib.json:

        plan_px_x = world_x / mpp + origin_x
        plan_px_y = world_y / mpp + origin_y   (invert_y=False)
        plan_px_y = -world_y / mpp + origin_y  (invert_y=True)

    Parameters
    ----------
    world_x, world_y : arrays of world coordinates in metres
    mpp              : metres per plan pixel
    origin           : [origin_x, origin_y] in plan pixels
    invert_y         : whether the plan's Y axis is flipped

    Returns
    -------
    px_x, px_y : integer pixel coordinates on the plan image
    """
    ox, oy = origin
    px_x = (world_x / mpp + ox).astype(int)
    if invert_y:
        px_y = (-world_y / mpp + oy).astype(int)
    else:
        px_y = (world_y / mpp + oy).astype(int)
    return px_x, px_y


# ---------------------------------------------------------------------------
# Autoregressive prediction loop
# ---------------------------------------------------------------------------

def predict_future(model: nn.Module,
                   seed_window: np.ndarray,
                   feat_scaler,
                   target_scaler,
                   steps: int,
                   device: torch.device) -> np.ndarray:
    """
    Starting from a 10-frame seed, predict `steps` future (world_x, world_y)
    positions one step at a time.  Each prediction is fed back as the start
    of the next window (autoregressive / "rollout" inference).

    For the non-positional features that we cannot recompute for a predicted
    frame (dist_to_obstacle, dist_to_boundary, dist_to_entrance, frame_number)
    we carry forward the values from the last seed frame.  This is a simple
    but reasonable approximation — the model has learned to weight them during
    training.

    Parameters
    ----------
    seed_window  : np.ndarray, shape (WINDOW_SIZE, 6) — raw (unscaled) values
    feat_scaler  : fitted MinMaxScaler for the 6 input features
    target_scaler: fitted MinMaxScaler for the 2 target columns
    steps        : number of future positions to produce

    Returns
    -------
    predictions  : np.ndarray, shape (steps, 2) — world (x, y) in metres
    """
    model.eval()

    # Work on a copy so we don't modify the original seed
    window = seed_window.copy().astype(np.float64)   # (W, 8)
    W, F = window.shape

    # Feature column indices (must match FEATURE_COLS in train_model.py):
    #   0=world_x  1=world_y  2=dist_obs  3=dist_bnd  4=dist_ent
    #   5=frame    6=delta_x  7=delta_y
    last_frame = window[-1, 5]    # frame number of the last seed frame
    last_aux   = window[-1, 2:5]  # [dist_obstacle, dist_boundary, dist_entrance]

    predictions = []

    with torch.no_grad():
        for step in range(steps):
            # --- Scale the current window ---
            scaled = feat_scaler.transform(window)             # (W, 8)
            x_tensor = torch.tensor(
                scaled[np.newaxis, :, :], dtype=torch.float32
            ).to(device)                                       # (1, W, 8)

            # --- Run the model ---
            out_scaled = model(x_tensor).cpu().numpy()         # (1, 2)

            # --- Invert the target scaler to get real-world metres ---
            pred_world = target_scaler.inverse_transform(out_scaled)[0]  # (2,)
            predictions.append(pred_world.copy())

            # --- Build the new row to append to the sliding window ---
            # delta_x/delta_y for the predicted frame = displacement from the
            # previous (last) position in the window.
            prev_wx = window[-1, 0]
            prev_wy = window[-1, 1]
            next_frame = last_frame + step + 1
            new_row = np.array([
                pred_world[0],            # world_x        (predicted)
                pred_world[1],            # world_y        (predicted)
                last_aux[0],              # dist_obstacle  (carried forward)
                last_aux[1],              # dist_boundary  (carried forward)
                last_aux[2],              # dist_entrance  (carried forward)
                next_frame,               # frame_number   (incremented)
                pred_world[0] - prev_wx,  # delta_x        (computed)
                pred_world[1] - prev_wy,  # delta_y        (computed)
            ], dtype=np.float64)

            # --- Slide the window: drop the oldest frame, add the new one ---
            window = np.vstack([window[1:], new_row])          # (W, 8)

    return np.array(predictions)   # (steps, 2)


# ---------------------------------------------------------------------------
# trajectory-extraction overlay helpers
# ---------------------------------------------------------------------------

def draw_flow_field(img: np.ndarray,
                    flow_df: pd.DataFrame,
                    mpp: float,
                    origin: list,
                    invert_y: bool) -> None:
    """
    Draw one arrow per grid cell showing the average pedestrian direction.

    Arrow colour encodes direction_consistency (0=random, 1=perfectly aligned):
      low  → grey      high → bright cyan
    Arrow length is proportional to mean_speed (scaled to plan pixels).
    Only cells with at least 5 observations are drawn to avoid noise.
    """
    flow_df = flow_df[flow_df["n_vectors"] >= 5].copy()
    if flow_df.empty:
        return

    # Convert cell centres from world metres → plan pixels
    cx, cy = world_to_plan_px(
        flow_df["cell_x"].to_numpy(dtype=np.float64),
        flow_df["cell_y"].to_numpy(dtype=np.float64),
        mpp, origin, invert_y,
    )

    speed    = flow_df["mean_speed"].to_numpy(dtype=np.float64)
    dx_world = flow_df["mean_dx"].to_numpy(dtype=np.float64)   # m/frame
    dy_world = flow_df["mean_dy"].to_numpy(dtype=np.float64)   # m/frame
    consist  = flow_df["direction_consistency"].to_numpy(dtype=np.float64)

    # mean_dx/mean_dy are in m/frame (very small numbers).
    # mean_speed is in m/s (much larger).  We must NOT mix the two units.
    #
    # Correct approach:
    #   1. Normalise dx/dy to a unit direction vector (pure direction, no magnitude).
    #   2. Set visual arrow length from mean_speed, scaled to plan pixels.
    #      MAX_ARROW_PX caps the longest arrow so nothing dominates the image.

    magnitude = np.sqrt(dx_world ** 2 + dy_world ** 2)
    safe_mag  = np.where(magnitude > 1e-9, magnitude, 1.0)
    unit_dx   = dx_world / safe_mag          # dimensionless direction
    unit_dy   = dy_world / safe_mag

    MAX_ARROW_PX = 28                        # longest arrow in plan pixels
    speed_norm   = speed / (speed.max() + 1e-9)          # 0–1
    arrow_len    = (speed_norm * MAX_ARROW_PX).clip(4)   # at least 4 px

    # Y direction: world_to_plan_px maps world_y → plan_py via  py = wy/mpp + oy
    # so d(plan_py) = d(world_y)/mpp.  When invert_y=True the mapping is negated.
    y_sign = -1 if invert_y else 1
    tip_x = (cx + unit_dx * arrow_len).astype(int)
    tip_y = (cy + unit_dy * arrow_len * y_sign).astype(int)

    h, w = img.shape[:2]

    for i in range(len(flow_df)):
        if not (0 <= cx[i] < w and 0 <= cy[i] < h):
            continue
        # Colour: grey (low consistency) → cyan (high consistency)
        alpha  = float(np.clip(consist[i], 0, 1))
        colour = (
            int(120 + 135 * alpha),   # B
            int(120 + 80  * alpha),   # G
            int(120 - 120 * alpha),   # R
        )
        cv2.arrowedLine(
            img,
            (int(cx[i]), int(cy[i])),
            (int(tip_x[i]), int(tip_y[i])),
            colour,
            thickness  = 1,
            tipLength  = 0.35,
            line_type  = cv2.LINE_AA,
        )


def draw_bottlenecks(img: np.ndarray,
                     bn_df: pd.DataFrame,
                     mpp: float,
                     origin: list,
                     invert_y: bool,
                     homography: np.ndarray | None = None) -> None:
    """
    Draw bottleneck cells as heat circles on the floor plan.

    Circle size and opacity both scale with bottleneck_score so the worst
    areas are visually prominent.  Colour is a yellow → red ramp.
    Only the top-20 cells are drawn to avoid clutter.

    Coordinate handling
    -------------------
    New format (cx_px / cy_px): video-frame pixel coordinates.  Reprojects via
    the homography matrix (image pixels → world metres) then via the linear
    plan-pixel formula (world metres → plan pixels).
    Old format (cell_x / cell_y): world metres, converted directly to plan pixels.
    """
    top = bn_df.nlargest(20, "bottleneck_score").copy()
    if top.empty:
        return

    if "cx_px" in top.columns and "cy_px" in top.columns and homography is not None:
        # New format: reproject image pixels → world metres → plan pixels
        img_pts = top[["cx_px", "cy_px"]].to_numpy(dtype=np.float64)  # (N, 2)
        ones = np.ones((len(img_pts), 1), dtype=np.float64)
        hom_pts = (homography @ np.hstack([img_pts, ones]).T).T  # (N, 3)
        world_x = hom_pts[:, 0] / hom_pts[:, 2]
        world_y = hom_pts[:, 1] / hom_pts[:, 2]
        cx, cy = world_to_plan_px(world_x, world_y, mpp, origin, invert_y)
    else:
        # Old format: cell_x / cell_y already in world metres
        cx, cy = world_to_plan_px(
            top["cell_x"].to_numpy(dtype=np.float64),
            top["cell_y"].to_numpy(dtype=np.float64),
            mpp, origin, invert_y,
        )

    scores     = top["bottleneck_score"].to_numpy(dtype=np.float64)
    score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    h, w = img.shape[:2]
    overlay = img.copy()

    for i in range(len(top)):
        if not (0 <= cx[i] < w and 0 <= cy[i] < h):
            continue
        radius = int(8 + score_norm[i] * 18)   # 8–26 px
        # Yellow (low) → Red (high): B=0, G decreases, R=255
        green  = int(200 * (1.0 - score_norm[i]))
        colour = (0, green, 255)
        cv2.circle(overlay, (int(cx[i]), int(cy[i])), radius, colour, -1, cv2.LINE_AA)

    # Blend with 40 % opacity so floor plan remains readable beneath
    cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_path(img: np.ndarray,
              px_x: np.ndarray,
              px_y: np.ndarray,
              colour: tuple,
              dot_radius: int = 4,
              line_thickness: int = 2) -> None:
    """Draw dots connected by lines for a sequence of pixel positions."""
    pts = list(zip(px_x.tolist(), px_y.tolist()))

    # Lines first (drawn under the dots)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], colour, line_thickness, cv2.LINE_AA)

    # Dots on top
    for pt in pts:
        cv2.circle(img, pt, dot_radius, colour, -1, cv2.LINE_AA)
        cv2.circle(img, pt, dot_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(img: np.ndarray,
                has_flow: bool = False,
                has_bn: bool = False) -> None:
    """Draw a small legend in the top-left corner."""
    entries = [
        (COL_REAL, "Real path (all frames)"),
        (COL_SEED, "Seed  (first 10 frames)"),
        (COL_PRED, "Predicted  (next 30 steps)"),
    ]
    if has_flow:
        entries.append(((200, 200, 0), "Flow field (trajectory-extraction)"))
    if has_bn:
        entries.append(((0, 80, 255), "Bottleneck zones (trajectory-extraction)"))
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 10
    line_h     = 26
    box_w      = 290
    box_h      = pad * 2 + line_h * len(entries)

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad),
                  (pad + box_w, pad + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, (col, label) in enumerate(entries):
        y = pad * 2 + i * line_h
        # Colour swatch
        cv2.rectangle(img,
                      (pad + 6,      y - 10),
                      (pad + 6 + 20, y +  6),
                      col, -1)
        # Label
        cv2.putText(img, label,
                    (pad + 34, y + 4),
                    font, font_scale, COL_TEXT, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Validate inputs ---
    for p in (CSV_PATH, MODEL_PATH, SCALER_PATH, CALIB_PATH, IMAGE_PATH):
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}\n"
                     f"        Run the earlier pipeline steps first.")

    # --- Load data ---
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded {len(df):,} rows  |  {df['person_id'].nunique()} persons")

    with open(SCALER_PATH, "rb") as f:
        scaler_bundle = pickle.load(f)
    feat_scaler   = scaler_bundle["feature_scaler"]
    target_scaler = scaler_bundle["target_scaler"]
    feature_cols  = scaler_bundle["feature_cols"]
    window_size   = scaler_bundle["window_size"]

    with open(CALIB_PATH) as f:
        calib = json.load(f)
    mpp      = float(calib["meters_per_plan_pixel"])
    origin   = calib["plan_origin_pixel"]          # [ox, oy]
    invert_y = bool(calib.get("invert_plan_y", False))
    homography = np.array(calib["homography_matrix"], dtype=np.float64) \
        if "homography_matrix" in calib else None

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        sys.exit(f"[ERROR] Cannot read image: {IMAGE_PATH}")

    # --- Pick the person with the most tracked frames ---
    frame_counts = df.groupby("person_id")["frame_number"].count()
    best_person  = frame_counts.idxmax()
    best_count   = frame_counts.max()
    print(f"[INFO] Selected person {best_person}  ({best_count} frames tracked)")

    if best_count < window_size + 1:
        sys.exit(
            f"[ERROR] Person {best_person} has only {best_count} frames, "
            f"need at least {window_size + 1}."
        )

    # Get this person's trajectory, sorted by frame
    track = (df[df["person_id"] == best_person]
             .sort_values("frame_number")
             .reset_index(drop=True))

    # --- Add velocity features to the track (must match train_model.py) ---
    # delta_x[t] = world_x[t] - world_x[t-1], 0 for the first frame
    wx = track["world_x"].to_numpy(dtype=np.float64)
    wy = track["world_y"].to_numpy(dtype=np.float64)
    delta_x      = np.zeros(len(track), dtype=np.float64)
    delta_y      = np.zeros(len(track), dtype=np.float64)
    delta_x[1:]  = wx[1:] - wx[:-1]
    delta_y[1:]  = wy[1:] - wy[:-1]
    track = track.copy()
    track["delta_x"] = delta_x
    track["delta_y"] = delta_y

    # --- Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TrajectoryLSTM(
        input_size  = len(feature_cols),   # 8
        hidden_size = 128,
        output_size = 2,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"[INFO] Model loaded from {MODEL_PATH.name}  |  device: {device}")

    # --- Extract seed window (first 10 frames) ---
    seed_raw = track[feature_cols].iloc[:window_size].to_numpy(dtype=np.float64)
    # shape: (10, 8)

    # --- Run autoregressive prediction ---
    print(f"[INFO] Predicting {PREDICT_STEPS} future steps …")
    pred_world = predict_future(
        model, seed_raw, feat_scaler, target_scaler,
        steps=PREDICT_STEPS, device=device
    )
    # pred_world: (PREDICT_STEPS, 2)  — columns are [world_x, world_y]

    # --- Collect real-path world coordinates ---
    real_wx = track["world_x"].to_numpy(dtype=np.float64)
    real_wy = track["world_y"].to_numpy(dtype=np.float64)

    # Seed portion (first 10 frames)
    seed_wx = real_wx[:window_size]
    seed_wy = real_wy[:window_size]

    # Predicted portion
    pred_wx = pred_world[:, 0]
    pred_wy = pred_world[:, 1]

    # --- Convert world coords → floor-plan pixels ---
    def to_px(wx, wy):
        return world_to_plan_px(
            np.asarray(wx, dtype=np.float64),
            np.asarray(wy, dtype=np.float64),
            mpp, origin, invert_y,
        )

    real_px_x, real_px_y = to_px(real_wx, real_wy)
    seed_px_x, seed_px_y = to_px(seed_wx, seed_wy)
    pred_px_x, pred_px_y = to_px(pred_wx, pred_wy)

    # --- Draw on a copy of the floor plan ---
    canvas = image.copy()

    # Layer 0: trajectory-extraction analysis overlays (drawn first, under trajectories)
    has_flow = has_bn = False

    if FLOW_CSV:
        flow_df = pd.read_csv(FLOW_CSV)
        draw_flow_field(canvas, flow_df, mpp, origin, invert_y)
        has_flow = True
        print(f"[INFO] Flow field overlay drawn ({len(flow_df)} cells from {FLOW_CSV.name})")

    if BOTTLENECK_CSV:
        bn_df = pd.read_csv(BOTTLENECK_CSV)
        draw_bottlenecks(canvas, bn_df, mpp, origin, invert_y, homography)
        has_bn = True
        print(f"[INFO] Bottleneck overlay drawn ({len(bn_df)} cells from {BOTTLENECK_CSV.name})")

    # Layer 1: trajectory paths
    draw_path(canvas, real_px_x, real_px_y, COL_REAL, dot_radius=3, line_thickness=1)
    draw_path(canvas, seed_px_x, seed_px_y, COL_SEED, dot_radius=5, line_thickness=2)
    draw_path(canvas, pred_px_x, pred_px_y, COL_PRED, dot_radius=5, line_thickness=2)

    # Mark the very first seed point with a larger dot so it's easy to find
    cv2.circle(canvas, (int(seed_px_x[0]), int(seed_px_y[0])), 9, COL_SEED, -1)
    cv2.circle(canvas, (int(seed_px_x[0]), int(seed_px_y[0])), 10, (255,255,255), 1)

    # Mark the join between seed and prediction
    join_pt = (int(pred_px_x[0]), int(pred_px_y[0]))
    cv2.circle(canvas, join_pt, 7, COL_PRED, -1)
    cv2.circle(canvas, join_pt, 8, (255, 255, 255), 1)

    draw_legend(canvas, has_flow=has_flow, has_bn=has_bn)

    # Person label
    cv2.putText(canvas,
                f"Person {best_person}  |  real: {len(track)} frames  |  "
                f"seed: {window_size}  |  predicted: {PREDICT_STEPS}",
                (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Save ---
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMAGE), canvas)
    print(f"[INFO] Saved: {OUT_IMAGE.name}")

    # --- Display ---
    win_name = "Trajectory Prediction  (press any key to close)"
    h, w     = canvas.shape[:2]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(w, 1400), min(h, 900))
    cv2.imshow(win_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Console summary ---
    print("\n" + "=" * 52)
    print("  PREDICTION SUMMARY")
    print("=" * 52)
    print(f"  Person           : {best_person}")
    print(f"  Real frames      : {len(track)}")
    print(f"  Seed frames      : {window_size}")
    print(f"  Predicted steps  : {PREDICT_STEPS}")
    print(f"  Pred world range : "
          f"X=[{pred_wx.min():.2f}, {pred_wx.max():.2f}] m  "
          f"Y=[{pred_wy.min():.2f}, {pred_wy.max():.2f}] m")
    print(f"  Real world range : "
          f"X=[{real_wx.min():.2f}, {real_wx.max():.2f}] m  "
          f"Y=[{real_wy.min():.2f}, {real_wy.max():.2f}] m")
    print("=" * 52)


if __name__ == "__main__":
    main()
