# Manual Mask Annotation (Encoder V3 preparation)

Tools to hand-paint **architectural** walkable / obstacle masks for each
validated Barcelona recording, one recording at a time.

## Why this exists

The old encoder (`encode_spatial_auto_v2.py`) derived obstacle masks from
**trajectory coverage** — `obstacle = where nobody walked`. The Encoding Truth
Audit proved this is not architecture: Plaça Catalunya, full of trees, planters
and islands, was encoded as **0 interior obstacles / 0.0 m²**. Phase-0
feasibility and Phase-1 semi-automatic masks confirmed the architecture *is*
recoverable but not reliably automatic.

These manually reviewed masks become the **authoritative architectural layer**.
They must be human-approved **before** Encoder V3 is built, because every
downstream step — feature recomputation, directional features, turn classifier,
model retraining — inherits whatever this layer claims is obstacle vs walkable.
A wrong mask silently poisons all of it, exactly as the trajectory-coverage
mask did.

This tool **never** runs an encoder, rebuilds a dataset, or trains a model.

## Semantic definitions

**GREEN / walkable** — areas pedestrians can physically move through:
plazas, sidewalks, crosswalks, walkable bridges, walkable stairs, ramps, paths.

**RED / obstacle** — areas pedestrians cannot pass through or that physically
constrain movement: buildings, walls, fences, planters, vegetation islands,
fountains, monuments, railings, kiosks, permanent furniture, hard barriers.

**Do NOT mark as obstacles:** shadows, road markings, temporary vehicles,
image texture, lighting differences. (Parked cars and market stalls are
temporary — leave them as whatever surface they sit on.)

Walkable and obstacle are **mutually exclusive**: any pixel is at most one of
the two. Unpainted pixels are simply "unannotated".

## Workflow (one recording at a time)

```
python manual_annotation\annotate_masks_v3.py   --recording stairs_montjuic_01
python manual_annotation\validate_manual_masks_v3.py --recording stairs_montjuic_01
```

When the window opens you land on a **startup help screen** — read the legend,
then **press SPACE to begin**. You cannot paint until you do, which prevents the
"obstacle-only, no walkable" mistake. Then **review the outputs manually**
(open `overlay_manual.png`). Only after you accept a recording do you move on to
the next one. When all five are done:

```
python manual_annotation\validate_manual_masks_v3.py --all
```

## On-screen UI (HUD)

The controls are now self-explanatory **inside the window** — no need to
memorize keys:

- **Startup help screen** — full-screen legend (GREEN = walkable, RED =
  obstacle) + controls. Press **SPACE to begin**.
- **MODE panel (top-left)** — current mode (BRUSH / POLYGON), a colour-coded
  **`ACTIVE: WALKABLE / OBSTACLE / ERASE`** line (green / red / white), current
  brush size, and a `UNSAVED *` flag.
- **CONTROLS panel (top-left, below MODE)** — the full key list, always visible.
  Toggle it with **`h`** if it covers something.
- **Legend (bottom-right)** — Green = Walkable, Red = Obstacle.
- **`[SAVED]` flash** — after you press `s`, a green banner with the recording id
  and timestamp appears for ~2 seconds.
- **Unsaved-changes dialog** — pressing `q` with unsaved edits asks
  **Y = Save + Exit / N = Exit Without Saving / Esc = Cancel** instead of
  silently quitting.

## Controls

| Key | Action |
|---|---|
| `b` | brush mode |
| `p` | polygon mode |
| `1` | select **walkable** (green) |
| `2` | select **obstacle** (red) |
| `3` | select **erase** (brush mode only) |
| left-drag | paint (brush mode) |
| left-click | add polygon vertex (polygon mode) |
| `backspace` | remove last polygon vertex |
| `enter` | fill current polygon |
| `esc` | cancel current polygon |
| `[` / `]` | smaller / larger brush |
| `space` | (startup screen) begin annotating |
| `s` | save (shows a `[SAVED]` flash) |
| `h` | toggle the on-screen help / controls panel |
| `q` | save + quit (asks Y / N / Esc if there are unsaved changes) |

Tips:
- **Polygon mode** is fastest for large convex structures (building footprints,
  fountains, planted medians). **Brush mode** is best for organic edges
  (tree canopies, hedges) and touch-ups.
- The MODE panel (top-left) shows current mode, the colour-coded active target,
  brush size, and an `UNSAVED *` flag.
- Large images are scaled to fit the screen; painting still happens at full
  resolution. Adjust with `--max-width` / `--max-height`.

## Commands per validated recording

```
python manual_annotation\annotate_masks_v3.py --recording stairs_montjuic_01
python manual_annotation\annotate_masks_v3.py --recording red_bridge_combined_01
python manual_annotation\annotate_masks_v3.py --recording esplanade_espanya_01
python manual_annotation\annotate_masks_v3.py --recording placa_espanya_01
python manual_annotation\annotate_masks_v3.py --recording placa_catalunya_01
```

Custom image:
```
python manual_annotation\annotate_masks_v3.py --image path\to\image.png --recording custom_id
```

## Outputs

Per recording, under `mp-data\annotations\manual_masks_v3\<recording_id>\`:

| File | Meaning |
|---|---|
| `annotation_rgb.png` | copy of the source image that was annotated |
| `walkable_mask_v3_manual.png` | binary walkable mask (0/255) |
| `obstacle_mask_v3_manual.png` | binary obstacle mask (0/255) |
| `overlay_manual.png` | source + green/red overlay for visual review |
| `metadata.json` | recording id, source path, timestamp, size, pixel counts, coverage %, overlap, notes |
| `validation_report.md` | per-recording validation (after running the validator) |

Resume is automatic: re-running the annotator on a recording reloads its
existing masks so you can keep editing.

## Validation

```
python manual_annotation\validate_manual_masks_v3.py --recording placa_catalunya_01
python manual_annotation\validate_manual_masks_v3.py --all
```

Reports existence, image size, walkable/obstacle coverage %, **overlap pixels
(must be 0)**, and empty-mask warnings. A recording PASSES when both masks are
present, non-empty, and have zero overlap.

## Guardrails

This tooling deliberately does **not**:
- modify `encode_spatial_auto_v2.py` or `run_barcelona_v1_encoding.py`
- run Encoder V3 or recompute features
- rebuild the master dataset
- train any model

Mask creation and review only. Encoder V3 begins in a later, separate phase.
