# Motion Pixels — Final Behavior Maps Batch Report

**Visualization-only.** No retracking, recalibration, bottleneck-score recomputation, or source-data modification. Bottleneck scores read from `bottleneck_cells.csv`. See `FINAL_BEHAVIOR_MAP_RECIPES.md` for the exact recipe extracted for each family.

## Shared dimensions & HUD
- Map **1920×1080**, HUD footer **70** → canvas **1920×1150**.
- GIF **760×455** @ **10 fps** (stride 2); MP4 **1920×1150** H.264 @ 20 fps; still PNG 1920×1150.
- Font **Roboto**; HUD geometry identical across families. Flow Fields = DARK footer; Speed Population & Bottleneck Density = LIGHT footer. Timer = real source-video time. World window = 16:9 fit of each plan (full plan visible; neutral padding; no stretch).
- Durations: Flow Fields 17 s · Speed Population 17 s · Bottleneck Density 10 s.

## Recordings discovered

| Recording | Place | Tracks | Rows | Time | Flow | Speed | Bottleneck |
|---|---|---|---|---|---|---|---|
| esplanade_espanya_01 | Esplanade Espanya | 568 | 283875 | 04:04 | True | True | True |
| placa_catalunya_01 | Plaça Catalunya | 1537 | 358351 | 05:17 | True | True | True |
| placa_espanya_01 | Plaça Espanya | 923 | 213618 | 03:02 | True | True | True |
| placa_montjuic_01 | Plaça Montjuïc | 165 | 93184 | 04:11 | True | True | True |
| red_bridge_combined_01 | Red Bridge | 716 | 87076 | 05:52 | True | True | True |
| stairs_montjuic_01 | Stairs Montjuïc I | 392 | 128259 | 05:01 | True | True | True |
| stairs_montjuic_02 | Stairs Montjuïc II | 439 | 88239 | 03:32 | True | True | True |

## Results per recording

### esplanade_espanya_01 — Esplanade Espanya
- **Flow Fields**: ok · GIF 10.0 MB · MP4 4.2 MB · particles scaled to 3712/5745 for 411 flow paths
- **Speed Population**: ok · GIF 27.0 MB · MP4 3.9 MB · persistence trace 5 s; 556 tracks shown
- **Bottleneck Density**: ok · GIF 13.7 MB · MP4 1.5 MB · 2320 cells in-frame (dropped 9)

### placa_catalunya_01 — Plaça Catalunya
- **Flow Fields**: ok · GIF 15.8 MB · MP4 6.1 MB · full particle counts
- **Speed Population**: ok · GIF 42.4 MB · MP4 4.5 MB · persistence trace 5 s; 1328 tracks shown
- **Bottleneck Density**: ok · GIF 20.1 MB · MP4 1.8 MB · 633 cells in-frame (dropped 121)

### placa_espanya_01 — Plaça Espanya
- **Flow Fields**: ok · GIF 16.2 MB · MP4 7.5 MB · full particle counts
- **Speed Population**: ok · GIF 48.2 MB · MP4 6.4 MB · persistence trace 5 s; 836 tracks shown
- **Bottleneck Density**: ok · GIF 25.0 MB · MP4 4.3 MB · 3480 cells in-frame (dropped 661)

### placa_montjuic_01 — Plaça Montjuïc
- **Flow Fields**: ok · GIF 15.3 MB · MP4 5.3 MB · particles scaled to 876/1356 for 97 flow paths
- **Speed Population**: ok · GIF 46.8 MB · MP4 3.3 MB · persistence trace 5 s; 133 tracks shown
- **Bottleneck Density**: ok · GIF 23.3 MB · MP4 1.9 MB · 676 cells in-frame (dropped 680)

### red_bridge_combined_01 — Red Bridge
- **Flow Fields**: ok · GIF 11.2 MB · MP4 2.2 MB · particles scaled to 800/1200 for 36 flow paths
- **Speed Population**: ok · GIF 40.6 MB · MP4 1.5 MB · persistence trace 5 s; 643 tracks shown
- **Bottleneck Density**: ok · GIF 21.5 MB · MP4 0.7 MB · 78 cells in-frame (dropped 16)

### stairs_montjuic_01 — Stairs Montjuïc I
- **Flow Fields**: ok · GIF 11.4 MB · MP4 3.5 MB · particles scaled to 939/1454 for 104 flow paths
- **Speed Population**: ok · GIF 36.2 MB · MP4 2.8 MB · persistence trace 5 s; 365 tracks shown
- **Bottleneck Density**: ok · GIF 19.8 MB · MP4 1.5 MB · 119 cells in-frame (dropped 22)

### stairs_montjuic_02 — Stairs Montjuïc II
- **Flow Fields**: ok · GIF 12.2 MB · MP4 3.8 MB · particles scaled to 985/1524 for 109 flow paths
- **Speed Population**: ok · GIF 39.6 MB · MP4 2.5 MB · persistence trace 5 s; 298 tracks shown
- **Bottleneck Density**: ok · GIF 21.3 MB · MP4 2.1 MB · 428 cells in-frame (dropped 1053)

## Adjustments for sparse recordings
- **Flow Fields:** particle counts scale with available flow paths (Plaça Catalunya reference = 465 paths → N_A 4200 / N_B 6500; ≈9/14 per path). Sparser recordings get proportionally fewer particles (clamped, never faked); per-recording counts in the table notes.
- **Speed Population:** same 5 s fading-trace persistence window for all; no duplicated pedestrians.
- **Pillarbox/letterbox:** plans narrower than 16:9 (placa_montjuic, red_bridge, stairs_montjuic_01/02) are padded on the sides; wider plans padded top/bottom. The full plan is always shown; nothing is stretched or cropped.

## Confirmation
- No source data modified; all CSVs/plans/calibrations read-only.
- Bottleneck scores not recomputed.
- All families share the same canvas size and HUD geometry.

## Output locations
- `behaviormaps_final/flow_fields/` · `…/speed_population/` · `…/bottleneck_density/`
- Manifest: `behaviormaps_final/manifest/` · Reports: `behaviormaps_final/reports/`
