# Motion Pixels — Behavioral Map Opportunity Audit

**Date:** 2026-06-16
**Status:** Discovery & audit only. No maps, figures, or animations generated. No retracking, recalibration, re-inference, or data modification performed.
**Scope:** Everything in `motion-pixels/` plus the canonical dataset roots at `Desktop\new_datasets\` (Barcelona_v1 / v3 source recordings + `filtered_250m` products) and the encoded datasets in `Desktop\new_datasets\Barcelona_v1_encoded\`.

**Guiding question:** *What architecturally meaningful behavioral maps can we build from data we already have — without touching the pipeline?*

---

## 0. Executive summary

The project is **map-rich and map-ready**. Seven calibrated Barcelona recordings each carry: a plan image, a homography, world-space trajectories (~22k–358k rows), and a complete derived-metric suite (speed, stops, dwell, shifts, bottlenecks, flow fields, linger zones) plus a spatial-encoding layer (walkable / obstacle / boundary masks, three distance fields, entry/exit clusters).

Only **3 behavioral map families are currently rendered** (bottleneck, flow-field quiver, linger-zone scatter) and only the bottleneck map has been polished to presentation quality (Plaça Catalunya, v1→v2d). **At least 12 further architecturally compelling maps are fully supported by existing data** with zero new computation of source signals — they are rendering tasks, not analysis tasks.

The single biggest under-exploited asset is the **flow-field vector layer** (`flow_field_vectors.csv` + `flow_field_cells.csv` with `direction_consistency`). It already encodes movement direction, coherence, and speed per cell and per step — enough to build desire-line/corridor maps, conflict/crossing maps, coherence maps, and turning-intensity maps that no current figure exploits.

**Top recommendation:** build the **Desire-Line / Corridor map**, the **Dwell-Attractor map**, and the **Origin–Destination connectivity map** first — they are the three that read as architecture in 10 seconds and reuse data already on disk.

> ⚠️ **Known caveat carried from prior audits (do not ignore when interpreting maps):** the obstacle mask is partly a *trajectory-coverage artifact*, not verified built geometry, and world-space calibration has a documented vertical-extent mismatch on Plaça Catalunya (cells extend below the plan). Coverage/dead-zone maps must be framed as "where we have tracks," not "where the space is empty." See §3 and §7.

---

## 1. Data inventory

### 1.1 Recordings (the unit of everything)

Seven calibrated Barcelona recordings. Each directory under `Desktop\new_datasets\<recording>\` has `plan/`, `calibration/calib.json` (homography + reprojection diagnostics), `raw_video/`, `tracking/`, and a `filtered_250m/` product.

| Recording | Typology (architectural read) | Encoded rows | Plan | Calib | filtered_250m |
|---|---|---|---|---|---|
| `placa_catalunya_01` | Large civic plaza, multi-entry | 358,352 | ✓ | ✓ | ✓ |
| `esplanade_espanya_01` | Open esplanade / approach axis | 283,876 | ✓ | ✓ | ✓ |
| `placa_espanya_01` | Roundabout / high-turn node | 213,619 | ✓ | ✓ | ✓ |
| `stairs_montjuic_01` | Monumental stair (vertical) | 128,260 | ✓ | ✓ | ✓ |
| `stairs_montjuic_02` | Monumental stair (2nd take) | (source) | ✓ | ✓ | ✓ |
| `red_bridge_combined_01` | Linear bridge / forced corridor | 87,077 | ✓ | ✓ | ✓ |
| `placa_montjuic_01` | Plaza (Montjuïc) | (source) | ✓ | ✓ | ✓ |

These seven span an unusually clean **typological gradient**: forced corridor (bridge) → linear vertical (stairs) → directional approach (esplanade) → open plaza (catalunya/montjuïc) → turning node (espanya). That gradient is itself a story most behavioral maps can be read against.

### 1.2 Trajectory data (per recording)

**Encoded master trajectory** — `Barcelona_v1_encoded\<rec>\spatial_v21C\trajectories_encoded.csv`
Columns: `frame, time_s, track_id, conf, x1,y1,x2,y2, cx,cy, foot_x,foot_y, image_x,image_y, world_x,world_y, dist_to_obstacle, dist_to_boundary, dist_to_entrance`

**Filtered world trajectory** — `<rec>\filtered_250m\trajectories_world_filtered_250m.csv`
Columns: `frame, time_s, track_id, conf, x1..y2, cx,cy, foot_x,foot_y, image_x,image_y, world_x,world_y` (≤250 m path-length filtered — the clean render-ready source).

**Aggregated master** (model training) — `mp-data\processed\master_dataset\master_dataset.csv` (18,085 rows, adds `recording_id` + `split`). Also `Barcelona_v1_master_dataset\` and `Barcelona_v3_manual_master_dataset\`.

### 1.3 Derived metric tables (per recording, in `filtered_250m/`)

| File | Grain | Key fields |
|---|---|---|
| `metrics/speed_per_observation.csv` | per point | `speed_m_s, speed_smooth_m_s, step_dist_m` |
| `metrics/speed_per_track.csv` | per track | `mean/median/p95 speed, path_len_m, n_obs` |
| `metrics/stop_flags_per_observation.csv` | per point | `speed_smooth_m_s, is_stop` |
| `metrics/shift_flags_per_observation.csv` | per point | `is_shift` (lateral re-direction events) |
| `metrics/dwell_events.csv` | per event | `start/end_time, duration_s, cx_mean,cy_mean, n_obs` |
| `bottlenecks/bottleneck_cells.csv` | 1 m grid | `observation_count, unique_track_count, mean/median_speed, stop_fraction, density_score, slowness_score, bottleneck_score` |
| `bottlenecks/bottleneck_top_cells.csv` | ranked | top cells |
| `flow_fields/flow_field_cells.csv` | 1 m grid | `n_vectors, mean_dx, mean_dy, mean_speed, direction_consistency` |
| `flow_fields/flow_field_vectors.csv` | per step | `dx,dy,dt,speed, ux,uy, mid_x,mid_y, cell_*` |
| `linger_zones/linger_zones.csv` | zone | `n_events, unique_tracks, mean/median/total_duration_s, centroid_x/y` |
| `linger_zones/dwell_events_enriched.csv` | per event | `duration_s, x,y, dwell_class, zone_id` |

A repo-level copy of the same products exists at `mp-data\outputs\behavior\{bottlenecks,flow_fields,linger_zones}\` (MACBA-era), and the metric-generating scripts are in `mp-core\trajectory-extraction\` (`compute_bottlenecks.py`, `compute_flow_fields.py`, `compute_linger_zones.py`, `compute_metrics.py`, `compute_heatmap.py`).

### 1.4 Spatial encoding layers (per recording, `Barcelona_v1_encoded\<rec>\spatial_v21C\`)

Raster masks @ **0.1 m/px** + entry/exit clusters:
- `walkable_mask.png`, `obstacle_mask.png`, `boundary_mask.png`
- `distance_to_obstacle_m.png`, `distance_to_boundary_m.png`, `distance_to_entrance_m.png` (also as per-point columns in the encoded CSV)
- `entry_exit_points.csv` (DBSCAN clusters: `center_x, center_y, n_points`)
- `summary.md` with grid extent, mask areas, feature distributions

### 1.5 Model outputs (prediction)

`mp-core\trajectory-prediction\` — MODEL_X (baseline H10), MODEL_XR (magnitude-aware), MODEL_XC (curvature-aware), plus MODEL_X_HORIZON_SWEEP and MP_X experiments.
- `MODEL_X/evaluation/`: `per_recording_metrics.csv`, `per_window_metrics.csv`, `all_window_arrays.npz` (raw GT vs pred arrays), `plot_windows.pkl`
- Rollout dynamics audits: `direction_metrics_by_timestep.csv`, `step_length_by_timestep.csv`, `net_displacement_growth_by_timestep.csv`, etc.

### 1.6 Existing rendered behavior maps (baseline to beat)

- `mp-visualization\behavior_maps\outputs\` — Plaça Catalunya bottleneck map, iterated v1→v2d (plan-clipped, architectural underlay, variable-size cells). **This is the visual-language reference for everything else.**
- `mp-visualization\behavior_maps\bottleneck_animations\` — bottleneck animation test scaffold.
- `Desktop\new_datasets\Barcelona_v1_filtered_250m_plots\` — ALL-recording contact sheets for bottlenecks / flow fields / linger zones / world tracks (raw pipeline quality, not presentation-polished).
- Per recording: `bottleneck_heatmap.png`, `flow_field_quiver.png`, `linger_zones_plot.png`, `trajectories_world_plot.png`.

---

## 2. Metric inventory — discovery answers

For each signal: **available? where? spatial res? temporal res? reliability? visualizable?**

| Signal | Source | Spatial res | Temporal res | Reliability | Visualizable |
|---|---|---|---|---|---|
| Position (world x,y) | trajectories | continuous (m) | per frame (30 fps; espanya 240) | High where calibrated; vertical-extent mismatch on catalunya | ★★★★★ |
| Occupancy / density | derivable count per cell | 1 m grid | aggregate or windowed | High (robust to noise) | ★★★★★ |
| Speed | `speed_per_observation` (+smoothed) | per point → 1 m grid | per frame | High (smoothed col denoises) | ★★★★★ |
| Stop events | `stop_flags_per_observation.is_stop` | per point | per frame | Med-High (threshold on smoothed speed) | ★★★★★ |
| Dwell / linger | `dwell_events`, `linger_zones` | zone centroid + events | event-level | High (already clustered) | ★★★★★ |
| Flow direction | `flow_field_vectors (ux,uy)`, `cells (mean_dx,dy)` | per step → 1 m grid | per step | High in corridors, noisy in milling crowds | ★★★★★ |
| Direction consistency / coherence | `flow_field_cells.direction_consistency` | 1 m grid | aggregate | High (0–1, interpretable) | ★★★★★ |
| Bottleneck score | `bottleneck_cells` (density×slowness×stop) | 1 m grid | aggregate | High (composite, already tuned) | ★★★★★ |
| Heading / turn rate / angular change | **derivable** from (x,y,frame) | per point → cell | per frame | Med (sensitive to jitter; N≈10 smoothing needed — see denoising note) | ★★★★☆ |
| Acceleration | **derivable** (Δspeed/Δt) | per point → cell | per frame | Med (noisy; smooth first) | ★★★★☆ |
| Curvature | **derivable** (heading change / step) | per point → cell | per frame | Med (jitter-sensitive) | ★★★★☆ |
| Path / displacement / sinuosity | **derivable** per track | per track | track lifetime | High | ★★★★☆ |
| Entry / exit nodes | `entry_exit_points.csv` clusters | cluster centers (m) | n/a | Med (DBSCAN params; some clusters are projection artifacts) | ★★★★★ |
| Distance-to-{obstacle,boundary,entrance} | encoded CSV cols + rasters | 0.1 m grid | per point | Med (obstacle mask is coverage-derived, not verified geometry) | ★★★★☆ |
| Shift events (lateral) | `shift_flags_per_observation.is_shift` | per point | per frame | Med (definition-dependent) | ★★★☆☆ |
| Prediction error / behavior | MODEL_X arrays + per-window metrics | per window | window | Med (model has known straight-line/turn weakness) | ★★★☆☆ |

**Reliability headline:** the *aggregated, cell-binned* signals (density, speed, stops, dwell, flow direction, coherence, bottleneck) are the trustworthy backbone — averaging over many tracks washes out per-frame jitter. The *instantaneous geometric* signals (turn rate, curvature, acceleration) are real but require the documented N≈10 temporal smoothing before they read cleanly (see `project_temporal_denoising_v1`). Distance-field and obstacle-derived maps carry the coverage-artifact caveat.

---

## 3. Available spatial layers (render substrates)

Every map can be composited on one or more of these, all already aligned to world coordinates:

1. **Architectural plan underlay** — per-recording plan PNG warped plan→world via `calib.json` homography (method proven in `generate_bottleneck_map_test.py::warp_plan_to_world`). This is the authoritative visual frame and the house style (pale desaturated underlay).
2. **Walkable / obstacle / boundary masks** (0.1 m/px) — for masking, figure-ground, or "where the space is."
3. **Distance fields** (obstacle / boundary / entrance) — continuous scalar backdrops or analysis axes.
4. **Entry/exit cluster nodes** — anchor points for OD arcs and gravity maps.
5. **Raw world-track scatter** — the literal footprint, useful as a faint base layer.

**Caveat for any map that uses the plan frame:** the catalunya vertical-extent mismatch means out-of-frame cells must be clipped (already handled in the bottleneck script). Reuse that clipping logic everywhere.

---

## 4. Candidate map catalogue

Grouped by data family. Each entry: what it shows · required inputs · build status.

### A. Occupancy / density family
1. **Occupancy heatmap** — cumulative footfall density. Inputs: world x,y. Status: scriptable now (compute_heatmap.py exists).
2. **Density-class map** (transit vs crowd) — threshold occupancy into bands. Inputs: occupancy grid.
3. **Coverage / dead-zone map** — where tracks exist vs plan walkable area (⚠️ coverage artifact framing). Inputs: tracks + walkable mask.
4. **Temporal occupancy animation** — density evolving over time. Inputs: x,y,frame.

### B. Speed / friction family
5. **Mean-speed map** — where the space speeds up / slows people. Inputs: `speed_per_observation` → cell.
6. **Acceleration map** — where people start/stop accelerating (thresholds, openings). Inputs: derive Δspeed.
7. **Speed-variance / hesitation map** — within-cell speed spread = indecision. Inputs: speed per cell.

### C. Stop / dwell family
8. **Dwell-attractor map** — where people pause and stay (seating, views, meeting). Inputs: `linger_zones` + `dwell_events_enriched` (has `dwell_class`).
9. **Stop-event map** — discrete stop locations. Inputs: `stop_flags.is_stop`.
10. **Dwell-duration graduated map** — bubble size = total dwell time per zone. Inputs: `linger_zones.total_duration_s`.
11. **Transit-vs-destination classification** — per cell, throughput vs stay-time. Inputs: speed + dwell.

### D. Direction / flow family
12. **Directional flow field (quiver)** — dominant movement vectors. Inputs: `flow_field_cells`. Status: raw version exists.
13. **Desire-line / corridor (streamline) map** — integrate the flow field into continuous flow lines = how people *actually* route. Inputs: `flow_field_cells (mean_dx,dy)`. **Highest architectural payoff, not yet built.**
14. **Flow-coherence map** — `direction_consistency` as a field: ordered corridors vs chaotic milling. Inputs: `flow_field_cells.direction_consistency`.
15. **Conflict / crossing map** — high density × low direction consistency = where flows fight. Inputs: density + coherence (both in cells CSVs). **Strong friction story.**
16. **Bidirectional split map** — cells where two opposing headings coexist. Inputs: `flow_field_vectors` heading histogram per cell.

### E. Turning / decision family
17. **Turning-intensity map** — angular change per cell = where the space forces choices (decision nodes). Inputs: derive heading change (smooth N≈10). **Especially relevant to placa_espanya roundabout.**
18. **Decision-point / branch map** — cells where track headings fan out into multiple modes. Inputs: heading multimodality per cell.
19. **Sinuosity / wandering map** — per-track path/displacement ratio binned to space. Inputs: derive per track.

### F. Connectivity / route family
20. **Origin–Destination (OD) connectivity arcs** — entry/exit clusters joined by weighted arcs = how the space's portals connect. Inputs: `entry_exit_points` + track endpoints. **Reads as architecture instantly.**
21. **Route-preference / bundled desire-lines** — actual trajectories edge-bundled into dominant routes. Inputs: world tracks.
22. **Path-entropy / route-diversity map** — predictability of movement per cell (low entropy = channelized, high = free plaza). Inputs: heading distribution per cell. **Novel within thesis.**
23. **Transition map** — flow between named zones / quadrants as a chord/Sankey-on-plan. Inputs: tracks + zone definitions.

### G. Spatial-affordance family (uses encoded distance fields)
24. **Edge-affinity / wall-hugging map** — do people hug boundaries or use the center? `dist_to_boundary` distribution per cell. Inputs: encoded CSV. **Subtle but deeply architectural.**
25. **Entrance-gravity map** — how behavior (speed, dwell) varies with `dist_to_entrance`. Inputs: encoded CSV.
26. **Obstacle-avoidance map** — clearance kept around obstacles (⚠️ artifact caveat). Inputs: `dist_to_obstacle`.

### H. Model-derived (lower priority — model has known limits)
27. **Prediction-error map** — where the model fails spatially (FDE per region). Inputs: MODEL_X arrays.
28. **Predictability map** — regions where motion is/ isn't forecastable. Inputs: per-window metrics + positions.

---

## 5. Ranking table

Scoring each 1–5: **A** Architectural relevance · **B** Visual readability · **C** Presentation potential · **D** Animation potential · **E** Novelty within thesis. *Total* weights A and E slightly via judgment, but raw sum shown.

| # | Map | A | B | C | D | E | Σ | Data ready? |
|---|---|---|---|---|---|---|---|---|
| 13 | Desire-line / corridor (streamlines) | 5 | 5 | 5 | 5 | 4 | 24 | ✓ flow_field_cells |
| 8 | Dwell-attractor map | 5 | 5 | 5 | 4 | 3 | 22 | ✓ linger_zones |
| 20 | Origin–Destination arcs | 5 | 4 | 5 | 4 | 4 | 22 | ✓ entry_exit + tracks |
| 15 | Conflict / crossing map | 5 | 4 | 5 | 4 | 5 | 23 | ✓ density+coherence |
| 5 | Mean-speed / friction map | 4 | 5 | 4 | 4 | 3 | 20 | ✓ speed_per_obs |
| 14 | Flow-coherence map | 4 | 4 | 4 | 4 | 5 | 21 | ✓ direction_consistency |
| 17 | Turning-intensity / decision-node | 5 | 4 | 4 | 4 | 5 | 22 | ◐ derive heading (smooth) |
| 1 | Occupancy / density heatmap | 3 | 5 | 4 | 5 | 1 | 18 | ✓ tracks |
| 24 | Edge-affinity (wall-hugging) | 5 | 3 | 4 | 2 | 5 | 19 | ✓ dist_to_boundary |
| 22 | Path-entropy / route-diversity | 4 | 3 | 4 | 3 | 5 | 19 | ◐ derive heading dist |
| 21 | Route-preference (bundled) | 4 | 4 | 5 | 4 | 3 | 20 | ✓ tracks |
| 9 | Stop-event map | 4 | 4 | 4 | 4 | 2 | 18 | ✓ stop_flags |
| 11 | Transit-vs-destination | 5 | 4 | 4 | 3 | 4 | 20 | ✓ speed+dwell |
| 4 | Temporal occupancy animation | 3 | 4 | 4 | 5 | 3 | 19 | ✓ x,y,frame |
| 25 | Entrance-gravity map | 4 | 3 | 3 | 3 | 4 | 17 | ✓ dist_to_entrance |
| 6 | Acceleration map | 3 | 3 | 3 | 4 | 3 | 16 | ◐ derive |
| 3 | Coverage / dead-zone map | 3 | 4 | 3 | 2 | 3 | 15 | ✓ (⚠️ framing) |
| 27 | Prediction-error map | 2 | 3 | 3 | 3 | 4 | 15 | ✓ model arrays |

(Bottleneck map omitted from ranking — already built and polished; it is the established baseline.)

---

## 6. Recommended maps

Three tiers.

**Tier 1 — build first (architectural headline, data on disk):**
Desire-line/corridor streamlines (#13), Dwell-attractor (#8), Origin–Destination arcs (#20), Conflict/crossing (#15).
These four answer the four questions an architect asks of any space: *Where do people go? Where do they stop? How do the portals connect? Where does it break down?*

**Tier 2 — high value, reuse the same render pipeline:**
Mean-speed/friction (#5), Flow-coherence (#14), Turning-intensity/decision-node (#17), Transit-vs-destination (#11).

**Tier 3 — distinctive / novel, slightly more derivation:**
Edge-affinity wall-hugging (#24), Path-entropy/route-diversity (#22), Temporal occupancy animation (#4).

**Cross-cutting recommendation:** render the Tier-1 set **across all 7 recordings as a typology matrix** (corridor → stairs → esplanade → plaza → node). The same map read down the typology gradient is a stronger thesis artifact than any single hero image — it shows the *method generalizing across architectural types*, which is the thesis claim.

---

## 7. TOP 10 MAPS TO BUILD NEXT (prioritized shortlist)

> All reuse the proven `warp_plan_to_world` + architectural-underlay + plan-clip pipeline from `generate_bottleneck_map_test.py`. "Effort" assumes that pipeline as a starting library.

### 1. Desire-Line / Corridor Map  ⭐ flagship
- **Shows:** continuous flow streamlines integrated from the cell-averaged movement field — the routes people actually carve through the space vs. the designed circulation.
- **Inputs:** `flow_fields/flow_field_cells.csv` (`cell_x,cell_y,mean_dx,mean_dy,mean_speed`) + plan + calib.
- **Effort:** Low–Med (streamplot over a regridded vector field; main work is masking weak/empty cells and styling).
- **Visual style:** thin tapered flow lines, warm-on-pale-plan, line density ∝ `n_vectors`, optional speed-tinted. No quiver arrows — lines, not darts.
- **Animation:** High — animate tracer particles advecting along the field (the literal "motion pixels").
- **Why it matters:** This is the thesis thesis. One glance shows emergent circulation structure; particle animation is the signature Motion Pixels visual.

### 2. Dwell-Attractor Map
- **Shows:** where the space invites people to stop and stay — seating, viewpoints, meeting spots.
- **Inputs:** `linger_zones/linger_zones.csv` (centroid + `total_duration_s`, `unique_tracks`) + `dwell_events_enriched.csv` (`dwell_class`).
- **Effort:** Low (graduated symbols on plan; data pre-clustered).
- **Visual style:** soft radial glows or graduated circles sized by total dwell time, color by mean duration (brief→long); plan underlay.
- **Animation:** Med — "fill up" zones over time, or pulse by live occupancy.
- **Why it matters:** Directly maps program/use onto form — the architectural payoff of pedestrian data. Pairs as the inverse of the corridor map (stay vs flow).

### 3. Origin–Destination Connectivity Map
- **Shows:** how the space's portals connect — which entries feed which exits, and the dominant internal desire paths between them.
- **Inputs:** `entry_exit_points.csv` (cluster nodes) + track first/last points from `trajectories_world_filtered_250m.csv`.
- **Effort:** Med (assign track endpoints to nearest clusters; weight arcs; optional edge-bundling).
- **Visual style:** curved arcs between node circles, arc width ∝ flow volume, gentle bundling; nodes sized by throughput.
- **Animation:** Med–High — light pulses traveling along arcs.
- **Why it matters:** Turns the plaza into a graph — reads the space as a connectivity diagram an architect can act on (which thresholds matter, which are dead).

### 4. Conflict / Crossing Map
- **Shows:** friction points where high pedestrian density coincides with low directional agreement — where opposing flows collide.
- **Inputs:** `flow_field_cells.csv` (`n_vectors`/density × `direction_consistency`) — both columns already present.
- **Effort:** Low (cell field = density × (1 − coherence); same render as bottleneck).
- **Visual style:** sparse hot accents on calm plan; only the conflict cells light up (cool plan, sharp warm conflict points).
- **Animation:** Med — flicker/intensity tied to instantaneous crossing events.
- **Why it matters:** Pinpoints exactly where layout fails pedestrians — the most *actionable* map for a designer, and novel (no current figure shows it).

### 5. Mean-Speed / Friction Map
- **Shows:** where people accelerate (open, legible) vs slow (friction, congestion, threshold, ambiguity).
- **Inputs:** `metrics/speed_per_observation.csv` (`speed_smooth_m_s`) binned to 1 m cells over plan.
- **Effort:** Low (bin + smooth + render; reuse bottleneck rasterizer).
- **Visual style:** diverging blue(fast)→red(slow) or sequential; soft continuous field.
- **Animation:** Med — speed field over time-of-recording.
- **Why it matters:** Friction is a proxy for spatial quality; complements bottleneck (which is friction × crowding) by isolating pure pace.

### 6. Turning-Intensity / Decision-Node Map
- **Shows:** where the space forces directional choices — branch points, the roundabout core, stair landings.
- **Inputs:** derive heading from (world_x, world_y, frame) per track, smooth (N≈10 per `project_temporal_denoising_v1`), bin |Δheading| per cell.
- **Effort:** Med (heading derivation + denoising; must apply the documented smoothing to avoid jitter noise).
- **Visual style:** intensity field, hot at decision nodes; overlay flow lines for context.
- **Animation:** Med.
- **Why it matters:** Decision points are where architecture does cognitive work. `placa_espanya` (2,241 turns in model eval) is the natural hero recording. Novel for the thesis.

### 7. Flow-Coherence Map
- **Shows:** ordered channelized movement vs chaotic milling — distinguishes corridors from gathering spaces *behaviorally*.
- **Inputs:** `flow_field_cells.direction_consistency` (0–1) directly as a field.
- **Effort:** Low (column already exists; just render).
- **Visual style:** single-hue ramp, low=desaturated/chaotic, high=saturated/ordered; plan underlay.
- **Animation:** Low–Med.
- **Why it matters:** Behavioral classification of space type (corridor vs room) without any architectural label — pairs powerfully against the typology gradient. Cheap, novel.

### 8. Transit-vs-Destination Map
- **Shows:** a behavioral figure-ground — which areas are pass-through vs places people dwell.
- **Inputs:** per-cell speed (`speed_per_observation`) + dwell density (`dwell_events`).
- **Effort:** Med (combine two normalized fields into one bipolar index).
- **Visual style:** two-tone diverging (cool transit / warm destination) on plan; clean binary-ish read.
- **Animation:** Low.
- **Why it matters:** Collapses behavior into the single distinction architects care most about — circulation vs occupation — in one legible image.

### 9. Edge-Affinity / Wall-Hugging Map
- **Shows:** whether people hug boundaries or commit to open centers — the "wall effect" and how strongly edges shape movement.
- **Inputs:** `dist_to_boundary` (encoded CSV) per point, binned; or histogram of dist_to_boundary at occupied cells.
- **Effort:** Med (per-cell mean dist_to_boundary, or occupancy-weighted boundary-distance profile).
- **Visual style:** field tinted by boundary proximity of users; or a radial/section profile chart paired with the plan.
- **Animation:** Low.
- **Why it matters:** Quantifies a classic architectural intuition (edges as comfort) directly from behavior. Highly novel; differentiates open plaza (center-using) from corridor (edge-bound).

### 10. Temporal Occupancy Animation
- **Shows:** the space breathing — density building and dissolving over the recording.
- **Inputs:** `world_x, world_y, frame` from filtered trajectories.
- **Effort:** Low–Med (windowed KDE per frame-bin → frame sequence → encode).
- **Visual style:** soft density bloom on plan, warm; time ticker.
- **Animation:** High (it *is* the animation) — the natural motion-graphics centerpiece.
- **Why it matters:** The most immediately legible "alive" artifact for presentation; foundational density made temporal. Low risk, high impact for a talk/reel.

---

## 8. What is NOT recommended (and why)

- **Generic GIS/satellite basemaps** — out of scope; the warped architectural plan is the correct, more legible substrate already in hand.
- **Obstacle-avoidance map as a primary figure** — the obstacle mask is a coverage artifact (`project_encoding_truth_audit`); only safe as a caveated secondary layer.
- **Model prediction-error maps as hero images** — the model's straight-line/turn limitations (`project_model_x_audits`) make these a story about the *model*, not the *space*; keep them in the prediction track, not the behavioral-map deliverable.
- **Maps recommended purely because they're easy** (raw occupancy alone, raw quiver) — kept in the catalogue as building blocks but not in the top-10 hero set, since they don't pass the "learn something in 10 seconds" test on their own.

---

## 9. One-line verdict

The data already supports a coherent **suite of ~10 architecturally-legible behavioral maps across 7 typologically-diverse recordings**, all renderable through the existing plan-warp pipeline with no retracking, recalibration, or re-inference. Build the corridor / dwell / OD / conflict quartet first, render them as a typology matrix, and the behavioral-map layer of the thesis is made.

---
*End of audit. No maps were generated. Next step on approval: implement Top-10 #1 (Desire-Line / Corridor Map) as the pipeline template, then fan out.*
