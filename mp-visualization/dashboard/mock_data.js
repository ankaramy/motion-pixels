/**
 * mock_data.js — Motion Pixels Dashboard
 * ----------------------------------------
 * All placeholder data lives here. When wiring real pipeline outputs,
 * replace these objects with fetched JSON from your pipeline outputs
 * (e.g. fetch('../outputs/metrics_summary.json')).
 *
 * The dashboard reads from window.MOCK at startup via app.js.
 */

window.MOCK = {

  // Session context — shown in header and status bar
  session: {
    dataset:    "input_skate_1.MOV",
    experiment: "Run 03 — 2025-03-19",
    frameCount: 4820,
    fps:        25,
    calibFile:  "calib.json",
    status:     "Mock Data",
  },

  // Scalar metrics — shown in the Metrics panel
  // Wire to: outputs/ metrics CSV or compute_metrics.py JSON export
  metrics: [
    { id: "avg_speed",   label: "Avg Speed",    value: "1.2",  unit: "m/s", delta: "+0.07", trend: "up"   },
    { id: "peak_density",label: "Peak Density",  value: "4.3",  unit: "/m²", delta: "+0.4",  trend: "up"   },
    { id: "total_dwell", label: "Total Dwell",   value: "5.2",  unit: "hr",  delta: "—",     trend: "flat" },
    { id: "tracks",      label: "Active Tracks", value: "12",   unit: "",    delta: "—",     trend: "flat" },
    { id: "crossings",   label: "Crossings",     value: "38",   unit: "",    delta: "+3",    trend: "up"   },
  ],

  // Layer toggle state — controls SVG group visibility in the spatial map
  // Wire to: user preferences or per-experiment config
  layers: [
    { id: "trajectories", label: "Trajectories", enabled: true  },
    { id: "heatmap",      label: "Heatmap",      enabled: true  },
    { id: "dwell",        label: "Dwell Zones",  enabled: false },
    { id: "ids",          label: "Track IDs",    enabled: true  },
    { id: "view3d",       label: "3D View",      enabled: false, disabled: true },
  ],

  // Observation notes — shown in the Notes panel
  // Wire to: a markdown file, a JSON sidecar, or manual annotations
  observations: [
    { type: "crossing", text: "High crossing concentration near central intersection." },
    { type: "dwell",    text: "Dwell cluster detected near curb zone (west side)." },
    { type: "flow",     text: "Dominant flow direction: SW → NE. Minor counter-flow on east corridor." },
    { type: "note",     text: "Reprojection median error: 0.30 m (5-pair RANSAC, calib.json)." },
    { type: "note",     text: "Adaptive speed thresholds — stop < 0.31 m/s, mid < 0.97 m/s (p20/p65)." },
    { type: "note",     text: "Scaffold overlay only. Replace with real pipeline outputs to extend." },
  ],

  // Dataset options for the header selector
  datasets: [
    "input_skate_1.MOV",
    "input_skate_2.MOV",
    "input_macba.MOV",
    "input_raval.MOV",
  ],

};
