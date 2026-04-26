/**
 * app.js — Motion Pixels Dashboard
 * ----------------------------------
 * Lightweight scaffold logic only. No real data flow.
 *
 * Responsibilities:
 *  1. Populate dynamic content from mock_data.js (MOCK global)
 *  2. Render metric cards
 *  3. Render layer toggle checkboxes and wire them to SVG group visibility
 *  4. Render observation notes
 *  5. Populate dataset selector
 *
 * To wire real data: replace MOCK.* reads with fetch() calls to your
 * pipeline output JSONs (e.g. outputs/metrics_summary.json).
 */

document.addEventListener("DOMContentLoaded", () => {
  const M = window.MOCK;

  populateHeader(M);
  populateMetrics(M.metrics);
  populateLayers(M.layers);
  populateObservations(M.observations);
  populateDatasetSelector(M.datasets, M.session.dataset);
  updateStatusBar(M.session);
  animateFrameCounter(M.session.frameCount);
});


// ---- Header ----------------------------------------------------------------

function populateHeader(M) {
  const el = document.getElementById("experiment-label");
  if (el) el.textContent = M.session.experiment;
}

function populateDatasetSelector(datasets, current) {
  const sel = document.getElementById("dataset-select");
  if (!sel) return;
  datasets.forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === current) opt.selected = true;
    sel.appendChild(opt);
  });
}


// ---- Metrics ---------------------------------------------------------------

function populateMetrics(metrics) {
  const container = document.getElementById("metrics-grid");
  if (!container) return;

  metrics.forEach(m => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.setAttribute("data-metric", m.id);

    const deltaClass = m.trend === "up" ? "up" : m.trend === "down" ? "down" : "";

    card.innerHTML = `
      <div class="metric-label">${m.label}</div>
      <div class="metric-value">${m.value}<span class="metric-unit">${m.unit}</span></div>
      <div class="metric-delta ${deltaClass}">${m.delta !== "—" ? (m.trend === "up" ? "▲" : "▼") + " " + m.delta : "—"}</div>
    `;
    container.appendChild(card);
  });
}


// ---- Layer toggles ---------------------------------------------------------

function populateLayers(layers) {
  const container = document.getElementById("layer-list");
  if (!container) return;

  layers.forEach(layer => {
    const item = document.createElement("label");
    item.className = "layer-item" + (layer.disabled ? " disabled" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = layer.enabled;
    cb.disabled = !!layer.disabled;
    cb.setAttribute("data-layer", layer.id);

    // Wire checkbox to SVG group visibility
    cb.addEventListener("change", () => {
      setSpatialLayer(layer.id, cb.checked);
    });
    // Apply initial state
    setSpatialLayer(layer.id, layer.enabled);

    const label = document.createElement("span");
    label.className = "layer-label";
    label.textContent = layer.label;

    const hint = document.createElement("span");
    hint.className = "layer-hint";
    hint.textContent = layer.disabled ? "soon" : "";

    item.appendChild(cb);
    item.appendChild(label);
    item.appendChild(hint);
    container.appendChild(item);
  });
}

function setSpatialLayer(layerId, visible) {
  // Each SVG group carries class="layer-<id>" — toggle layer-hidden
  const groups = document.querySelectorAll(`#spatial-svg .layer-${layerId}`);
  groups.forEach(g => {
    g.classList.toggle("layer-hidden", !visible);
  });
}


// ---- Observations ----------------------------------------------------------

function populateObservations(obs) {
  const container = document.getElementById("obs-list");
  if (!container) return;

  obs.forEach(o => {
    const item = document.createElement("div");
    item.className = "obs-item";

    const dot = document.createElement("div");
    dot.className = `obs-dot ${o.type}`;

    const text = document.createElement("span");
    text.textContent = o.text;

    item.appendChild(dot);
    item.appendChild(text);
    container.appendChild(item);
  });
}


// ---- Status bar ------------------------------------------------------------

function updateStatusBar(session) {
  const el = document.getElementById("status-dataset");
  if (el) el.textContent = session.dataset;

  const cal = document.getElementById("status-calib");
  if (cal) cal.textContent = session.calibFile;

  const fps = document.getElementById("status-fps");
  if (fps) fps.textContent = `${session.fps} fps`;
}

// Mock frame counter — counts up slowly to give a sense of playback
function animateFrameCounter(total) {
  const el = document.getElementById("frame-counter");
  if (!el) return;

  let frame = 847; // start at a mid-point for the mock
  el.textContent = `Frame ${frame} / ${total}`;

  // Advance 1 frame every 40ms (fake playback at ~25fps display rate).
  // This is purely cosmetic — no real video processing.
  setInterval(() => {
    frame = (frame + 1) % total;
    el.textContent = `Frame ${frame} / ${total}`;
  }, 40);
}
