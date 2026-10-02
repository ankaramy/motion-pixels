import React from 'react'
import { mockMetrics, LAYER_COLORS } from '../data/mockPredictionData'

const LAYERS = [
  { key: 'trajectories', label: 'observed paths',  desc: 'past movement' },
  { key: 'predictions',  label: 'predicted paths', desc: 'future simulation' },
  { key: 'flow',         label: 'flow vectors',    desc: 'directional fields' },
  { key: 'bottlenecks',  label: 'bottlenecks',     desc: 'congestion zones' },
  { key: 'lingering',    label: 'lingering zones', desc: 'dwell fields' },
]

function MetricBar({ value, max, color }) {
  return (
    <div style={{ height: 2, background: '#f0f0f0', borderRadius: 1, marginTop: 3 }}>
      <div style={{
        height: '100%',
        width: `${Math.min(100, (value / max) * 100)}%`,
        background: color,
        borderRadius: 1,
        transition: 'width 0.4s',
      }}/>
    </div>
  )
}

// Soft layer swatch — a small visual preview of how the layer is drawn on canvas
function LayerSwatch({ layerKey, active }) {
  const color = LAYER_COLORS[layerKey]
  const dimmed = !active

  if (layerKey === 'predictions') {
    return (
      <svg width="20" height="14" style={{ flexShrink: 0, opacity: dimmed ? 0.3 : 1 }}>
        <line x1="0" y1="7" x2="20" y2="7" stroke={color} strokeWidth="1.5"
              strokeDasharray="3 2" strokeLinecap="round"/>
      </svg>
    )
  }
  if (layerKey === 'flow') {
    return (
      <svg width="20" height="14" style={{ flexShrink: 0, opacity: dimmed ? 0.3 : 1 }}>
        <line x1="0" y1="7" x2="15" y2="7" stroke={color} strokeWidth="1.2" strokeLinecap="round"/>
        <polygon points="15,4 20,7 15,10" fill={color}/>
      </svg>
    )
  }
  if (layerKey === 'bottlenecks') {
    return (
      <svg width="20" height="14" style={{ flexShrink: 0, opacity: dimmed ? 0.3 : 1 }}>
        <ellipse cx="10" cy="7" rx="9" ry="5" fill={color} fillOpacity="0.18"/>
        <ellipse cx="10" cy="7" rx="5" ry="3" fill={color} fillOpacity="0.45"/>
      </svg>
    )
  }
  if (layerKey === 'lingering') {
    return (
      <svg width="20" height="14" style={{ flexShrink: 0, opacity: dimmed ? 0.3 : 1 }}>
        <circle cx="10" cy="7" r="6.5" fill={color} fillOpacity="0.15"/>
        <circle cx="10" cy="7" r="3" fill={color} fillOpacity="0.42"/>
      </svg>
    )
  }
  // trajectories
  return (
    <svg width="20" height="14" style={{ flexShrink: 0, opacity: dimmed ? 0.3 : 1 }}>
      <path d="M 0,11 C 5,8 8,5 12,4 L 20,3" fill="none" stroke={color}
            strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  )
}

export default function MetricsPanel({
  layers, onToggleLayer,
  simHorizon, numPeople,
  onSimHorizonChange, onNumPeopleChange,
}) {
  return (
    <>
      {/* ── Parameters ── */}
      <div className="right-section">
        <div className="right-section-title">parameters</div>

        <div className="slider-row" style={{ marginBottom: 14 }}>
          <div className="slider-header">
            <span className="slider-label">simulation horizon</span>
            <span className="slider-val">{simHorizon} s</span>
          </div>
          <input type="range" min={5} max={120} step={5}
                 value={simHorizon} onChange={e => onSimHorizonChange(Number(e.target.value))}/>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
            {[0, 30, 60, 90, 120].map(v => (
              <span key={v} className="mono" style={{ fontSize: 7, color: 'var(--text-dim)' }}>{v}s</span>
            ))}
          </div>
        </div>

        <div className="slider-row">
          <div className="slider-header">
            <span className="slider-label">number of people</span>
            <span className="slider-val">{numPeople}</span>
          </div>
          <input type="range" min={5} max={200} step={1}
                 value={numPeople} onChange={e => onNumPeopleChange(Number(e.target.value))}/>
        </div>
      </div>

      {/* ── Behavioral layers ── */}
      <div className="right-section">
        <div className="right-section-title">behavioral layers</div>
        {LAYERS.map(l => (
          <div
            key={l.key}
            className="layer-toggle-row"
            onClick={() => onToggleLayer(l.key)}
          >
            <LayerSwatch layerKey={l.key} active={layers[l.key]}/>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="layer-name" style={{ opacity: layers[l.key] ? 1 : 0.45 }}>
                {l.label}
              </div>
              <div className="mono" style={{ fontSize: 7.5, textTransform: 'none', letterSpacing: '0.03em', color: 'var(--text-dim)', marginTop: 1 }}>
                {l.desc}
              </div>
            </div>
            <div className={`toggle-pill ${layers[l.key] ? 'on' : ''}`}
                 style={layers[l.key] ? { background: LAYER_COLORS[l.key] } : {}}
            />
          </div>
        ))}
      </div>

      {/* ── Metrics ── */}
      <div className="right-section">
        <div className="right-section-title">metrics</div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
          {[
            { label: 'avg flow',        value: mockMetrics.avgFlow,         unit: 'm/s',   max: 3,   color: '#7cb342' },
            { label: 'peak density',    value: mockMetrics.peakDensity,     unit: 'p/m²',  max: 8,   color: '#ff5722' },
            { label: 'avg speed',       value: mockMetrics.avgSpeed,        unit: 'm/s',   max: 3,   color: '#00bcd4' },
            { label: 'bottlenecks',     value: mockMetrics.bottleneckCount, unit: '',      max: 5,   color: '#ff9800' },
            { label: 'linger zones',    value: mockMetrics.lingerZoneCount, unit: '',      max: 6,   color: '#9c27b0' },
          ].map(m => (
            <div key={m.label}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>{m.label}</span>
                <span style={{ fontFamily: "'SF Mono', monospace", fontSize: 11, color: 'var(--text)' }}>
                  {m.value}{m.unit && ` ${m.unit}`}
                </span>
              </div>
              <MetricBar value={m.value} max={m.max} color={m.color}/>
            </div>
          ))}
        </div>
      </div>

      {/* ── Export ── */}
      <div className="right-section">
        <div className="right-section-title">export</div>
        <div className="export-btn-group">
          {[
            { label: 'export png',   sub: 'current view' },
            { label: 'export csv',   sub: 'trajectories + predictions' },
            { label: 'export video', sub: 'simulated sequence' },
          ].map(b => (
            <button key={b.label} className="export-btn">
              <span className="mono" style={{ fontSize: 8, color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
                {b.label}
              </span>
              <span className="mono" style={{ fontSize: 7, marginLeft: 'auto', color: 'var(--text-dim)', textTransform: 'none', letterSpacing: '0.03em' }}>
                {b.sub}
              </span>
            </button>
          ))}
        </div>
      </div>
    </>
  )
}
