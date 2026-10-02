import React, { useState } from 'react'
import { PlanLayer } from './ArchitecturalPlan'

// ─────────────────────────────────────────────────────────────────────────────
// Control point registry
// vx/vy  — position in video frame  (viewBox 0 0 140 100)
// px/py  — position in plan          (viewBox 0 0 800 520)
// ─────────────────────────────────────────────────────────────────────────────
const CP = [
  { n: 1, color: '#5ec9c1', label: 'NW arcade',     vx: 28,  vy: 44, px: 196, py: 160, err: '0.18 m' },
  { n: 2, color: '#d97a74', label: 'NE arcade',     vx: 112, vy: 44, px: 604, py: 160, err: '0.22 m' },
  { n: 3, color: '#c98d4a', label: 'SW entry',      vx: 18,  vy: 75, px: 260, py: 400, err: '0.31 m' },
  { n: 4, color: '#9d88c4', label: 'SE entry',      vx: 122, vy: 70, px: 490, py: 400, err: '0.29 m' },
  { n: 5, color: '#7ab894', label: 'plaza center',  vx: 70,  vy: 56, px: 400, py: 260, err: '0.21 m' },
]

// Camera FOV polygon — corners of the calibrated area in both coordinate systems
const FOV_V = CP.filter(c => c.n !== 5).map(c => [c.vx, c.vy])   // 4 corners, video
const FOV_P = CP.filter(c => c.n !== 5).map(c => [c.px, c.py])   // 4 corners, plan

// Reusable CP marker: filled circle with white number + thin crosshair
function CPMarker({ x, y, n, color, r = 4.5, fontSize = 4.2 }) {
  return (
    <g>
      {/* Crosshair arms */}
      <line x1={x - r * 1.9} y1={y} x2={x - r * 1.1} y2={y} stroke={color} strokeWidth="0.55" opacity="0.75"/>
      <line x1={x + r * 1.1} y1={y} x2={x + r * 1.9} y2={y} stroke={color} strokeWidth="0.55" opacity="0.75"/>
      <line x1={x} y1={y - r * 1.9} x2={x} y2={y - r * 1.1} stroke={color} strokeWidth="0.55" opacity="0.75"/>
      <line x1={x} y1={y + r * 1.1} x2={x} y2={y + r * 1.9} stroke={color} strokeWidth="0.55" opacity="0.75"/>
      {/* Filled circle */}
      <circle cx={x} cy={y} r={r} fill={color} opacity="0.92"/>
      {/* White number */}
      <text x={x} y={y + fontSize * 0.38}
            textAnchor="middle"
            fontSize={fontSize}
            fontFamily="'SF Mono', monospace"
            fontWeight="700"
            fill="white">
        {n}
      </text>
    </g>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Video frame — perspective plaza view from mounted camera
// viewBox 0 0 140 100  →  perfectly fits the ~1.4:1 panel ratio
// ─────────────────────────────────────────────────────────────────────────────
function VideoPanel({ highlighted }) {
  const vp = { x: 70, y: 38 } // vanishing point

  return (
    <svg
      viewBox="0 0 140 100"
      width="100%" height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ display: 'block', background: '#edecea' }}
    >
      {/* Sky band */}
      <rect width="140" height="38" fill="#e6e4e0"/>

      {/* Left building mass */}
      <rect x="0"  y="0" width="22" height="38" fill="#d4d2ce"/>
      {/* Right building mass */}
      <rect x="118" y="0" width="22" height="38" fill="#d4d2ce"/>
      {/* Centre facade (back) */}
      <rect x="22" y="0" width="96" height="38" fill="#dddbd7"/>
      {/* Arcade frieze line */}
      <line x1="22" y1="35" x2="118" y2="35" stroke="#c8c6c2" strokeWidth="0.4"/>

      {/* Windows on left building */}
      {[5, 12].map(y => [3, 10].map(x => (
        <rect key={`lw-${x}-${y}`} x={x} y={y} width="4" height="6" fill="#c0beba" opacity="0.55"/>
      )))}
      {/* Windows on right building */}
      {[5, 12].map(y => [124, 131].map(x => (
        <rect key={`rw-${x}-${y}`} x={x} y={y} width="4" height="6" fill="#c0beba" opacity="0.55"/>
      )))}
      {/* Centre facade windows / openings */}
      {[26, 44, 58, 72, 86, 100, 114].map((x, i) => (
        <rect key={`cw-${i}`} x={x} y={8} width="7" height="14" rx="0.5" fill="#c8c6c0" opacity="0.45"/>
      ))}
      {/* Central arch / entrance */}
      <rect x="63" y="18" width="14" height="18" rx="1" fill="#bcb9b4" opacity="0.6"/>
      <path d="M 63,18 Q 70,12 77,18" fill="#bcb9b4" opacity="0.6"/>

      {/* Arcade columns (perspective) — foreground arches */}
      {[22, 38, 52, 88, 102, 118].map((x, i) => (
        <g key={`ac-${i}`}>
          <line x1={x} y1="35" x2={x} y2="100" stroke="#c8c6c0" strokeWidth="1.0" opacity="0.6"/>
        </g>
      ))}

      {/* Ground plane */}
      <rect x="0" y="38" width="140" height="62" fill="#e4e2de"/>

      {/* Perspective grid lines — to vanishing point */}
      {[0, 10, 22, 35, 50, 70, 90, 105, 118, 130, 140].map((x, i) => (
        <line key={`pv-${i}`} x1={x} y1={100} x2={vp.x} y2={vp.y}
              stroke="#d8d6d0" strokeWidth="0.28"/>
      ))}
      {/* Horizontal ground lines */}
      {[45, 54, 64, 75, 86, 96].map((y, i) => {
        const t = (y - 38) / 62
        const hw = 70 * (0.15 + t * 0.85)
        return <line key={`ph-${i}`} x1={70 - hw} y1={y} x2={70 + hw} y2={y}
                     stroke="#d4d2cc" strokeWidth="0.25"/>
      })}

      {/* Paving centre strip */}
      <polygon points="58,100 82,100 74,38 66,38" fill="none" stroke="#d0cec8" strokeWidth="0.3" opacity="0.6"/>

      {/* Pedestrian silhouettes — semi-transparent */}
      {[
        { x: 44, y: 58, h: 12 },
        { x: 62, y: 55, h: 11 },
        { x: 80, y: 56, h: 11 },
        { x: 96, y: 60, h: 12 },
        { x: 32, y: 65, h: 13 },
      ].map((p, i) => (
        <g key={`ped-${i}`} opacity="0.55">
          <rect x={p.x - 1.5} y={p.y} width="3" height={p.h * 0.65} rx="0.5" fill="#a8a6a0"/>
          <circle cx={p.x} cy={p.y - 1.5} r={p.h * 0.14} fill="#a8a6a0"/>
        </g>
      ))}

      {/* Camera FOV polygon — dashed blue outline */}
      <polygon
        points={FOV_V.map(p => p.join(',')).join(' ')}
        fill="rgba(30,100,220,0.04)"
        stroke="#4060c8"
        strokeWidth="0.55"
        strokeDasharray="2.5 2"
        opacity="0.60"
      />

      {/* Control point markers */}
      {CP.map(cp => (
        <g key={cp.n} style={{
          opacity: highlighted === null || highlighted === cp.n ? 1 : 0.35,
          transition: 'opacity 0.15s',
        }}>
          <CPMarker x={cp.vx} y={cp.vy} n={cp.n} color={cp.color} r={3.8} fontSize={3.4}/>
        </g>
      ))}

      {/* HUD overlays */}
      {/* Top-left: camera ID */}
      <rect x="1" y="1" width="32" height="5.5" fill="rgba(0,0,0,0.40)" rx="0.4"/>
      <text x="2.2" y="5.2" fontSize="3.2" fontFamily="monospace" fill="white" opacity="0.95">
        CAM-01 · FIXED
      </text>
      {/* Top-right: timestamp */}
      <rect x="106" y="1" width="33" height="5.5" fill="rgba(0,0,0,0.40)" rx="0.4"/>
      <text x="107.5" y="5.2" fontSize="3.2" fontFamily="monospace" fill="white" opacity="0.95">
        10:24:08 UTC
      </text>
      {/* Bottom-left: lens info */}
      <rect x="1" y="93.5" width="52" height="5.5" fill="rgba(0,0,0,0.30)" rx="0.4"/>
      <text x="2.2" y="97.6" fontSize="3.0" fontFamily="monospace" fill="white" opacity="0.85">
        f=4.2 mm · 1920×1080
      </text>
    </svg>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan panel — architectural plan with camera FOV + matched CPs
// viewBox 0 0 800 520  (same as PlanLayer)
// ─────────────────────────────────────────────────────────────────────────────
function PlanPanel({ highlighted }) {
  return (
    <svg
      viewBox="0 0 800 520"
      width="100%" height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ display: 'block', background: 'white' }}
    >
      {/* Architectural plan */}
      <PlanLayer/>

      {/* Camera FOV fill — light blue wash */}
      <polygon
        points={FOV_P.map(p => p.join(',')).join(' ')}
        fill="rgba(30,100,220,0.04)"
        stroke="#4060c8"
        strokeWidth="0.9"
        strokeDasharray="4 3.5"
        opacity="0.55"
      />

      {/* Camera position indicator — north wall mid */}
      <g transform="translate(400, 68)">
        <polygon points="0,-7 -5,4 0,1 5,4" fill="#4060c8" opacity="0.55"/>
        <text textAnchor="middle" y="13"
              fontSize="7.5" fontFamily="'SF Mono', monospace"
              fill="#8090b0" letterSpacing="0.08em">
          CAM-01
        </text>
      </g>

      {/* Control point markers */}
      {CP.map(cp => (
        <g key={cp.n} style={{
          opacity: highlighted === null || highlighted === cp.n ? 1 : 0.35,
          transition: 'opacity 0.15s',
        }}>
          <CPMarker x={cp.px} y={cp.py} n={cp.n} color={cp.color} r={9} fontSize={9}/>
          {/* Label */}
          <text x={cp.px + 12} y={cp.py - 3}
                fontSize="8" fontFamily="'SF Mono', monospace"
                fill={cp.color} opacity="0.85"
                letterSpacing="0.05em">
            {cp.label}
          </text>
          <text x={cp.px + 12} y={cp.py + 8}
                fontSize="7" fontFamily="'SF Mono', monospace"
                fill="#aaaaaa" letterSpacing="0.04em">
            err {cp.err}
          </text>
        </g>
      ))}
    </svg>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Calibration legend / CP list
// ─────────────────────────────────────────────────────────────────────────────
function CPLegend({ highlighted, onHover }) {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
      {CP.map(cp => (
        <div
          key={cp.n}
          onMouseEnter={() => onHover(cp.n)}
          onMouseLeave={() => onHover(null)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '3px 7px',
            border: `1px solid ${highlighted === cp.n ? cp.color : '#e8e8e8'}`,
            background: highlighted === cp.n ? `${cp.color}10` : 'white',
            cursor: 'default',
            transition: 'border-color 0.15s, background 0.15s',
          }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14">
            <circle cx="7" cy="7" r="6" fill={cp.color} opacity="0.9"/>
            <text x="7" y="10.2" textAnchor="middle"
                  fontSize="7" fontFamily="monospace" fontWeight="700" fill="white">
              {cp.n}
            </text>
          </svg>
          <span style={{
            fontFamily: "'SF Mono', monospace",
            fontSize: 9,
            color: highlighted === cp.n ? '#444' : '#999',
            letterSpacing: '0.04em',
            textTransform: 'lowercase',
          }}>
            {cp.label}
          </span>
          <span style={{
            fontFamily: "'SF Mono', monospace",
            fontSize: 8,
            color: highlighted === cp.n ? cp.color : '#ccc',
            letterSpacing: '0.04em',
          }}>
            {cp.err}
          </span>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main export
// ─────────────────────────────────────────────────────────────────────────────
export default function CalibrationStep({ onNext }) {
  const [highlighted, setHighlighted] = useState(null)

  const rmse = '0.28 m'

  return (
    <div className="calib-screen">
      {/* Header */}
      <div style={{ textAlign: 'center' }}>
        <div className="upload-step-title">align control points</div>
        <p className="step-screen-desc" style={{ marginTop: 5 }}>
          Match corresponding points in the camera frame and spatial reference.
          Homographic projection calibrates trajectories onto the plan.
        </p>
      </div>

      {/* Two panels */}
      <div className="calib-panels">
        {/* Video frame */}
        <div className="calib-panel">
          <div className="calib-panel-hdr">
            <span className="mono">video frame — camera view</span>
            <span className="mono" style={{ color: '#4060c8', fontSize: 8, letterSpacing: '0.06em' }}>
              calibrated region
            </span>
          </div>
          <div className="calib-panel-body">
            <VideoPanel highlighted={highlighted}/>
          </div>
        </div>

        {/* Connector */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          width: 28,
          flexShrink: 0,
          gap: 4,
          paddingTop: 32,
        }}>
          {CP.map(cp => (
            <div key={cp.n} style={{
              width: 8, height: 8,
              borderRadius: '50%',
              background: cp.color,
              opacity: highlighted === cp.n ? 1 : 0.3,
              transition: 'opacity 0.15s',
            }}/>
          ))}
        </div>

        {/* Plan */}
        <div className="calib-panel">
          <div className="calib-panel-hdr">
            <span className="mono">spatial reference — plan</span>
            <span className="mono" style={{ color: '#4060c8', fontSize: 8, letterSpacing: '0.06em' }}>
              camera FOV
            </span>
          </div>
          <div className="calib-panel-body">
            <PlanPanel highlighted={highlighted}/>
          </div>
        </div>
      </div>

      {/* CP legend — hover to highlight both panels */}
      <div style={{ width: '100%', maxWidth: 1080 }}>
        <CPLegend highlighted={highlighted} onHover={setHighlighted}/>
      </div>

      {/* Status + action row */}
      <div className="step-action-row" style={{ width: '100%', maxWidth: 1080 }}>
        <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 8.5 }}>
            5 / 5 control points matched
          </span>
          <div style={{ width: 1, height: 12, background: '#e0e0e0' }}/>
          <span className="mono" style={{ fontSize: 8.5, color: 'var(--traj-5)' }}>
            RMSE {rmse}
          </span>
          <div style={{ width: 1, height: 12, background: '#e0e0e0' }}/>
          <span className="mono" style={{ fontSize: 8.5, color: 'var(--text-muted)' }}>
            homography · valid · rank 3
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn-secondary">+ add point</button>
          <button className="btn-primary" onClick={onNext}>confirm calibration →</button>
        </div>
      </div>
    </div>
  )
}
