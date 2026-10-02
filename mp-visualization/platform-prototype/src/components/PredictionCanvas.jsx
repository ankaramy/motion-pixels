import React, { useRef, useState, useCallback } from 'react'
import { PlanLayer } from './ArchitecturalPlan'
import {
  mockPaths,
  mockPredictedPaths,
  mockBottlenecks,
  mockLingerZones,
  mockFlowVectors,
} from '../data/mockPredictionData'

const VB_W = 800
const VB_H = 520

function Arrow({ x, y, dx, dy, color }) {
  const ex = x + dx, ey = y + dy
  const angle = Math.atan2(dy, dx)
  const aLen = 8
  const ax1 = ex - aLen * Math.cos(angle - 0.34)
  const ay1 = ey - aLen * Math.sin(angle - 0.34)
  const ax2 = ex - aLen * Math.cos(angle + 0.34)
  const ay2 = ey - aLen * Math.sin(angle + 0.34)
  // Tail starts a little in from origin for a cleaner look
  const tx = x + dx * 0.10, ty = y + dy * 0.10
  return (
    <g stroke={color} fill={color} strokeLinecap="round">
      <line x1={tx} y1={ty} x2={ex} y2={ey} strokeWidth="0.9" opacity="0.65"/>
      <polygon points={`${ex},${ey} ${ax1},${ay1} ${ax2},${ay2}`} stroke="none" opacity="0.82"/>
    </g>
  )
}

// phase: 'plan' | 'observed' | 'predicted' | 'complete'
export default function PredictionCanvas({ layers, cinematicPhase = 'complete' }) {
  const svgRef = useRef(null)
  const [tooltip, setTooltip] = useState(null)

  const handleZoneEnter = useCallback((e, zone, type) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    setTooltip({
      x: e.clientX - rect.left + 14,
      y: e.clientY - rect.top  - 44,
      label: zone.label,
      detail: type === 'bottleneck' ? `density: ${zone.density}` : `avg dwell: ${zone.avgDwell}`,
      color: type === 'bottleneck' ? zone.color : '#9370b8',
    })
  }, [])
  const handleZoneLeave = useCallback(() => setTooltip(null), [])

  const observedVisible  = cinematicPhase === 'observed' || cinematicPhase === 'predicted' || cinematicPhase === 'complete'
  const predictedVisible = cinematicPhase === 'predicted' || cinematicPhase === 'complete'
  const uiVisible        = cinematicPhase === 'complete'

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        width="100%" height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
        aria-label="Motion Pixels — prediction canvas"
      >
        <defs>
          {/* Subtle glow for observed paths */}
          <filter id="path-glow" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation="1.0" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          {/* Gentle glow for predicted paths */}
          <filter id="pred-glow" x="-15%" y="-15%" width="130%" height="130%">
            <feGaussianBlur stdDeviation="1.6" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          {/* Watercolor zone filter — inner concentration */}
          <filter id="watercolor-zone" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="16"/>
          </filter>

          {/* Watercolor linger filter — outer bloom */}
          <filter id="watercolor-linger" x="-55%" y="-55%" width="210%" height="210%">
            <feGaussianBlur stdDeviation="28"/>
          </filter>
        </defs>

        {/* White background */}
        <rect width={VB_W} height={VB_H} fill="white"/>

        {/* ── Architectural plan ── */}
        <PlanLayer/>

        {/* ── Linger zones (lavender/purple wash) — below paths ── */}
        {layers.lingering && (
          <g style={{ opacity: observedVisible ? 1 : 0, transition: 'opacity 1.0s ease 0.2s' }}>
            {mockLingerZones.map(lz => (
              <g key={lz.id}
                 onMouseEnter={e => handleZoneEnter(e, lz, 'linger')}
                 onMouseLeave={handleZoneLeave}
                 style={{ cursor: 'crosshair' }}
              >
                {/* Outer bloom */}
                <circle cx={lz.cx} cy={lz.cy} r={lz.r * 1.7}
                        fill={lz.color} fillOpacity="0.06"
                        filter="url(#watercolor-linger)"/>
                {/* Inner concentration */}
                <circle cx={lz.cx} cy={lz.cy} r={lz.r * 0.9}
                        fill={lz.color} fillOpacity="0.13"
                        filter="url(#watercolor-zone)"/>
              </g>
            ))}
          </g>
        )}

        {/* ── Bottleneck zones (coral/orange wash) — below paths ── */}
        {layers.bottlenecks && (
          <g style={{ opacity: observedVisible ? 1 : 0, transition: 'opacity 1.0s ease 0.4s' }}>
            {mockBottlenecks.map(bn => (
              <g key={bn.id}
                 onMouseEnter={e => handleZoneEnter(e, bn, 'bottleneck')}
                 onMouseLeave={handleZoneLeave}
                 style={{ cursor: 'crosshair' }}
              >
                {/* Outer bloom */}
                <ellipse cx={bn.cx} cy={bn.cy} rx={bn.rx * 2.0} ry={bn.ry * 2.0}
                         fill={bn.color} fillOpacity="0.06"
                         filter="url(#watercolor-linger)"/>
                {/* Inner concentration */}
                <ellipse cx={bn.cx} cy={bn.cy} rx={bn.rx * 0.95} ry={bn.ry * 0.95}
                         fill={bn.color} fillOpacity="0.18"
                         filter="url(#watercolor-zone)"/>
              </g>
            ))}
          </g>
        )}

        {/* ── Observed paths — thin, solid, calm ── */}
        {layers.trajectories && (
          <g
            filter="url(#path-glow)"
            style={{ opacity: observedVisible ? 1 : 0, transition: 'opacity 1.0s ease' }}
          >
            {mockPaths.map(p => (
              <path
                key={p.id}
                d={p.d}
                stroke={p.color}
                strokeWidth={p.strokeWidth}
                fill="none"
                opacity={p.opacity}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </g>
        )}

        {/* ── Flow vectors — directional movement across space ── */}
        {layers.flow && (
          <g style={{ opacity: observedVisible ? 1 : 0, transition: 'opacity 0.8s ease 0.3s' }}>
            {mockFlowVectors.map((v, i) => (
              <Arrow key={i} x={v.x} y={v.y} dx={v.dx} dy={v.dy} color={v.color}/>
            ))}
          </g>
        )}

        {/* ── Predicted paths — slightly thicker, dashed, foregrounded ── */}
        {layers.predictions && (
          <g
            filter="url(#pred-glow)"
            style={{ opacity: predictedVisible ? 1 : 0, transition: 'opacity 1.0s ease' }}
          >
            {mockPredictedPaths.map(p => (
              <path
                key={p.id}
                d={p.d}
                stroke={p.color}
                strokeWidth={p.strokeWidth}
                fill="none"
                opacity={p.opacity}
                strokeLinecap="round"
                strokeDasharray="6 5"
              />
            ))}
          </g>
        )}

        {/* ── Canvas legend (bottom-left, inside canvas) ── */}
        {uiVisible && (
          <g transform="translate(148, 450)">
            <rect x="-4" y="-6" width="292" height="56" fill="rgba(255,255,255,0.92)" rx="1"/>

            {/* Row 1: path types */}
            <line x1="0" y1="7" x2="20" y2="7" stroke="#5ec9c1" strokeWidth="1.3" strokeLinecap="round"/>
            <text x="24" y="11" fontSize="8" fontFamily="'SF Mono', monospace" fill="#777">observed paths</text>

            <line x1="108" y1="7" x2="128" y2="7" stroke="#c85c55" strokeWidth="1.6"
                  strokeLinecap="round" strokeDasharray="5 4"/>
            <text x="132" y="11" fontSize="8" fontFamily="'SF Mono', monospace" fill="#777">predicted paths</text>

            {/* Row 2: zones + flow */}
            <ellipse cx="8" cy="30" rx="7" ry="5" fill="#d4724a" fillOpacity="0.32"/>
            <text x="18" y="34" fontSize="8" fontFamily="'SF Mono', monospace" fill="#777">bottleneck</text>

            <circle cx="108" cy="30" r="5.5" fill="#9370b8" fillOpacity="0.28"/>
            <text x="118" y="34" fontSize="8" fontFamily="'SF Mono', monospace" fill="#777">lingering</text>

            <line x1="190" y1="30" x2="208" y2="30" stroke="#7ab894" strokeWidth="1.1" strokeLinecap="round"/>
            <polygon points="208,27 214,30 208,33" fill="#7ab894"/>
            <text x="218" y="34" fontSize="8" fontFamily="'SF Mono', monospace" fill="#777">flow</text>
          </g>
        )}

        {/* Mock data badge */}
        <text
          x="148" y="512"
          fontSize="7"
          fontFamily="'SF Mono', monospace"
          fill="#c8c8c8"
          letterSpacing="0.14em"
        >
          MOCK DATA · MOTION PIXELS THESIS
        </text>
      </svg>

      {/* Zone tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x,
            top:  tooltip.y,
            background: 'white',
            border: '1px solid #e8e8e8',
            padding: '5px 9px',
            fontFamily: "'SF Mono', monospace",
            fontSize: 9,
            letterSpacing: '0.06em',
            color: '#555',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
            boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          }}
        >
          <div style={{ color: tooltip.color, fontWeight: 600, marginBottom: 2 }}>
            {tooltip.label}
          </div>
          <div>{tooltip.detail}</div>
        </div>
      )}
    </div>
  )
}
