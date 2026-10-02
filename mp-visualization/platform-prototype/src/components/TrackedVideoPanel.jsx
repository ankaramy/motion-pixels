import React from 'react'

// Mock pedestrians for the video panel
const MOCK_PEDS = [
  { id: 'P01', x: 26, y: 53, w: 14, h: 32, color: '#00bcd4', conf: 0.93 },
  { id: 'P02', x: 51, y: 58, w: 12, h: 30, color: '#e91e63', conf: 0.88 },
  { id: 'P04', x: 70, y: 50, w: 11, h: 28, color: '#ff9800', conf: 0.79, dashed: true },
  { id: 'P07', x: 11, y: 62, w:  9, h: 24, color: '#7cb342', conf: 0.84 },
  { id: 'P09', x: 84, y: 55, w:  9, h: 22, color: '#9c27b0', conf: 0.71 },
]

const TRACK_TAILS = [
  { color: '#00bcd4', pts: [[33, 96], [33, 90], [33, 84], [33, 78], [33, 72]] },
  { color: '#e91e63', pts: [[57, 96], [57, 90], [57, 84], [57, 78], [57, 72]] },
  { color: '#ff9800', pts: [[75, 92], [75, 86], [75, 80], [75, 74], [75, 68]] },
  { color: '#7cb342', pts: [[15, 96], [15, 90], [15, 84], [15, 78], [15, 70]] },
  { color: '#9c27b0', pts: [[88, 94], [88, 88], [88, 82], [88, 77], [88, 72]] },
]

export default function TrackedVideoPanel() {
  return (
    <div className="video-panel">
      <div className="video-panel-header">
        <div className="video-panel-rec">
          <span className="rec-dot"/>
          <span className="mono" style={{ fontSize: 7.5, letterSpacing: '0.12em' }}>VIDEO — TRACKED</span>
        </div>
        <span className="mono" style={{ fontSize: 7.5, color: 'var(--text-dim)' }}>00:18:42</span>
      </div>

      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <svg
          viewBox="0 0 100 100"
          width="100%" height="100%"
          preserveAspectRatio="xMidYMid slice"
          style={{ display: 'block', background: '#f4f3f0' }}
        >
          {/* Grayscale urban scene */}
          <rect width="100" height="100" fill="#f4f3f0"/>
          <rect y="52" width="100" height="48" fill="#eeede8"/>

          {/* Perspective ground grid */}
          {[0, 12, 24, 38, 50, 62, 76, 88, 100].map((x, i) => (
            <line key={i} x1={x} y1={100} x2={50} y2={50}
                  stroke="#e0dfdb" strokeWidth="0.32"/>
          ))}
          {[62, 72, 82, 92].map((y, i) => (
            <line key={i} x1={0} y1={y} x2={100} y2={y}
                  stroke="#e6e5e0" strokeWidth="0.3"/>
          ))}

          {/* Building facades */}
          <rect x="0"   y="0" width="18" height="52" fill="#d8d8d8"/>
          <rect x="82"  y="0" width="18" height="52" fill="#d8d8d8"/>
          <rect x="18"  y="0" width="22" height="52" fill="#e0e0e0"/>
          <rect x="60"  y="0" width="22" height="52" fill="#e0e0e0"/>
          <rect x="38"  y="10" width="24" height="42" fill="#e6e6e6"/>
          {/* Windows */}
          {[6, 14, 24, 32, 70, 78, 88, 96].map((x, i) => (
            <rect key={`w-${i}`} x={x} y={8 + (i % 2) * 12} width="3" height="6" fill="#c8c8c8"/>
          ))}
          {/* Door / archway */}
          <rect x="48" y="32" width="4" height="20" fill="#bcbcbc"/>
          {/* Horizon line */}
          <line x1="0" y1="51" x2="100" y2="51" stroke="#c8c8c8" strokeWidth="0.4"/>

          {/* Pedestrian silhouettes (grayscale, under boxes) */}
          {MOCK_PEDS.map(p => (
            <g key={`sil-${p.id}`} opacity="0.55">
              <rect x={p.x + p.w / 2 - 2.5} y={p.y + 5} width="5" height={p.h - 8} rx="0.8" fill="#9c9c9c"/>
              <circle cx={p.x + p.w / 2} cy={p.y + 3.5} r="2.5" fill="#9c9c9c"/>
            </g>
          ))}

          {/* Track tails (colored polyline trails) */}
          {TRACK_TAILS.map((tail, ti) => (
            <g key={ti}>
              <polyline
                points={tail.pts.map(p => p.join(',')).join(' ')}
                stroke={tail.color} strokeWidth="0.9" fill="none" opacity="0.7"
                strokeLinecap="round"
              />
              {tail.pts.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r={0.7}
                        fill={tail.color} opacity={0.35 + i * 0.12}/>
              ))}
            </g>
          ))}

          {/* Bounding boxes */}
          {MOCK_PEDS.map(p => (
            <g key={p.id}>
              <rect x={p.x} y={p.y} width={p.w} height={p.h}
                    fill="none"
                    stroke={p.color}
                    strokeWidth={p.dashed ? 0.7 : 0.9}
                    strokeDasharray={p.dashed ? '2 1.5' : undefined}
                    opacity="0.92"
              />
              {/* Center crosshair */}
              <line x1={p.x + p.w/2 - 1.5} y1={p.y + p.h/2} x2={p.x + p.w/2 + 1.5} y2={p.y + p.h/2}
                    stroke={p.color} strokeWidth="0.3" opacity="0.5"/>
              <line x1={p.x + p.w/2} y1={p.y + p.h/2 - 1.5} x2={p.x + p.w/2} y2={p.y + p.h/2 + 1.5}
                    stroke={p.color} strokeWidth="0.3" opacity="0.5"/>

              {/* ID tag */}
              <rect x={p.x} y={p.y - 4.5} width={11} height={4.5} fill={p.color} opacity="0.9"/>
              <text x={p.x + 0.8} y={p.y - 1.2}
                    fontSize="3.0" fontFamily="monospace" fontWeight="bold"
                    fill="white">
                {p.id}
              </text>
              {/* Confidence */}
              <text x={p.x + p.w + 0.8} y={p.y + 3.5}
                    fontSize="2.6" fontFamily="monospace"
                    fill={p.color} opacity="0.85">
                {p.conf}
              </text>
            </g>
          ))}

          {/* YOLOv8s badge */}
          <rect x="1.5" y="1.5" width="22" height="5.5" fill="rgba(0,0,0,0.55)" rx="0.5"/>
          <text x="2.5" y="5.5" fontSize="3.2" fontFamily="monospace" fontWeight="bold" fill="white">
            YOLOv8s
          </text>

          {/* REC badge */}
          <circle cx="76" cy="4" r="1.3" fill="#e53935"/>
          <text x="79" y="5.5" fontSize="3" fontFamily="monospace" fill="#333" opacity="0.8">REC</text>

          {/* Track count footer */}
          <rect x="0" y="93" width="100" height="7" fill="rgba(0,0,0,0.55)"/>
          <text x="2" y="98" fontSize="2.8" fontFamily="monospace" fill="white" opacity="0.95">
            5 active · 25 fps · ByteTrack · MOCK
          </text>
          <text x="98" y="98" fontSize="2.8" fontFamily="monospace" fill="white"
                textAnchor="end" opacity="0.85">
            00:18:42 / 00:30:00
          </text>
        </svg>
      </div>
    </div>
  )
}
