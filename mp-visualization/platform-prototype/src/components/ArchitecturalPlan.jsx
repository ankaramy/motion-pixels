import React from 'react'

// Reusable architectural plan SVG.
// Renders as <g> elements — must be placed inside an <svg viewBox="0 0 800 520">.
// Entry openings: N(260-295) N(415-455) S(260-295) S(465-500) W(240-280) E(240-280)

export function PlanLayer({ opacity = 1 }) {
  return (
    <g style={{ opacity, transition: 'opacity 0.8s ease' }}>

      {/* ── Context mass (surrounding buildings) ── */}
      <rect x="0"   y="0"   width="800" height="80"  fill="#efefef" stroke="none"/>
      <rect x="0"   y="440" width="800" height="80"  fill="#efefef" stroke="none"/>
      <rect x="0"   y="80"  width="130" height="360" fill="#efefef" stroke="none"/>
      <rect x="670" y="80"  width="130" height="360" fill="#efefef" stroke="none"/>

      {/* Corner mass reinforcement (slightly darker) */}
      <rect x="0"   y="0"   width="130" height="80"  fill="#e8e8e8"/>
      <rect x="670" y="0"   width="130" height="80"  fill="#e8e8e8"/>
      <rect x="0"   y="440" width="130" height="80"  fill="#e8e8e8"/>
      <rect x="670" y="440" width="130" height="80"  fill="#e8e8e8"/>

      {/* Plaza interior projections (wings / arcaded side blocks) */}
      <rect x="130" y="80"  width="42" height="80"  fill="#e8e8e8" stroke="#d4d4d4" strokeWidth="0.5"/>
      <rect x="628" y="80"  width="42" height="80"  fill="#e8e8e8" stroke="#d4d4d4" strokeWidth="0.5"/>
      <rect x="130" y="360" width="42" height="80"  fill="#e8e8e8" stroke="#d4d4d4" strokeWidth="0.5"/>
      <rect x="628" y="360" width="42" height="80"  fill="#e8e8e8" stroke="#d4d4d4" strokeWidth="0.5"/>

      {/* ── Open plaza floor ── */}
      <rect x="130" y="80" width="540" height="360" fill="white"/>

      {/* ── Arcade / colonnade zone at N and S edges (shallow depth) ── */}
      <rect x="172" y="80" width="456" height="36" fill="#f9f9f9"/>
      <rect x="172" y="404" width="456" height="36" fill="#f9f9f9"/>

      {/* ── Plaza perimeter walls (with entry openings) ── */}

      {/* North wall: openings at x=260-295 and x=415-455 */}
      <line x1="172" y1="80" x2="260" y2="80" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="295" y1="80" x2="415" y2="80" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="455" y1="80" x2="628" y2="80" stroke="#c0c0c0" strokeWidth="1.5"/>

      {/* South wall: openings at x=260-295 and x=465-500 */}
      <line x1="172" y1="440" x2="260" y2="440" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="295" y1="440" x2="465" y2="440" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="500" y1="440" x2="628" y2="440" stroke="#c0c0c0" strokeWidth="1.5"/>

      {/* West wall: opening at y=235-275 */}
      <line x1="130" y1="160" x2="130" y2="235" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="130" y1="275" x2="130" y2="360" stroke="#c0c0c0" strokeWidth="1.5"/>

      {/* East wall: opening at y=235-275 */}
      <line x1="670" y1="160" x2="670" y2="235" stroke="#c0c0c0" strokeWidth="1.5"/>
      <line x1="670" y1="275" x2="670" y2="360" stroke="#c0c0c0" strokeWidth="1.5"/>

      {/* Arcade depth lines (colonnade inner edge) */}
      <line x1="172" y1="116" x2="628" y2="116" stroke="#d8d8d8" strokeWidth="0.7" strokeDasharray="3 5"/>
      <line x1="172" y1="404" x2="628" y2="404" stroke="#d8d8d8" strokeWidth="0.7" strokeDasharray="3 5"/>
      <line x1="172" y1="80"  x2="172" y2="440" stroke="#d8d8d8" strokeWidth="0.7" strokeDasharray="3 5"/>
      <line x1="628" y1="80"  x2="628" y2="440" stroke="#d8d8d8" strokeWidth="0.7" strokeDasharray="3 5"/>

      {/* Inner wall reveals (parallel lines, thickness hint) */}
      <line x1="172" y1="118.5" x2="628" y2="118.5" stroke="#e8e8e8" strokeWidth="0.4"/>
      <line x1="172" y1="401.5" x2="628" y2="401.5" stroke="#e8e8e8" strokeWidth="0.4"/>
      <line x1="174.5" y1="80" x2="174.5" y2="440" stroke="#e8e8e8" strokeWidth="0.4"/>
      <line x1="625.5" y1="80" x2="625.5" y2="440" stroke="#e8e8e8" strokeWidth="0.4"/>

      {/* Side-wall recesses (alcoves) */}
      <rect x="172" y="186" width="14" height="36" fill="#f3f3f3" stroke="#dcdcdc" strokeWidth="0.4"/>
      <rect x="172" y="298" width="14" height="36" fill="#f3f3f3" stroke="#dcdcdc" strokeWidth="0.4"/>
      <rect x="614" y="186" width="14" height="36" fill="#f3f3f3" stroke="#dcdcdc" strokeWidth="0.4"/>
      <rect x="614" y="298" width="14" height="36" fill="#f3f3f3" stroke="#dcdcdc" strokeWidth="0.4"/>

      {/* ── Arcade columns: North at y=98, step 44 ── */}
      {[196, 240, 284, 328, 372, 416, 460, 504, 548, 592].map((x, i) => (
        <circle key={`nc${i}`} cx={x} cy={98} r={4} fill="#e4e4e4" stroke="#c8c8c8" strokeWidth="0.6"/>
      ))}

      {/* South arcade columns at y=422 */}
      {[196, 240, 284, 328, 372, 416, 460, 504, 548, 592].map((x, i) => (
        <circle key={`sc${i}`} cx={x} cy={422} r={4} fill="#e4e4e4" stroke="#c8c8c8" strokeWidth="0.6"/>
      ))}

      {/* West arcade columns at x=148 */}
      {[135, 180, 225, 270, 315, 360, 405].map((y, i) => (
        <circle key={`wc${i}`} cx={148} cy={y} r={4} fill="#e4e4e4" stroke="#c8c8c8" strokeWidth="0.6"/>
      ))}

      {/* East arcade columns at x=652 */}
      {[135, 180, 225, 270, 315, 360, 405].map((y, i) => (
        <circle key={`ec${i}`} cx={652} cy={y} r={4} fill="#e4e4e4" stroke="#c8c8c8" strokeWidth="0.6"/>
      ))}

      {/* ── Interior column grid ── */}
      {[220, 285, 350, 400, 465, 530, 580].map((x, xi) =>
        [178, 260, 340].map((y, yi) => (
          <circle key={`ic${xi}-${yi}`} cx={x} cy={y} r={4.5}
            fill="#ebebeb" stroke="#cccccc" strokeWidth="0.8"/>
        ))
      )}

      {/* ── Paving grid (subtle pedestrian routes) ── */}
      {/* N–S main axis */}
      <line x1="400" y1="116" x2="400" y2="404"
            stroke="#ececec" strokeWidth="0.55" strokeDasharray="4 8"/>
      {/* E–W main axis */}
      <line x1="172" y1="260" x2="628" y2="260"
            stroke="#ececec" strokeWidth="0.55" strokeDasharray="4 8"/>
      {/* Diagonal cross-routes */}
      <line x1="172" y1="116" x2="628" y2="404"
            stroke="#f0f0f0" strokeWidth="0.45" strokeDasharray="3 10"/>
      <line x1="628" y1="116" x2="172" y2="404"
            stroke="#f0f0f0" strokeWidth="0.45" strokeDasharray="3 10"/>

      {/* Fine paving grid — east half */}
      {[200, 250, 300, 350].map(x => (
        <line key={`pvx-${x}`} x1={x} y1={120} x2={x} y2={400}
              stroke="#f4f4f4" strokeWidth="0.3" strokeDasharray="1 6"/>
      ))}
      {/* Fine paving grid — west half */}
      {[450, 500, 550, 600].map(x => (
        <line key={`pvx2-${x}`} x1={x} y1={120} x2={x} y2={400}
              stroke="#f4f4f4" strokeWidth="0.3" strokeDasharray="1 6"/>
      ))}
      {[160, 210, 310, 360].map(y => (
        <line key={`pvy-${y}`} x1={176} y1={y} x2={624} y2={y}
              stroke="#f4f4f4" strokeWidth="0.3" strokeDasharray="1 6"/>
      ))}

      {/* ── Central feature (raised platform / fountain) ── */}
      <rect x="348" y="222" width="104" height="76" rx="2"
            fill="#f6f6f6" stroke="#d4d4d4" strokeWidth="0.8"/>
      <rect x="362" y="234" width="76" height="52" rx="2"
            fill="#f2f2f2" stroke="#d8d8d8" strokeWidth="0.5"/>
      <circle cx="400" cy="260" r="18"
              fill="none" stroke="#d8d8d8" strokeWidth="0.8"/>
      <circle cx="400" cy="260" r="7"
              fill="none" stroke="#d4d4d4" strokeWidth="0.5"/>
      <circle cx="400" cy="260" r="1.5"
              fill="#d8d8d8"/>

      {/* ── Stair hints at entry openings (denser) ── */}
      {/* N-left entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`nls${offset}`} x1={260} y1={80 + offset} x2={295} y2={80 + offset}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}
      {/* N-right entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`nrs${offset}`} x1={415} y1={80 + offset} x2={455} y2={80 + offset}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}
      {/* S-left entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`sls${offset}`} x1={260} y1={440 - offset} x2={295} y2={440 - offset}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}
      {/* S-right entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`srs${offset}`} x1={465} y1={440 - offset} x2={500} y2={440 - offset}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}
      {/* W entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`ws${offset}`} x1={130 + offset} y1={235} x2={130 + offset} y2={275}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}
      {/* E entry stairs */}
      {[0, 4, 8, 12, 16, 20].map(offset => (
        <line key={`es${offset}`} x1={670 - offset} y1={235} x2={670 - offset} y2={275}
              stroke="#d4d4d4" strokeWidth="0.45"/>
      ))}

      {/* Door swing arcs at major entries */}
      <path d="M 260,80 A 18 18 0 0 1 278,98" fill="none" stroke="#dcdcdc" strokeWidth="0.4"/>
      <path d="M 415,80 A 18 18 0 0 1 433,98" fill="none" stroke="#dcdcdc" strokeWidth="0.4"/>

      {/* Small benches / planters near central feature */}
      <rect x="320" y="244" width="14" height="3" fill="#ececec" stroke="#d4d4d4" strokeWidth="0.3"/>
      <rect x="466" y="244" width="14" height="3" fill="#ececec" stroke="#d4d4d4" strokeWidth="0.3"/>
      <rect x="320" y="272" width="14" height="3" fill="#ececec" stroke="#d4d4d4" strokeWidth="0.3"/>
      <rect x="466" y="272" width="14" height="3" fill="#ececec" stroke="#d4d4d4" strokeWidth="0.3"/>

      {/* Tree / planter markers in corners */}
      {[[195, 130], [195, 390], [605, 130], [605, 390]].map(([x, y], i) => (
        <g key={`pln${i}`}>
          <circle cx={x} cy={y} r={6} fill="#f0f0f0" stroke="#dcdcdc" strokeWidth="0.5"/>
          <circle cx={x} cy={y} r={2.5} fill="none" stroke="#dcdcdc" strokeWidth="0.4"/>
        </g>
      ))}

      {/* ── Space label ── */}
      <text x="400" y="270" textAnchor="middle"
            fontSize="8" fontFamily="'SF Mono', monospace"
            fill="#e0e0e0" letterSpacing="0.25em">
        MAIN PLAZA
      </text>

      {/* ── Scale bar ── */}
      <g transform="translate(680, 500)">
        <line x1="0" y1="0" x2="50" y2="0" stroke="#c8c8c8" strokeWidth="1"/>
        <line x1="0" y1="-3" x2="0" y2="3" stroke="#c8c8c8" strokeWidth="1"/>
        <line x1="50" y1="-3" x2="50" y2="3" stroke="#c8c8c8" strokeWidth="1"/>
        <text x="25" y="-6" textAnchor="middle"
              fontSize="6.5" fontFamily="'SF Mono', monospace" fill="#c0c0c0" letterSpacing="0.08em">
          10 m
        </text>
      </g>

      {/* ── North arrow ── */}
      <g transform="translate(754, 28)">
        <polygon points="0,-14 -4,0 0,-5 4,0" fill="#c4c4c4"/>
        <polygon points="0,14  -4,0 0,5  4,0" fill="#e0e0e0"/>
        <text y="24" textAnchor="middle"
              fontSize="7" fontFamily="'SF Mono', monospace" fill="#bdbdbd" letterSpacing="0.1em">
          N
        </text>
      </g>

      {/* ── Dimension annotations ── */}
      <text x="142" y="262" fontSize="7" fontFamily="'SF Mono', monospace"
            fill="#d0d0d0" letterSpacing="0.06em"
            transform="rotate(-90 142 262)">
        ≈ 50 m
      </text>
      <text x="176" y="74" fontSize="7" fontFamily="'SF Mono', monospace"
            fill="#d0d0d0" letterSpacing="0.06em">
        ≈ 80 m
      </text>

    </g>
  )
}

// Standalone SVG wrapper for use in CalibrationStep etc.
export default function ArchitecturalPlan({ opacity, preserveAspectRatio = 'xMidYMid meet' }) {
  return (
    <svg
      viewBox="0 0 800 520"
      width="100%" height="100%"
      preserveAspectRatio={preserveAspectRatio}
      style={{ display: 'block' }}
    >
      <PlanLayer opacity={opacity}/>
    </svg>
  )
}
