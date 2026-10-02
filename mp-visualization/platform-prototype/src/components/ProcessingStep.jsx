import React, { useState, useEffect } from 'react'

const STEPS = [
  { id: 'tracking',   label: 'extract trajectories',     detail: 'YOLOv8s + ByteTrack',     mockCount: () => `${Math.floor(Math.random() * 40 + 60)} tracks` },
  { id: 'homog',      label: 'compute homography',       detail: 'calibrate_homography.py', mockCount: () => `5 CP · err 0.28 m` },
  { id: 'metrics',    label: 'compute behavioral metrics', detail: 'speed · dwell · linger',  mockCount: () => `3 zones detected` },
  { id: 'encoding',   label: 'encode spatial context',   detail: 'trajectories_encoded.csv', mockCount: () => `10 features / track` },
  { id: 'prediction', label: 'simulate future paths',    detail: 'Model C — LSTM inference', mockCount: () => `${Math.floor(Math.random() * 5 + 8)} pred. paths` },
]

// Live counter that ticks upward as steps complete
function useTickingCounter(active, target) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (!active) return
    let cancelled = false
    let current = val
    function step() {
      if (cancelled) return
      const diff = target - current
      if (Math.abs(diff) < 1) { setVal(target); return }
      current += diff * 0.12 + Math.sign(diff) * 1
      setVal(Math.round(current))
      requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, active])
  return val
}

// Reconstruction preview — paths and scattered dots appear as steps complete
function ReconstructionPreview({ doneCount }) {
  const colors = ['#00bcd4', '#e91e63', '#ff9800', '#9c27b0', '#7cb342']
  const paths = [
    'M 14,108 C 50,95 100,82 165,70 C 215,62 260,57 305,54',
    'M 96,12  C 92,46 86,80 80,116 C 74,150 70,180 66,200',
    'M 12,52 L 60,68 L 110,82 L 165,94 L 220,100 L 280,104 L 312,106',
    'M 312,150 C 250,138 180,122 120,104 L 70,90 L 30,82 L 14,78',
    'M 36,200 C 70,180 115,154 158,128 L 200,104 L 250,80 L 295,60',
  ]
  const sampleDots = [
    [60, 50], [120, 65], [180, 78], [240, 88], [280, 95],
    [50, 110], [130, 92], [200, 105], [250, 125], [290, 140],
    [80, 145], [150, 130], [210, 145], [260, 165],
    [40, 175], [110, 165], [170, 175], [225, 188],
  ]
  return (
    <div className="processing-preview" style={{ width: 340, padding: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
        <span className="mono" style={{ fontSize: 7.5, color: 'var(--text-dim)' }}>
          trajectory preview
        </span>
        <span className="mono" style={{ fontSize: 7, color: 'var(--text-dim)', letterSpacing: '0.08em' }}>
          live
        </span>
      </div>

      <div style={{ position: 'relative', overflow: 'hidden', background: '#fcfcfc', border: '1px solid #f0f0f0' }}>
        <svg viewBox="0 0 320 220" width="100%" style={{ display: 'block' }}>
          <defs>
            <linearGradient id="scan-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"  stopColor="rgba(0,229,255,0)"/>
              <stop offset="50%" stopColor="rgba(0,229,255,0.55)"/>
              <stop offset="100%" stopColor="rgba(0,229,255,0)"/>
            </linearGradient>
          </defs>

          {/* Plan hint */}
          <rect x="6" y="6" width="308" height="208" fill="none" stroke="#eeeeee" strokeWidth="0.6"/>
          {/* Column grid */}
          {[40, 90, 140, 190, 240, 280].map(x =>
            [40, 100, 160, 200].map(y => (
              <circle key={`${x}-${y}`} cx={x} cy={y} r="1.6" fill="none" stroke="#eeeeee" strokeWidth="0.4"/>
            ))
          )}
          {/* Scattered detection dots (appear once tracking step is done) */}
          {doneCount >= 1 && sampleDots.map(([x, y], i) => (
            <circle key={`d-${i}`} cx={x} cy={y} r="1.2"
                    fill={colors[i % colors.length]}
                    opacity={0.55}
                    className="dot-drift"
                    style={{ animationDelay: `${(i % 6) * 0.3}s` }}/>
          ))}
          {/* Paths appear one by one */}
          {paths.slice(0, Math.max(0, doneCount - 1)).map((d, i) => (
            <path key={i} d={d} stroke={colors[i]} strokeWidth="1.2" fill="none"
                  opacity="0.6" strokeLinecap="round"/>
          ))}
          {/* Predicted dashes when last step is done */}
          {doneCount >= 5 && (
            <>
              <path d="M 305,54 C 320,52 332,50 348,48"
                    stroke="#00e5ff" strokeWidth="1.5" fill="none" opacity="0.85" strokeDasharray="4 3"/>
              <path d="M 295,60 C 312,55 326,50 342,46"
                    stroke="#ffea00" strokeWidth="1.5" fill="none" opacity="0.78" strokeDasharray="4 3"/>
            </>
          )}

          {/* Scanline overlay while still processing */}
          {doneCount < STEPS.length && (
            <g className="scanline">
              <rect x="0" y="0" width="320" height="22" fill="url(#scan-grad)"/>
            </g>
          )}
        </svg>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
        <span className="mono" style={{ fontSize: 7, color: 'var(--text-dim)' }}>frame 312 / 750</span>
        <span className="mono" style={{ fontSize: 7, color: 'var(--text-dim)' }}>25 fps</span>
      </div>
    </div>
  )
}

export default function ProcessingStep({ onNext }) {
  const [doneCount, setDoneCount] = useState(0)
  const [counts,    setCounts]    = useState({})

  // Target track count grows as step 1 completes
  const targetTracks  = doneCount >= 1 ? 312 : 0
  const tracksCount   = useTickingCounter(true, targetTracks)
  const elapsedTarget = Math.round(doneCount * 3.6)
  const elapsed       = useTickingCounter(true, elapsedTarget)

  useEffect(() => {
    if (doneCount >= STEPS.length) {
      const t = setTimeout(onNext, 1100)
      return () => clearTimeout(t)
    }
    const delay = doneCount === 0 ? 600 : 750 + Math.random() * 500
    const t = setTimeout(() => {
      const step = STEPS[doneCount]
      setCounts(prev => ({ ...prev, [step.id]: step.mockCount() }))
      setDoneCount(c => c + 1)
    }, delay)
    return () => clearTimeout(t)
  }, [doneCount, onNext])

  return (
    <div className="screen-centered" style={{ gap: 32 }}>
      <div style={{ textAlign: 'center' }}>
        <span className="upload-step-title">processing pipeline</span>
        <p className="step-screen-desc" style={{ marginTop: 8 }}>
          Frames are tracked, homography is solved, and behavioral context is encoded for Model C.
        </p>
      </div>

      <div style={{ display: 'flex', gap: 36, alignItems: 'flex-start' }}>
        {/* Step list */}
        <div className="processing-list" style={{ width: 320 }}>
          {STEPS.map((s, i) => {
            const isDone   = i < doneCount
            const isActive = i === doneCount
            return (
              <div key={s.id} className={`processing-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}>
                <div className={`processing-dot ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}/>
                <div style={{ flex: 1 }}>
                  <div className="processing-label">{s.label}</div>
                  <div className="mono" style={{ marginTop: 2, fontSize: 8, textTransform: 'none', letterSpacing: '0.04em', color: 'var(--text-dim)' }}>
                    {s.detail}
                  </div>
                  <div className="processing-progress-bar">
                    <div
                      className="processing-progress-fill"
                      style={{
                        width: isDone ? '100%' : isActive ? '60%' : '0%',
                        background: isDone ? 'var(--traj-5)' : 'var(--text)',
                      }}
                    />
                  </div>
                </div>
                <div className="processing-status">
                  {isDone   ? counts[s.id] || 'done'
                 : isActive ? '...'
                 :            '—'}
                </div>
              </div>
            )
          })}
        </div>

        {/* Reconstruction preview */}
        <ReconstructionPreview doneCount={doneCount}/>

        {/* Counters column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 22, padding: '0 8px' }}>
          <div>
            <div className="processing-counter-label">tracks extracted</div>
            <div className="processing-counter">{String(tracksCount).padStart(3, '0')}</div>
          </div>
          <div>
            <div className="processing-counter-label">predictions</div>
            <div className="processing-counter" style={{ fontSize: 32, color: doneCount >= 5 ? 'var(--traj-5)' : 'var(--text-dim)' }}>
              {doneCount >= 5 ? '12' : '—'}
            </div>
          </div>
          <div>
            <div className="processing-counter-label">duration</div>
            <div className="processing-counter" style={{ fontSize: 22 }}>
              00:{String(Math.min(elapsed, 18)).padStart(2, '0')}
            </div>
          </div>
        </div>
      </div>

      {doneCount >= STEPS.length && (
        <div style={{ textAlign: 'center' }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--traj-5)', letterSpacing: '0.2em' }}>
            pipeline complete — opening canvas
          </span>
        </div>
      )}
    </div>
  )
}
