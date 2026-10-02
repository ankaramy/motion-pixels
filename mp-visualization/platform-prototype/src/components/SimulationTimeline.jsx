import React, { useState, useEffect, useRef } from 'react'

const TOTAL_SECONDS = 30
const MARKERS = [0, 10, 20, 30]

export default function SimulationTimeline({ simHorizon = 30 }) {
  const [playing, setPlaying]   = useState(false)
  const [progress, setProgress] = useState(0)   // 0–1
  const animRef   = useRef(null)
  const lastTRef  = useRef(null)

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(animRef.current)
      lastTRef.current = null
      return
    }
    function tick(ts) {
      if (lastTRef.current) {
        const dt = (ts - lastTRef.current) / 1000
        setProgress(p => {
          const next = p + dt / simHorizon
          if (next >= 1) { setPlaying(false); return 1 }
          return next
        })
      }
      lastTRef.current = ts
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [playing, simHorizon])

  const currentTime = Math.round(progress * simHorizon)

  function seek(e) {
    const bar = e.currentTarget.getBoundingClientRect()
    const p = Math.max(0, Math.min(1, (e.clientX - bar.left) / bar.width))
    setProgress(p)
    setPlaying(false)
    lastTRef.current = null
  }

  return (
    <div className="timeline-bar">
      {/* Section label */}
      <span className="tl-label">simulation horizon</span>

      {/* Play/Pause button */}
      <button
        className="tl-play-btn"
        onClick={() => {
          if (progress >= 1) setProgress(0)
          setPlaying(p => !p)
        }}
        aria-label={playing ? 'pause' : 'play'}
      >
        {playing ? (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
            <rect x="1" y="0" width="3" height="10"/>
            <rect x="6" y="0" width="3" height="10"/>
          </svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
            <polygon points="1,0 10,5 1,10"/>
          </svg>
        )}
      </button>

      {/* Time display */}
      <span className="tl-time-display">
        {String(currentTime).padStart(2, '0')}s
      </span>

      {/* Track */}
      <div className="tl-track-wrap" onClick={seek}>
        {/* Marker labels */}
        <div className="tl-markers">
          {MARKERS.map(m => (
            <span key={m} className="tl-marker-label" style={{ left: `${(m / simHorizon) * 100}%` }}>
              {m}s
            </span>
          ))}
        </div>

        {/* Bar background */}
        <div className="tl-bar-bg">
          {/* Filled progress */}
          <div className="tl-bar-fill" style={{ width: `${progress * 100}%` }}/>
          {/* Playhead */}
          <div className="tl-playhead" style={{ left: `${progress * 100}%` }}/>
          {/* Tick marks */}
          {MARKERS.slice(1, -1).map(m => (
            <div
              key={m}
              className="tl-tick"
              style={{ left: `${(m / simHorizon) * 100}%` }}
            />
          ))}
        </div>
      </div>

      {/* Horizon endpoint label */}
      <span className="mono" style={{ fontSize: 8, color: 'var(--text-dim)', whiteSpace: 'nowrap', letterSpacing: '0.12em' }}>
        {simHorizon}s · model C
      </span>
    </div>
  )
}
