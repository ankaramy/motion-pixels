import React from 'react'

const STEPS = ['upload', 'align', 'process', 'predict']

export default function TopNav({ step, onBrandClick }) {
  const stepIdx = STEPS.indexOf(step)

  return (
    <nav className="topnav">
      <span className="topnav-brand pixel-font" onClick={onBrandClick}>
        motion pixels
      </span>

      {stepIdx >= 0 && (
        <div className="topnav-step-indicator">
          {STEPS.map((s, i) => (
            <div
              key={s}
              className={`step-pip ${i < stepIdx ? 'done' : i === stepIdx ? 'active' : ''}`}
              title={s}
            />
          ))}
        </div>
      )}

      <div className="topnav-spacer" />

      <div className="topnav-links">
        <span className="topnav-link">dataset</span>
        <span className="topnav-link">pipeline</span>
        <span className="topnav-link">method</span>
        <span className="topnav-link">about</span>
      </div>
    </nav>
  )
}
