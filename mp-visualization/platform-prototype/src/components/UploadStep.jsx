import React, { useState } from 'react'

function UploadBox({ label, hint, subhint, filled, filename, onSimulateUpload, accentColor }) {
  return (
    <div
      className={`upload-area-box ${filled ? 'filled' : ''}`}
      style={filled ? { '--accent': accentColor } : {}}
      onClick={onSimulateUpload}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onSimulateUpload()}
    >
      <div className="upload-box-icon">
        {filled ? (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={accentColor} strokeWidth="1.2">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        ) : (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#c8c8c8" strokeWidth="1.2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        )}
      </div>
      {filled ? (
        <span className="upload-filename" style={{ color: accentColor }}>{filename}</span>
      ) : (
        <>
          <span className="upload-box-label">{label}</span>
          <span className="upload-box-hint">{hint}</span>
          {subhint && <span className="upload-box-subhint">{subhint}</span>}
        </>
      )}
      <span className="upload-box-tap-hint">{filled ? '' : 'click to select'}</span>
    </div>
  )
}

export default function UploadStep({ showPlan, onNext }) {
  const [videoFilled, setVideoFilled] = useState(false)
  const [planFilled,  setPlanFilled]  = useState(false)

  const canAdvance = showPlan ? videoFilled && planFilled : videoFilled

  return (
    <div className="screen-centered" style={{ gap: 36 }}>
      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <div className="upload-step-title">pair movement with space</div>
        <p className="step-screen-desc" style={{ marginTop: 8 }}>
          Upload your pedestrian footage and the corresponding spatial reference.
          <br/>Movement and space will be aligned.
        </p>
      </div>

      {/* Side-by-side upload boxes */}
      <div className="upload-pair">
        <UploadBox
          label="upload footage"
          hint="pedestrian video"
          subhint="mp4 · mov · avi"
          filled={videoFilled}
          filename="input_footage.mp4  ✓"
          accentColor="var(--traj-1)"
          onSimulateUpload={() => setVideoFilled(true)}
        />

        {/* Connector */}
        <div className="upload-connector">
          <span className="upload-connector-line"/>
          <span className="mono" style={{ fontSize: 8, color: 'var(--text-dim)', padding: '0 8px' }}>+</span>
          <span className="upload-connector-line"/>
        </div>

        <UploadBox
          label="upload spatial reference"
          hint="top-down plan or site image"
          subhint="png · jpg · svg"
          filled={planFilled}
          filename="top_view_plan.png  ✓"
          accentColor="var(--traj-4)"
          onSimulateUpload={() => setPlanFilled(true)}
        />
      </div>

      {/* Action */}
      <div className="step-action-row" style={{ width: '100%', maxWidth: 680 }}>
        <span className="mono" style={{ fontSize: 9 }}>
          {canAdvance
            ? 'both files ready — continue to calibration'
            : `click each panel to simulate upload (${[!videoFilled && 'footage', !planFilled && 'reference'].filter(Boolean).join(' + ')} needed)`
          }
        </span>
        <button className="btn-primary" disabled={!canAdvance} onClick={onNext}>
          continue →
        </button>
      </div>
    </div>
  )
}
