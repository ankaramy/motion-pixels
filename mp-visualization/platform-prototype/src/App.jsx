import React, { useState, useCallback, useEffect } from 'react'
import TopNav from './components/TopNav'
import UploadStep from './components/UploadStep'
import CalibrationStep from './components/CalibrationStep'
import ProcessingStep from './components/ProcessingStep'
import PredictionCanvas from './components/PredictionCanvas'
import MetricsPanel from './components/MetricsPanel'
import TrackedVideoPanel from './components/TrackedVideoPanel'
import SimulationTimeline from './components/SimulationTimeline'
import { mockMetrics } from './data/mockPredictionData'

const WORKFLOW_STEPS = [
  { id: 'upload footage',           label: 'upload footage' },
  { id: 'upload spatial reference', label: 'upload spatial reference' },
  { id: 'align control points',     label: 'align control points' },
  { id: 'extract trajectories',     label: 'extract trajectories' },
  { id: 'encode spatial context',   label: 'encode spatial context' },
  { id: 'simulate future paths',    label: 'simulate future paths' },
]

// ── Cinematic intro overlay ──────────────────────────────────────────────
// Multi-stage status text. Fades out once layers are all painted.

const CINEMATIC_COPY = {
  plan:      { title: 'reconstructing spatial model',  tag: 'building geometry from calibrated frame' },
  observed:  { title: 'extracting behavioral traces',  tag: 'projecting tracked pedestrians onto plan' },
  predicted: { title: 'simulating future paths',       tag: 'model C — LSTM inference · 30s horizon' },
}

function CinematicOverlay({ phase }) {
  const copy = CINEMATIC_COPY[phase]
  const visible = !!copy

  return (
    <div
      className="cinematic-overlay"
      style={{ opacity: visible ? 1 : 0 }}
    >
      {copy && (
        <>
          <span className="cinematic-status-text pixel-status">{copy.title}</span>
          <div className="cinematic-progress">
            <div className="cinematic-progress-fill"/>
          </div>
          <span className="cinematic-tag">{copy.tag}</span>
        </>
      )}
    </div>
  )
}

// ── Home Screen ──────────────────────────────────────────────────────────
function HomeScreen({ onStart }) {
  return (
    <div style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
      {/* Background grid + faint trajectory traces */}
      <svg
        className="home-bg-svg"
        width="100%" height="100%"
        viewBox="0 0 1200 700"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <defs>
          <pattern id="home-grid-sm" width="36" height="36" patternUnits="userSpaceOnUse">
            <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#f2f2f2" strokeWidth="0.5"/>
          </pattern>
          <pattern id="home-grid-lg" width="180" height="180" patternUnits="userSpaceOnUse">
            <path d="M 180 0 L 0 0 0 180" fill="none" stroke="#ebebeb" strokeWidth="0.7"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="white"/>
        <rect width="100%" height="100%" fill="url(#home-grid-sm)"/>
        <rect width="100%" height="100%" fill="url(#home-grid-lg)"/>

        {/* Very faint plan boundary hint — light gray only */}
        <rect x="160" y="80" width="880" height="540" fill="none" stroke="#f0f0f0" strokeWidth="0.6"/>
      </svg>

      {/* Hero content */}
      <div className="home-content">
        <h1 className="home-title pixel-font">motion pixels</h1>
        <p className="home-subtitle">observe movement — predict behavior</p>
        <button className="home-cta" onClick={onStart}>
          start analysis
        </button>
      </div>
    </div>
  )
}

// ── Predict Screen ───────────────────────────────────────────────────────
function PredictScreen({ onReset }) {
  const [layers, setLayers] = useState({
    trajectories: true,
    predictions:  true,
    flow:         false,
    bottlenecks:  true,
    lingering:    true,
  })
  const [simHorizon, setSimHorizon] = useState(mockMetrics.simulationHorizon)
  const [numPeople,  setNumPeople]  = useState(mockMetrics.numberOfPeople)

  // Cinematic intro state
  const [cinematicPhase, setCinematicPhase] = useState('plan')

  useEffect(() => {
    const timers = [
      setTimeout(() => setCinematicPhase('observed'),  1400),
      setTimeout(() => setCinematicPhase('predicted'), 2600),
      setTimeout(() => setCinematicPhase('complete'),  3900),
    ]
    return () => timers.forEach(clearTimeout)
  }, [])

  const toggleLayer = useCallback(key => {
    setLayers(prev => ({ ...prev, [key]: !prev[key] }))
  }, [])

  return (
    <div className="predict-shell">
      {/* ── LEFT PANEL ── */}
      <div className="predict-left">
        {/* Workflow steps */}
        <div className="panel-section">
          <div className="panel-section-title">pipeline</div>
          {WORKFLOW_STEPS.map((s, i) => {
            const isDone   = i < WORKFLOW_STEPS.length - 1
            const isActive = i === WORKFLOW_STEPS.length - 1
            return (
              <div
                key={s.id}
                className={`workflow-step-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}
              >
                <span className="workflow-step-num">{String(i + 1).padStart(2, '0')}</span>
                <span className="workflow-step-label">{s.label}</span>
                {isDone && <span className="workflow-check">✓</span>}
              </div>
            )
          })}
        </div>

        {/* Session metadata */}
        <div className="panel-section">
          <div className="panel-section-title">session</div>
          {[
            ['dataset', 'input_skate_1.mov'],
            ['plan',    'top_view.png'],
            ['calib',   'calib.json'],
            ['model',   'Model C · LSTM'],
            ['status',  'mock data'],
          ].map(([k, v]) => (
            <div key={k} className="session-row">
              <span className="mono" style={{ fontSize: 8 }}>{k}</span>
              <span className="mono" style={{ fontSize: 8, color: 'var(--text)', textTransform: 'none', letterSpacing: '0.04em' }}>
                {v}
              </span>
            </div>
          ))}
        </div>

        {/* Tracked video panel */}
        <TrackedVideoPanel/>

        {/* Reset */}
        <div className="panel-section" style={{ marginTop: 'auto' }}>
          <button className="btn-secondary" style={{ width: '100%', fontSize: 9 }} onClick={onReset}>
            ← new analysis
          </button>
        </div>
      </div>

      {/* ── CENTER: canvas + timeline ── */}
      <div className="predict-canvas-col">
        <div className="predict-canvas-wrap" style={{ position: 'relative' }}>
          <PredictionCanvas layers={layers} cinematicPhase={cinematicPhase}/>
          <CinematicOverlay phase={cinematicPhase}/>
        </div>
        <SimulationTimeline simHorizon={simHorizon}/>
      </div>

      {/* ── RIGHT PANEL ── */}
      <div className="predict-right">
        <MetricsPanel
          layers={layers}
          onToggleLayer={toggleLayer}
          simHorizon={simHorizon}
          numPeople={numPeople}
          onSimHorizonChange={setSimHorizon}
          onNumPeopleChange={setNumPeople}
        />
      </div>
    </div>
  )
}

// ── Root App ─────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState('home')

  const navStep = screen === 'upload'     ? 'upload'
                : screen === 'calibrate'  ? 'align'
                : screen === 'processing' ? 'process'
                : screen === 'predict'    ? 'predict'
                : null

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'white' }}>
      <TopNav step={navStep} onBrandClick={() => setScreen('home')}/>

      {screen === 'home' && (
        <HomeScreen onStart={() => setScreen('upload')}/>
      )}
      {screen === 'upload' && (
        <UploadStep showPlan onNext={() => setScreen('calibrate')}/>
      )}
      {screen === 'calibrate' && (
        <CalibrationStep onNext={() => setScreen('processing')}/>
      )}
      {screen === 'processing' && (
        <ProcessingStep onNext={() => setScreen('predict')}/>
      )}
      {screen === 'predict' && (
        <PredictScreen key="predict" onReset={() => setScreen('home')}/>
      )}
    </div>
  )
}
