// All coordinates are in viewBox space: 0 0 800 520
// Open plaza area: x=130-670, y=80-440
// Entry points: N(270,80) N(430,80) S(270,440) S(490,440) W(130,255) E(670,255)
//
// Color palette: soft watercolor / transparent-ink tones
// Soft teal:   #5ec9c1  Muted rose: #d97a74  Warm amber: #c98d4a
// Lavender:    #9d88c4  Sage:       #7ab894  Pale blue:  #7aacc8
// Coral (BN):  #d4724a  Purple (LZ): #9370b8

// ── OBSERVED PATHS ────────────────────────────────────────────────────────
// 12 paths with clear hierarchy: primary (1.2-1.3px, 0.45-0.55), secondary (0.9-1.0px, 0.28-0.36)

export const mockPaths = [
  // ── Primary: W→E horizontal flow ──
  {
    id: 'track_01',
    color: '#5ec9c1', strokeWidth: 1.3, opacity: 0.52,
    d: 'M 130,215 L 195,212 L 265,208 C 330,204 390,200 455,196 C 520,192 590,189 660,188 L 670,188',
  },
  {
    id: 'track_02',
    color: '#4ab8b0', strokeWidth: 1.0, opacity: 0.34,
    d: 'M 130,290 C 200,285 290,278 380,273 C 460,268 550,265 640,263 L 670,262',
  },
  // ── Primary: E→W flow ──
  {
    id: 'track_03',
    color: '#5ec9c1', strokeWidth: 1.1, opacity: 0.40,
    d: 'M 670,230 L 600,234 L 520,238 L 440,242 C 370,246 300,250 230,252 L 165,254 L 130,255',
  },
  // ── Primary: N→S ──
  {
    id: 'track_04',
    color: '#d97a74', strokeWidth: 1.3, opacity: 0.50,
    d: 'M 270,80 C 268,140 264,200 258,255 C 252,312 248,370 244,440',
  },
  {
    id: 'track_05',
    color: '#c86e68', strokeWidth: 1.0, opacity: 0.35,
    d: 'M 430,80 C 432,135 435,195 438,250 C 441,305 444,365 448,440',
  },
  // ── Primary: Diagonal NW→SE ──
  {
    id: 'track_06',
    color: '#c98d4a', strokeWidth: 1.3, opacity: 0.46,
    d: 'M 145,100 L 200,135 L 265,175 L 330,220 C 390,262 440,295 510,335 L 575,372 L 635,408 L 670,428',
  },
  // ── Primary: Diagonal NE→SW ──
  {
    id: 'track_07',
    color: '#b87e3e', strokeWidth: 1.0, opacity: 0.34,
    d: 'M 670,105 L 610,148 L 545,192 L 480,234 C 415,272 365,300 305,338 L 240,378 L 185,415 L 145,438',
  },
  // ── Primary: Angular L-path (lavender) ──
  {
    id: 'track_08',
    color: '#9d88c4', strokeWidth: 1.2, opacity: 0.44,
    d: 'M 270,80 L 272,125 L 280,165 L 310,195 L 355,210 L 400,215 L 440,208 L 470,185 L 475,150 L 472,105 L 465,80',
  },
  // ── Secondary: Angular cluster (lavender, fainter) ──
  {
    id: 'track_09',
    color: '#8e7ab4', strokeWidth: 0.9, opacity: 0.32,
    d: 'M 130,350 L 180,335 L 245,315 L 310,295 L 360,280 L 395,268 L 420,268 L 445,275 L 470,290 L 490,310 L 495,335 L 490,360 L 475,380 L 450,395 L 415,400',
  },
  // ── Secondary: Blue horizontal ──
  {
    id: 'track_10',
    color: '#7aacc8', strokeWidth: 0.9, opacity: 0.28,
    d: 'M 130,145 L 200,148 L 280,150 L 360,148 L 440,145 L 520,143 L 600,140 L 660,138',
  },
  // ── Secondary: Sage diagonal ──
  {
    id: 'track_11',
    color: '#7ab894', strokeWidth: 0.9, opacity: 0.28,
    d: 'M 145,95 C 195,120 255,155 320,195 L 380,235 L 425,270 C 465,302 505,335 555,368 L 610,400 L 660,428',
  },
  // ── Secondary: Cluster near bottleneck ──
  {
    id: 'track_12',
    color: '#d97a74', strokeWidth: 1.0, opacity: 0.36,
    d: 'M 330,200 L 355,225 L 370,255 L 378,285 L 370,315 L 352,335 L 330,342 L 308,335 L 292,315 L 285,285 L 292,255 L 310,228 L 330,212',
  },
]

// ── PREDICTED PATHS ───────────────────────────────────────────────────────
// 7 paths. Slightly thicker, dashed, slightly more saturated — foregrounded.

export const mockPredictedPaths = [
  // N→S exits
  {
    id: 'pred_01',
    color: '#38b2aa', strokeWidth: 1.8, opacity: 0.78,
    d: 'M 244,440 C 242,455 240,468 239,482',
  },
  {
    id: 'pred_02',
    color: '#c85c55', strokeWidth: 1.8, opacity: 0.76,
    d: 'M 448,440 C 449,458 450,470 451,484',
  },
  // W exit continuation
  {
    id: 'pred_03',
    color: '#5ec9c1', strokeWidth: 1.6, opacity: 0.70,
    d: 'M 130,255 C 106,256 82,257 58,258',
  },
  // N exit from bottleneck cluster
  {
    id: 'pred_04',
    color: '#c98d4a', strokeWidth: 1.8, opacity: 0.76,
    d: 'M 400,215 C 402,170 405,130 408,90 L 410,80',
  },
  // E exit upper
  {
    id: 'pred_05',
    color: '#7aacc8', strokeWidth: 1.6, opacity: 0.68,
    d: 'M 670,188 C 700,184 730,180 758,176',
  },
  // From linger zone → S
  {
    id: 'pred_06',
    color: '#9370b8', strokeWidth: 1.6, opacity: 0.66,
    d: 'M 415,400 L 440,422 L 460,440 C 472,452 480,465 484,478',
  },
  // E exit lower (diagonal)
  {
    id: 'pred_07',
    color: '#7ab894', strokeWidth: 1.5, opacity: 0.65,
    d: 'M 670,428 C 698,420 722,412 746,404',
  },
]

// ── BOTTLENECK ZONES ──────────────────────────────────────────────────────
// 2 zones — warm coral / orange wash

export const mockBottlenecks = [
  {
    id: 'bn_01',
    cx: 390, cy: 258,
    rx: 66, ry: 48,
    color: '#d4724a',
    label: 'bottleneck A',
    density: '4.1 p/m²',
  },
  {
    id: 'bn_02',
    cx: 282, cy: 210,
    rx: 42, ry: 30,
    color: '#c86030',
    label: 'bottleneck B',
    density: '3.2 p/m²',
  },
]

// ── LINGERING ZONES ───────────────────────────────────────────────────────
// 3 zones — lavender / purple wash, spatially separated

export const mockLingerZones = [
  {
    id: 'lz_01',
    cx: 218, cy: 318,
    r: 50,
    color: '#9370b8',
    label: 'linger zone 1',
    avgDwell: '5.2 s',
  },
  {
    id: 'lz_02',
    cx: 560, cy: 205,
    r: 42,
    color: '#8460a8',
    label: 'linger zone 2',
    avgDwell: '3.8 s',
  },
  {
    id: 'lz_03',
    cx: 408, cy: 392,
    r: 44,
    color: '#9370b8',
    label: 'linger zone 3',
    avgDwell: '4.5 s',
  },
]

// ── FLOW VECTORS ──────────────────────────────────────────────────────────
// 12 vectors — longer, directional, well-spaced

export const mockFlowVectors = [
  // W→E main flow
  { x: 158, y: 212, dx: 82, dy: -10, color: '#5ec9c1' },
  { x: 288, y: 205, dx: 80, dy: -6,  color: '#5ec9c1' },
  { x: 430, y: 200, dx: 76, dy: -3,  color: '#4ab8b0' },
  // N→S main flow
  { x: 265, y: 112, dx:  6, dy: 80,  color: '#d97a74' },
  { x: 432, y: 108, dx:  5, dy: 76,  color: '#d97a74' },
  // Diagonal cross-flows
  { x: 155, y: 104, dx: 58, dy: 54,  color: '#c98d4a' },
  { x: 610, y: 104, dx:-54, dy: 54,  color: '#c98d4a' },
  { x: 155, y: 400, dx: 54, dy:-50,  color: '#b87e3e' },
  { x: 610, y: 400, dx:-50, dy:-48,  color: '#b87e3e' },
  // Dispersal from central zone
  { x: 390, y: 258, dx:-72, dy:-14,  color: '#7ab894' },
  { x: 390, y: 258, dx: 72, dy: 14,  color: '#7ab894' },
  { x: 390, y: 258, dx: 12, dy:-74,  color: '#68a882' },
]

export const mockMetrics = {
  simulationHorizon: 30,
  numberOfPeople: 47,
  avgFlow: 1.2,
  bottleneckCount: 2,
  lingerZoneCount: 3,
  peakDensity: 4.1,
  avgSpeed: 1.1,
}

export const LAYER_COLORS = {
  trajectories: '#5ec9c1',
  predictions:  '#c85c55',
  flow:         '#7ab894',
  bottlenecks:  '#d4724a',
  lingering:    '#9370b8',
}
