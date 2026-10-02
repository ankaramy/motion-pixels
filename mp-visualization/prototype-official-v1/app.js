const h = React.createElement;
const BUILD_STATUS = "OFFICIAL_RELEASE";
const BUILD_VERSION = "V1";
class MotionPixelsApp extends React.Component {

  constructor(props){
    super(props);
    this.videoInputRef = React.createRef();
    this.planInputRef = React.createRef();
    this.cx = 400; this.cy = 262;
    this.sharedFrame = {
      id: 'mp-shared-plan-world-v1',
      viewBox: { width: 800, height: 520 },
      origin: { x: 400, y: 262 },
      metersPerPixel: 0.125,
      zScale: 0.72,
      projection: { xSkew: -0.42, ySkew: -0.58 },
    };
    this.HF = {40:0.5, 60:0.74, 100:1, 200:1.7, 400:2.6};
    this.HSEC = {40:18, 60:26, 100:42, 200:72, 400:120};
    const resetDemoState = this.shouldResetDemoState();
    if(resetDemoState) this.clearDemoStorage();
    this.state = {
      screen: 'home',
      profileOpen: false,
      exportOpen: false,
      saveOpen: false,
      saveSelected: null,
      saveNewName: '',
      saveConfirm: null,
      exportSnapshotSrc: null,
      exportSnapshotSvg: null,
      savedStudies: [],
      // workspace
      layers: { speed:false, flow:false, bottlenecks:false, predictions:false },
      aesthetic: 'ink',
      horizon: 100,
      playbackSpeed: 1,
      playing: false,
      progress: 0,
      panels: { detect:{x:28,y:106,open:false}, metrics:{x:28,y:144,open:false} },
      docks: { detect:{x:28,y:106}, metrics:{x:28,y:154} },
      railOpen: { saved:false, layers:false, export:false },
      dragging: null,
      project: 'colom_mobility',
      // calibration
      calibrationPairs: [],
      pendingCalibPoint: null,
      selectedCalibPair: null,
      cpActive: [],
      hoverCP: null,
      // encoding
      encStep: 7,
      encPlaying: false,
      // processing
      procDone: 6,
      calibration: this.buildCalibrationState([]),
      encoding: this.buildEncodingState(7),
      processing: this.buildProcessingState(6),
      dataset: this.buildPlacaDataset(resetDemoState ? { selected_video_filename: '', selected_plan_filename: '' } : {}),
      realData: { loaded:false, tracks:[], detections:[], flow:[], bottlenecks:[], metrics:null, calibration:null },
      calibDrag: null,
    };
  }

  componentDidMount(){
    window.__motionPixelsApp = this;
    this._move = (e) => {
      if(this.state.calibDrag) { this.moveCalibrationPoint(e); return; }
      const d = this.state.dragging; if(!d) return;
      const nx = e.clientX - d.dx, ny = e.clientY - d.dy;
      if(d.kind === 'dock' && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) > 4) this._dockClickOk = false;
      if(d.kind === 'dock'){
        this.setState(s => ({ docks: { ...s.docks, [d.key]: { x: Math.max(8,nx), y: Math.max(72,ny) } } }));
        return;
      }
      this.setState(s => ({ panels: { ...s.panels, [d.key]: { ...s.panels[d.key], x: Math.max(8,nx), y: Math.max(72,ny) } } }));
    };
    this._up = () => { if(this.state.dragging) this.setState({ dragging:null }); };
    this._calibUp = () => { if(this.state.calibDrag) this.setState({ calibDrag:null }); };
    window.addEventListener('pointermove', this._move);
    window.addEventListener('pointerup', this._up);
    window.addEventListener('pointerup', this._calibUp);
    this.loadRealDatasetData();
    this.applyReviewRoute();
  }
  componentWillUnmount(){
    window.removeEventListener('pointermove', this._move);
    window.removeEventListener('pointerup', this._up);
    window.removeEventListener('pointerup', this._calibUp);
    cancelAnimationFrame(this._raf);
  }

  shouldResetDemoState(){
    try {
      return new URLSearchParams(window.location.search).get('reset') === '1';
    } catch(err) {
      return false;
    }
  }

  clearDemoStorage(){
    try { window.localStorage && window.localStorage.clear(); } catch(err) {}
    try { window.sessionStorage && window.sessionStorage.clear(); } catch(err) {}
  }

  applyReviewRoute(){
    const q=new URLSearchParams(window.location.search);
    const hash=window.location.hash.replace('#','');
    if(!q.has('screen') && !q.has('layers') && hash!=='workspace') return;
    const next={};
    const screen=q.get('screen');
    if(['home','upload','calibrate','encode','process','workspace'].includes(screen)) next.screen=screen;
    if(hash==='workspace') next.screen='workspace';
    if(q.get('layers')==='all') next.layers={speed:true,flow:true,bottlenecks:true,predictions:false};
    if(hash==='workspace') next.layers={speed:true,flow:true,bottlenecks:true,predictions:false};
    if(q.has('progress')) next.progress=Math.max(0,Math.min(1,Number(q.get('progress')) || 0));
    this.setState(next);
  }

  buildPlacaDataset(overrides = {}){
    return {
      dataset_id: 'placa_espanya',
      selected_video_filename: '',
      selected_plan_filename: '',
      demo_video_path: 'demo_assets/placa_espanya/video/tracked.mp4',
      tracked_video_path: 'demo_assets/placa_espanya/video/tracked.mp4',
      demo_plan_path: 'demo_assets/placa_espanya/plan/placa-espanya.png',
      calibration_path: 'demo_assets/placa_espanya/calibration/calib.json',
      calibration_preview_path: 'demo_assets/placa_espanya/calibration/calib_preview.png',
      calibration_plot_path: 'demo_assets/placa_espanya/calibration/calib_plot.png',
      trajectory_paths: [
        'demo_assets/placa_espanya/tracking/trajectories_world_sample.csv',
        'demo_assets/placa_espanya/tracking/trajectories_image_space.csv',
        'demo_assets/placa_espanya/tracking/tracks_summary.csv',
        'demo_assets/placa_espanya/tracking/speed_per_track.csv',
      ],
      behavior_paths: [
        'demo_assets/placa_espanya/behavior_maps_final/bottleneck_density_still.png',
        'demo_assets/placa_espanya/behavior_maps_final/flow_fields_still.png',
        'demo_assets/placa_espanya/behavior_maps_final/speed_population_still.png',
        'demo_assets/placa_espanya/behavior_maps/flow_field_vectors_sample.csv',
        'demo_assets/placa_espanya/behavior_maps/bottleneck_cells.csv',
      ],
      prediction_paths: [],
      prediction_visual_path: 'demo_assets/placa_espanya/prediction_maps/prediction_anim_placa_espanya_01_plain.mp4',
      metrics_paths: [
        'demo_assets/placa_espanya/tracking/tracks_summary.csv',
      ],
      sharedSpatialCoordinates: {
        id: 'placa-espanya-01-plan-world',
        viewBox: [0, 0, 800, 520],
        nativePlanSize: [2311, 1227],
        imageFit: { scale: 800 / 2311, offsetX: 0, offsetY: (520 - (1227 * (800 / 2311))) / 2 },
        origin: [1569.119, 1021.214],
        metersPerPixel: 0.09972699226400716,
        zScale: 0.72,
        projection: { xSkew: -0.42, ySkew: -0.58 },
      },
      ...overrides,
    };
  }

  parseCSV(text){
    const lines = text.trim().split(/\r?\n/);
    const headers = lines.shift().split(',');
    return lines.map(line => {
      const vals = line.split(',');
      const row = {};
      headers.forEach((h,i) => { const n = Number(vals[i]); row[h] = Number.isFinite(n) && vals[i] !== '' ? n : vals[i]; });
      return row;
    });
  }

  planPixelToFrame(x, y){
    const fit = this.state?.dataset?.sharedSpatialCoordinates?.imageFit || {scale:800/2311,offsetX:0,offsetY:47.7};
    return [x * fit.scale + fit.offsetX, y * fit.scale + fit.offsetY];
  }

  realWorldToFrame(x, y){
    const f = this.state?.dataset?.sharedSpatialCoordinates || this.buildPlacaDataset().sharedSpatialCoordinates;
    return this.planPixelToFrame(f.origin[0] + x / f.metersPerPixel, f.origin[1] + y / f.metersPerPixel);
  }

  get calibVideoRect(){ return { x:20, y:18, w:280, h:484 }; }
  calibImageToFrame(x, y){
    const r=this.calibVideoRect;
    return [r.x + x / 1080 * r.w, r.y + y / 1920 * r.h];
  }
  frameToCalibImage(x, y){
    const r=this.calibVideoRect;
    return [Math.max(0,Math.min(1080,(x - r.x) / r.w * 1080)), Math.max(0,Math.min(1920,(y - r.y) / r.h * 1920))];
  }
  frameToPlanPixel(x, y){
    const fit = this.state?.dataset?.sharedSpatialCoordinates?.imageFit || {scale:800/2311,offsetX:0,offsetY:47.7};
    return [Math.max(0,Math.min(2311,(x - fit.offsetX) / fit.scale)), Math.max(0,Math.min(1227,(y - fit.offsetY) / fit.scale))];
  }

  getSvgPoint(e){
    const svg=e.currentTarget.ownerSVGElement || e.currentTarget;
    const pt=svg.createSVGPoint();
    pt.x=e.clientX; pt.y=e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  startCalibrationDrag = (e, n, space) => {
    e.stopPropagation();
    e.preventDefault();
    this._calibSvg = e.currentTarget.ownerSVGElement;
    this.setState({ calibDrag:{ n, space } });
  };

  moveCalibrationPoint(e){
    const drag=this.state.calibDrag, svg=this._calibSvg;
    if(!drag || !svg) return;
    const pt=svg.createSVGPoint();
    pt.x=e.clientX; pt.y=e.clientY;
    const p=pt.matrixTransform(svg.getScreenCTM().inverse());
    this.setState(s=>{
      const pairs=s.calibrationPairs.map(pair=>{
        if(pair.n!==drag.n) return pair;
        return drag.space==='image'
          ? {...pair, video:[Math.max(0,Math.min(800,p.x)), Math.max(0,Math.min(520,p.y))]}
          : {...pair, plan:[Math.max(0,Math.min(800,p.x)), Math.max(0,Math.min(520,p.y))]};
      });
      return this.withCalibrationStats({...s, calibrationPairs:pairs, selectedCalibPair:drag.n});
    });
  }

  calibrationVideoClick = (e) => {
    const p=this.getSvgPoint(e);
    const point=[Math.max(0,Math.min(800,p.x)), Math.max(0,Math.min(520,p.y))];
    this.setState(s=>{
      if(s.pendingCalibPoint?.space==='plan'){
        const pairs=[...s.calibrationPairs,{n:s.calibrationPairs.length+1,video:point,plan:s.pendingCalibPoint.point,c:this.cpColor(s.calibrationPairs.length)}];
        return this.withCalibrationStats({...s, calibrationPairs:pairs, pendingCalibPoint:null, selectedCalibPair:pairs.length});
      }
      return { pendingCalibPoint:{space:'video',point}, selectedCalibPair:null };
    });
  };

  calibrationPlanClick = (e) => {
    const p=this.getSvgPoint(e);
    const point=[Math.max(0,Math.min(800,p.x)), Math.max(0,Math.min(520,p.y))];
    this.setState(s=>{
      if(s.pendingCalibPoint?.space==='video'){
        const pairs=[...s.calibrationPairs,{n:s.calibrationPairs.length+1,video:s.pendingCalibPoint.point,plan:point,c:this.cpColor(s.calibrationPairs.length)}];
        return this.withCalibrationStats({...s, calibrationPairs:pairs, pendingCalibPoint:null, selectedCalibPair:pairs.length});
      }
      return { pendingCalibPoint:{space:'plan',point}, selectedCalibPair:null };
    });
  };

  undoCalibrationPair = () => this.setState(s => {
    const pairs=s.calibrationPairs.slice(0,-1).map((p,i)=>({...p,n:i+1,c:this.cpColor(i)}));
    return this.withCalibrationStats({...s, calibrationPairs:pairs, pendingCalibPoint:null, selectedCalibPair:pairs.length || null});
  });

  clearCalibrationPairs = () => this.setState(s => this.withCalibrationStats({...s, calibrationPairs:[], pendingCalibPoint:null, selectedCalibPair:null}));

  deleteCalibrationPair = (n) => this.setState(s => {
    const pairs=s.calibrationPairs.filter(p=>p.n!==n).map((p,i)=>({...p,n:i+1,c:this.cpColor(i)}));
    return this.withCalibrationStats({...s, calibrationPairs:pairs, selectedCalibPair:null});
  });

  cpColor(i){
    return ['#5ec9c1','#d97a74','#c98d4a','#9d88c4','#7ab894','#7aacc8','#d4a14a','#6282c8'][i%8];
  }

  solveLinear(A,b){
    const n=b.length;
    const M=A.map((row,i)=>[...row,b[i]]);
    for(let col=0; col<n; col++){
      let pivot=col;
      for(let r=col+1;r<n;r++) if(Math.abs(M[r][col])>Math.abs(M[pivot][col])) pivot=r;
      if(Math.abs(M[pivot][col])<1e-9) return null;
      [M[col],M[pivot]]=[M[pivot],M[col]];
      const div=M[col][col];
      for(let c=col;c<=n;c++) M[col][c]/=div;
      for(let r=0;r<n;r++){
        if(r===col) continue;
        const f=M[r][col];
        for(let c=col;c<=n;c++) M[r][c]-=f*M[col][c];
      }
    }
    return M.map(row=>row[n]);
  }

  computeHomography(pairs){
    if(pairs.length<4) return null;
    const A=[], b=[];
    pairs.slice(0,4).forEach(pair=>{
      const [x,y]=pair.video, [u,v]=pair.plan;
      A.push([x,y,1,0,0,0,-u*x,-u*y]); b.push(u);
      A.push([0,0,0,x,y,1,-v*x,-v*y]); b.push(v);
    });
    const h=this.solveLinear(A,b);
    return h ? [...h,1] : null;
  }

  projectHomography(H, point){
    const [x,y]=point;
    const w=H[6]*x + H[7]*y + H[8];
    if(Math.abs(w)<1e-9) return null;
    return [(H[0]*x + H[1]*y + H[2])/w, (H[3]*x + H[4]*y + H[5])/w];
  }

  withCalibrationStats(s){
    const pairs=s.calibrationPairs || [];
    const H=this.computeHomography(pairs);
    let rmse=null;
    if(H){
      const sum=pairs.reduce((acc,pair)=>{
        const q=this.projectHomography(H,pair.video);
        if(!q) return acc + 999;
        const dx=q[0]-pair.plan[0], dy=q[1]-pair.plan[1];
        return acc + dx*dx + dy*dy;
      },0);
      const displayPx=Math.sqrt(sum/Math.max(1,pairs.length));
      const fit=s.dataset?.sharedSpatialCoordinates?.imageFit || this.buildPlacaDataset().sharedSpatialCoordinates.imageFit;
      const metersPerDisplayPx=(s.dataset?.sharedSpatialCoordinates?.metersPerPixel || 0.0997) / fit.scale;
      rmse=Number((displayPx*metersPerDisplayPx).toFixed(2));
    }
    const valid=!!H && pairs.length>=4 && Number.isFinite(rmse);
    return {
      calibrationPairs:pairs,
      cpActive:pairs.map(p=>p.n),
      pendingCalibPoint:s.pendingCalibPoint,
      selectedCalibPair:s.selectedCalibPair,
      calibration:{
        calibrationComplete: valid,
        matchedPoints: pairs.length,
        rmse: valid ? rmse : null,
        homographyStatus: valid ? 'Valid' : pairs.length ? 'Needs 4 point pairs' : 'Awaiting points',
        homography:H,
      }
    };
  }

  async loadRealDatasetData(){
    const d = this.state.dataset || this.buildPlacaDataset();
    try {
      const [calib, trajText, flowText, bottText, speedText, summaryText] = await Promise.all([
        fetch(d.calibration_path).then(r=>r.json()),
        fetch(d.trajectory_paths[0]).then(r=>r.text()),
        fetch('demo_assets/placa_espanya/behavior_maps/flow_field_vectors_sample.csv').then(r=>r.text()),
        fetch('demo_assets/placa_espanya/behavior_maps/bottleneck_cells.csv').then(r=>r.text()),
        fetch('demo_assets/placa_espanya/tracking/speed_per_track.csv').then(r=>r.text()),
        fetch('demo_assets/placa_espanya/tracking/tracks_summary.csv').then(r=>r.text()),
      ]);
      const traj = this.parseCSV(trajText);
      const byTrack = new Map();
      traj.forEach(row => {
        if(!byTrack.has(row.track_id)) byTrack.set(row.track_id, []);
        const arr = byTrack.get(row.track_id);
        if(arr.length < 70 && row.frame % 12 === 0) arr.push(this.realWorldToFrame(row.world_x, row.world_y));
      });
      const tracks = [...byTrack.values()].filter(t=>t.length>2).slice(0,18);
      const detections = traj.filter(r=>r.frame===0).slice(0,80).map(r=>this.realWorldToFrame(r.world_x, r.world_y));
      const flow = this.parseCSV(flowText).slice(0,80);
      const bottlenecks = this.parseCSV(bottText).slice(0,14).map(r=>({ p:this.realWorldToFrame(r.cell_x, r.cell_y), score:r.bottleneck_score || r.density_score || 0.5 }));
      const speedRows = this.parseCSV(speedText);
      const summaryRows = this.parseCSV(summaryText);
      const avgSpeed = speedRows.length ? speedRows.reduce((a,r)=>a+(Number(r.mean_speed)||0),0)/speedRows.length : 0;
      const metrics = {
        trackCount: summaryRows.length,
        avgSpeed,
        bottleneckCount: bottlenecks.length,
        flowCount: this.parseCSV(flowText).length,
      };
      this.setState({
        realData: { loaded:true, calibration:calib, tracks, detections, flow, bottlenecks, metrics },
      });
    } catch (err) {
      console.warn('Real Placa Espanya dataset could not be loaded', err);
    }
  }


  buildCalibrationState(cpActive = this.state?.cpActive || []){
    const count = cpActive.length;
    const matchedPoints = count;
    const rmse = count ? Math.max(0.18, 0.74 - count * 0.055) : 0;
    const valid = count >= 4;
    return {
      calibrationComplete: valid,
      matchedPoints,
      rmse: valid ? Number(rmse.toFixed(2)) : null,
      homographyStatus: valid ? 'Valid' : 'Needs more points',
    };
  }

  buildEncodingState(step = this.state?.encStep || 0){
    return {
      planVisible: step >= 1,
      boundaryMask: step >= 2,
      walkableSurface: step >= 3,
      obstacleMask: step >= 4,
      entranceMask: step >= 5,
      distanceField: step >= 6,
      contextGrid: step >= 7,
      encodingComplete: step >= 7,
    };
  }

  buildProcessingState(done = this.state?.procDone || 0){
    return {
      pedestriansTracked: done >= 1 ? 128 : 0,
      trajectoryCount: done >= 2 ? 312 : 0,
      averageSpeed: done >= 5 ? 1.2 : done >= 2 ? 0.8 : 0,
      behaviorSamples: done >= 5 ? 1840 : done >= 4 ? 920 : 0,
      predictionHorizons: [],
      behaviorDataset: done >= 5 ? 'Generated' : 'Pending',
      predictionModel: 'Pending',
    };
  }

  routeToPlacaDataset(patch){
    this.setState(s => ({
      dataset: this.buildPlacaDataset({ ...s.dataset, ...patch }),
    }));
  }

  triggerVideoPicker = () => this.videoInputRef.current?.click();
  triggerPlanPicker = () => this.planInputRef.current?.click();

  handleVideoSelected = (event) => {
    const file = event.target.files && event.target.files[0];
    if(!file) return;
    this.setState(s => ({
      uVideo: true,
      dataset: this.buildPlacaDataset({
        ...s.dataset,
        dataset_id: 'placa_espanya',
        selected_video_filename: file.name,
      }),
    }));
  };

  handlePlanSelected = (event) => {
    const file = event.target.files && event.target.files[0];
    if(!file) return;
    this.setState(s => ({
      uPlan: true,
      dataset: this.buildPlacaDataset({
        ...s.dataset,
        dataset_id: 'placa_espanya',
        selected_plan_filename: file.name,
      }),
    }));
  };

  // â”€â”€ geometry helpers â”€â”€
  ptC(r, deg){ const a = deg*Math.PI/180; return [this.sharedFrame.origin.x + r*Math.cos(a), this.sharedFrame.origin.y + r*Math.sin(a)]; }
  planToWorld(x, y, z=0){
    const f = this.sharedFrame;
    return { x:(x - f.origin.x) * f.metersPerPixel, y:(f.origin.y - y) * f.metersPerPixel, z };
  }
  worldToPlan(x, y, z=0){
    const f = this.sharedFrame;
    const px = f.origin.x + (x / f.metersPerPixel) + z * f.projection.xSkew;
    const py = f.origin.y - (y / f.metersPerPixel) + z * f.projection.ySkew;
    return [px, py];
  }
  isoPoint(x, y, z=0){
    const w = this.planToWorld(x, y, z);
    return this.worldToPlan(w.x, w.y, z * this.sharedFrame.zScale);
  }
  smooth(pts){
    if(pts.length < 2) return '';
    let d = `M ${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for(let i=0;i<pts.length-1;i++){
      const p0=pts[i-1]||pts[i], p1=pts[i], p2=pts[i+1], p3=pts[i+2]||p2;
      const c1x=p1[0]+(p2[0]-p0[0])/6, c1y=p1[1]+(p2[1]-p0[1])/6;
      const c2x=p2[0]-(p3[0]-p1[0])/6, c2y=p2[1]-(p3[1]-p1[1])/6;
      d += ` C ${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  }
  arcPts(a0, a1, r){
    const pts = [];
    pts.push(this.ptC(262, a0));
    pts.push(this.ptC(r+26, a0));
    const dir = a1 > a0 ? 1 : -1;
    for(let a=a0; dir>0 ? a<a1 : a>a1; a += dir*12) pts.push(this.ptC(r, a));
    pts.push(this.ptC(r, a1));
    pts.push(this.ptC(r+26, a1));
    pts.push(this.ptC(262, a1));
    return pts;
  }

  get observedSpecs(){
    return [
      [-160,-20,190,'#5ec9c1',1.5,.50],
      [200,40,172,'#d97a74',1.3,.46],
      [-100,80,184,'#c98d4a',1.4,.44],
      [40,170,162,'#9d88c4',1.2,.40],
      [120,258,198,'#7ab894',1.1,.36],
      [300,172,178,'#7aacc8',1.1,.34],
      [-44,-150,168,'#5ec9c1',1.0,.34],
      [80,-26,158,'#d97a74',1.0,.34],
      [222,318,192,'#9d88c4',1.0,.30],
      [12,118,150,'#7ab894',0.9,.30],
      [158,40,176,'#7aacc8',1.2,.40],
      [-70,46,200,'#c98d4a',1.0,.30],
    ];
  }


  buildSharedFrameGrid(){
    const k=[];
    for(let x=0;x<=800;x+=40) k.push(h('line',{key:'fgx'+x,x1:x,y1:0,x2:x,y2:520,stroke:'#eeeeeb',strokeWidth:.45,opacity:.45}));
    for(let y=0;y<=520;y+=40) k.push(h('line',{key:'fgy'+y,x1:0,y1:y,x2:800,y2:y,stroke:'#eeeeeb',strokeWidth:.45,opacity:.45}));
    k.push(h('line',{key:'origin-x',x1:0,y1:this.cy,x2:800,y2:this.cy,stroke:'#d9dad6',strokeWidth:.55,strokeDasharray:'4 8',opacity:.6}));
    k.push(h('line',{key:'origin-y',x1:this.cx,y1:0,x2:this.cx,y2:520,stroke:'#d9dad6',strokeWidth:.55,strokeDasharray:'4 8',opacity:.6}));
    return h('g',{key:'shared-grid'},k);
  }

  buildSharedFrameBadge(){
    return h('g',{key:'shared-badge',transform:'translate(355,6)'},
      h('rect',{x:0,y:0,width:90,height:28,rx:2,fill:'rgba(255,255,255,.90)',stroke:'#e1e1dc',strokeWidth:.8}),
      h('text',{x:45,y:18,textAnchor:'middle',fontFamily:"'Space Grotesk'",fontWeight:700,fontSize:8.5,letterSpacing:'.18em',fill:'#3a3d40'},'STUDIO')
    );
  }

  // PLAN: Placa Espanya
  buildPlan(){
    const planPath = this.state?.dataset?.demo_plan_path;
    return h('g',{},
      h('rect',{key:'plan-bg',width:800,height:520,fill:'#fbfcfd'}),
      planPath ? h('image',{key:'real-plan',href:planPath,x:0,y:0,width:800,height:520,preserveAspectRatio:'xMidYMid meet',opacity:.58,style:{filter:'saturate(.82) brightness(1.05)'}}) : null,
      // cool high-key wash keeps the plan clean and architectural rather than muddy
      h('rect',{key:'plan-wash',width:800,height:520,fill:'#f4f7fa',opacity:.34,style:{mixBlendMode:'screen'}})
    );
  }

  buildBehaviorAssetOverlay(key, opacity=.62, animated=false){
    const stills = {
      speed: 'demo_assets/placa_espanya/behavior_maps_final/speed_population_still.png',
      flow: 'demo_assets/placa_espanya/behavior_maps_final/flow_fields_still.png',
      bottlenecks: 'demo_assets/placa_espanya/behavior_maps_final/bottleneck_density_still.png',
    };
    const videos = {
      speed: 'demo_assets/placa_espanya/behavior_maps_final/speed_population.mp4',
      flow: 'demo_assets/placa_espanya/behavior_maps_final/flow_fields.mp4',
      bottlenecks: 'demo_assets/placa_espanya/behavior_maps_final/bottleneck_density.mp4',
      predictions: this.state?.dataset?.prediction_visual_path || 'demo_assets/placa_espanya/prediction_maps/prediction_anim_placa_espanya_01_plain.mp4',
    };
    // The source maps are glow-on-black renders. invert(1) hue-rotate(180deg) flips the
    // black ground to white (which vanishes under multiply) while preserving each data hue,
    // so every map composites as ONE clean layer over the light plan -- no stacked-footage
    // artifacts, no muddy darkening, and nothing remains when the layer is toggled off.
    const norm = {
      speed:       'invert(1) hue-rotate(180deg) saturate(1.25) contrast(1.05)',
      flow:        'invert(1) hue-rotate(180deg) saturate(1.32) contrast(1.06)',
      bottlenecks: 'invert(1) hue-rotate(180deg) saturate(1.4) contrast(1.1)',
      predictions: 'invert(1) hue-rotate(180deg) saturate(1.55) contrast(1.18) brightness(.96)',
    };
    if(animated && videos[key]){
      const frame = key === 'predictions' ? { x:0, y:47.62, width:800, height:424.75 } : { x:0, y:0, width:800, height:520 };
      const overlay = h('foreignObject',{key:'behavior-video-'+key,x:frame.x,y:frame.y,width:frame.width,height:frame.height,style:{pointerEvents:'none',overflow:'hidden',opacity, mixBlendMode:'multiply'}},
        h('video',{className:'behavior-map-video',src:videos[key],muted:true,loop:true,playsInline:true,preload:'metadata','data-layer-key':key,onLoadedMetadata:this.startBehaviorLayerVideo,style:{width:frame.width+'px',height:frame.height+'px',objectFit:'cover',display:'block',filter:norm[key]||'none'}})
      );
      return overlay;
    }
    return h('image',{key:'behavior-'+key,href:stills[key],x:0,y:0,width:800,height:520,preserveAspectRatio:'xMidYMid slice',opacity,style:{filter:norm[key]||'none',mixBlendMode:'multiply'}});
  }

  buildMapLegend(){
    const legendGradients = {
      Speed: [['0%','#f28b2e',.9], ['100%','#19c7f3',.95]],
      'Flow Fields': [['0%','#8edcff',.92], ['50%','#153f7a',.94], ['100%','#ff4fb8',.92]],
      Bottlenecks: [['0%','#ffe04a',.9], ['100%','#f28b2e',.95]],
      Predictions: [['0%','#ffffff',.95], ['100%','#d100d8',.95]],
    };
    const active=[
      this.state.layers.speed ? ['Speed','#5ec9c1','low speed','high speed'] : null,
      this.state.layers.flow ? ['Flow Fields','#153f7a','low flow','high flow'] : null,
      this.state.layers.bottlenecks ? ['Bottlenecks','#d4724a','clear','dense'] : null,
      this.state.layers.predictions ? ['Predictions','#d100d8','ground truth','predictions'] : null,
    ].filter(Boolean);
    if(!active.length) return null;
    const rows=active.map((item,i)=>{
      const [label,color,a,b]=item;
      return h('g',{key:label,transform:`translate(${20+i*194},0)`},
        h('circle',{cx:0,cy:8,r:3,fill:color,opacity:.85}),
        h('text',{x:10,y:11,fontFamily:"'Space Grotesk'",fontWeight:600,fontSize:7.5,letterSpacing:'.08em',fill:'#6b6e72'},label),
        h('text',{x:0,y:24,fontFamily:"'Space Grotesk'",fontWeight:400,fontSize:6.5,letterSpacing:'.06em',fill:'#a6a6a2'},a),
        h('rect',{x:52,y:18,width:90,height:3,fill:`url(#legend-${label.replace(/\s/g,'-')})`,rx:1.5}),
        h('text',{x:148,y:24,fontFamily:"'Space Grotesk'",fontWeight:400,fontSize:6.5,letterSpacing:'.06em',fill:'#a6a6a2'},b)
      );
    });
    const defs=active.map(item=>h('linearGradient',{key:item[0],id:'legend-'+item[0].replace(/\s/g,'-'),x1:'0%',x2:'100%'},
      ...(legendGradients[item[0]] || [['0%',item[1],.12], ['100%',item[1],.92]]).map(stop =>
        h('stop',{key:stop[0],offset:stop[0],stopColor:stop[1],stopOpacity:stop[2]})
      )
    ));
    return h('g',{key:'map-legend',transform:'translate(0,490)'},
      h('defs',{},defs),
      h('rect',{x:0,y:-8,width:800,height:38,fill:'#fcfcfa',stroke:'#e7e7e3',strokeWidth:.7}),
      rows
    );
  }

  buildBehaviorAccents(){
    const L=this.state.layers, rd=this.state.realData || {};
    const items=[];
    if(L.speed && rd.tracks?.length){
      rd.tracks.slice(0,20).forEach((pts,i)=>{
        pts.filter((_,idx)=>idx%10===0).slice(0,8).forEach(([x,y],j)=>{
          items.push(h('circle',{key:'spd'+i+'-'+j,cx:x,cy:y,r:2.3,fill:'#5ec9c1',opacity:.58,filter:'url(#mp-glow)'}));
        });
      });
    }
    if(L.flow && rd.flow?.length){
      rd.flow.slice(0,85).forEach((r,i)=>{
        const [x,y]=this.realWorldToFrame(r.cell_x, r.cell_y);
        const x2=x+(r.ux||0)*13, y2=y+(r.uy||0)*13;
        const ang=Math.atan2(y2-y,x2-x), ah=4.5;
        items.push(h('g',{key:'flow'+i,opacity:.68},
          h('line',{x1:x,y1:y,x2:x2,y2:y2,stroke:'#7ab894',strokeWidth:1.15,strokeLinecap:'round'}),
          h('path',{d:`M ${x2},${y2} l ${Math.cos(ang+2.55)*ah},${Math.sin(ang+2.55)*ah} M ${x2},${y2} l ${Math.cos(ang-2.55)*ah},${Math.sin(ang-2.55)*ah}`,stroke:'#7ab894',strokeWidth:1,strokeLinecap:'round'})
        ));
      });
    }
    if(L.bottlenecks && rd.bottlenecks?.length){
      rd.bottlenecks.slice(0,18).forEach((b,i)=>{
        const r=5+Math.min(13,(b.score||.5)*5);
        items.push(h('g',{key:'bot'+i},
          h('ellipse',{cx:b.p[0],cy:b.p[1],rx:r*1.55,ry:r*.75,fill:'#d4724a',opacity:.20,filter:'url(#mp-soft)'}),
          h('ellipse',{cx:b.p[0],cy:b.p[1],rx:r*.56,ry:r*.28,fill:'#d4724a',opacity:.58})
        ));
      });
    }
    return items.length ? h('g',{key:'behavior-accents'},items) : null;
  }

  // behavioral drawing scene â”€â”€
  buildScene(){
    const s = this.state, L = s.layers, prog = s.progress;
    const defs = h('defs',{key:'defs'},
      h('filter',{id:'mp-soft',x:'-50%',y:'-50%',width:'200%',height:'200%'}, h('feGaussianBlur',{stdDeviation:5})),
      h('filter',{id:'mp-glow',x:'-20%',y:'-20%',width:'140%',height:'140%'}, h('feGaussianBlur',{stdDeviation:1.2,result:'b'}), h('feMerge',{}, h('feMergeNode',{in:'b'}), h('feMergeNode',{in:'SourceGraphic'})))
    );
    const groups = [h('rect',{key:'wbg',width:800,height:520,fill:'#ffffff'}), h('g',{key:'plan'}, this.buildPlan())];
    if(L.speed) groups.push(this.buildBehaviorAssetOverlay('speed', .78, true));
    if(L.flow) groups.push(this.buildBehaviorAssetOverlay('flow', .82, true));
    if(L.bottlenecks) groups.push(this.buildBehaviorAssetOverlay('bottlenecks', .82, true));
    if(L.predictions) groups.push(this.buildBehaviorAssetOverlay('predictions', .92, true));
    // (raw-generation accents removed in polish pass 13 — only final cleaned map overlays remain)
    groups.push(this.buildSharedFrameBadge());
    groups.push(this.buildMapLegend());
    return h('svg',{viewBox:'0 0 800 520',width:'100%',height:'100%',preserveAspectRatio:'xMidYMid meet',style:{display:'block'}}, defs, groups);
  }

  // tracked video mini â”€â”€
  buildDetectVideo(){
    const video = this.state.dataset?.tracked_video_path || this.state.dataset?.demo_video_path;
    if(video) return h('video',{className:'tracking-video',src:video,muted:true,autoPlay:true,loop:true,playsInline:true,onLoadedMetadata:this.startMediaAtBeginning,style:this.portraitVideoStyle(154,214)});
    const peds=[[26,52,14,30,'#5ec9c1',.93],[51,57,12,28,'#d97a74',.88],[70,50,11,26,'#c98d4a',.79],[12,61,9,22,'#7ab894',.84],[84,55,9,20,'#9d88c4',.71]];
    const k=[h('rect',{key:'bg',width:100,height:100,fill:'#eceae6'}),h('rect',{key:'gr',y:52,width:100,height:48,fill:'#e6e4df'})];
    [0,12,24,38,50,62,76,88,100].forEach((x,i)=>k.push(h('line',{key:'g'+i,x1:x,y1:100,x2:50,y2:50,stroke:'#dedcd7',strokeWidth:.3})));
    k.push(h('rect',{key:'f1',x:0,y:0,width:20,height:52,fill:'#d8d6d1'}),h('rect',{key:'f2',x:80,y:0,width:20,height:52,fill:'#d8d6d1'}),h('rect',{key:'f3',x:20,y:6,width:60,height:46,fill:'#e0ded8'}));
    k.push(h('rect',{key:'arch',x:46,y:30,width:8,height:22,rx:1,fill:'#cdcbc5'}));
    peds.forEach((p,i)=>{
      const [x,y,w,ht,c]=p;
      k.push(h('g',{key:'s'+i,opacity:.5}, h('rect',{x:x+w/2-2.4,y:y+5,width:4.8,height:ht-8,rx:.8,fill:'#9c9a95'}), h('circle',{cx:x+w/2,cy:y+3.4,r:2.4,fill:'#9c9a95'})));
      k.push(h('rect',{key:'b'+i,x,y,width:w,height:ht,fill:'none',stroke:c,strokeWidth:.9,opacity:.92}));
      k.push(h('rect',{key:'t'+i,x,y:y-4.4,width:10,height:4.4,fill:c,opacity:.9}));
      k.push(h('text',{key:'id'+i,x:x+0.8,y:y-1.2,fontSize:2.8,fontFamily:'monospace',fontWeight:700,fill:'#fff'},'P0'+(i+1)));
    });
    k.push(h('rect',{key:'yb',x:1.5,y:1.5,width:20,height:5,fill:'rgba(0,0,0,.55)',rx:.5}),h('text',{key:'yt',x:2.5,y:5.3,fontSize:3,fontFamily:'monospace',fontWeight:700,fill:'#fff'},'YOLOv8'));
    return h('svg',{viewBox:'0 0 100 100',width:'100%',preserveAspectRatio:'xMidYMid slice',style:{display:'block',height:118}}, k);
  }

  // â”€â”€ persistent pipeline spine â”€â”€
  buildSpine(){
    const stages=[
      {id:'detect',n:'01',label:'Detection',sub:'YOLOv8',go:'process'},
      {id:'track',n:'02',label:'Tracking',sub:'ByteTrack',go:'process'},
      {id:'calibrate',n:'03',label:'Calibration',sub:'Homography',go:'calibrate'},
      {id:'encode',n:'04',label:'Spatial Encoding',sub:'Context',go:'encode'},
      {id:'metrics',n:'05',label:'Metrics',sub:'Speed + Flow',go:'process'},
      {id:'dataset',n:'06',label:'Dataset',sub:'Trajectories',go:'process'},
      {id:'predict',n:'07',label:'Prediction',sub:'Next pass',go:'workspace'},
      {id:'horizons',n:'08',label:'Studio',sub:'',go:'workspace'},
    ];
    const cur = { upload:1, calibrate:2, encode:3, process:5, workspace:7 }[this.state.screen];
    const curIdx = cur===undefined ? -1 : cur;
    const nodes = stages.map((st,i)=>{
      const done = i < curIdx, active = i === curIdx;
      const col = active ? '#16181a' : done ? '#7ab894' : '#cfcfca';
      const txt = active ? '#16181a' : done ? '#3a3d40' : '#b6b6b2';
      return h('div',{key:st.id, onClick:()=>this.setState({screen:st.go}), style:{display:'flex',alignItems:'center',gap:8,cursor:'pointer',flex:'0 0 auto'}},
        h('div',{style:{display:'flex',alignItems:'center',gap:7}},
          h('span',{style:{width:7,height:7,borderRadius:'50%',background:active?'#16181a':done?'#7ab894':'transparent',border:active||done?'none':'1.4px solid #d4d4d0',flexShrink:0,animation:active?'mp-pulse 1.6s ease-in-out infinite':'none'}}),
          h('span',{style:{fontFamily:"'Space Grotesk'",fontWeight:500,fontSize:8,color:txt}},st.n),
          h('span',{style:{fontFamily:"'Space Grotesk'",fontWeight:active?600:500,fontSize:10.5,color:txt,letterSpacing:'.01em'}},st.label),
          h('span',{style:{fontFamily:"'Space Grotesk'",fontWeight:300,fontSize:8.5,color:'#bcbcb8',letterSpacing:'.02em'}},st.sub)
        ),
        i<stages.length-1 ? h('span',{style:{width:18,height:1,background:done?'#cfe0d2':'#ececea'}}) : null
      );
    });
    return h('div',{style:{height:42,flexShrink:0,borderBottom:'1px solid #ececea',background:'#fcfcfb',display:'flex',alignItems:'center',gap:12,padding:'0 22px',overflowX:'auto'}}, nodes);
  }

  // â”€â”€ workspace small controls â”€â”€
  buildAesthetic(){
    const ae=this.state.aesthetic;
    const opts=[['ink','Ink'],['flow','Flow'],['field','Field']];
    const btn=([val,label])=>h('button',{key:val,onClick:()=>this.setState({aesthetic:val}),style:{fontFamily:"'Space Grotesk'",fontWeight:ae===val?600:400,fontSize:9.5,letterSpacing:'.06em',color:ae===val?'#16181a':'#9b9b97',background:ae===val?'#f1f1ee':'transparent',border:'none',padding:'6px 12px',cursor:'pointer',transition:'all .15s'}},label);
    return h('div',{style:{display:'flex',background:'rgba(255,255,255,.94)',backdropFilter:'blur(6px)',border:'1px solid #e2e2de',borderRadius:3,overflow:'hidden'}}, opts.map(btn));
  }
  buildHorizonRow(){
    const hz=this.state.horizon;
    const opts=[40,60,100,200,400];
    return h('div',{style:{display:'flex',gap:4}}, opts.map(v=>h('button',{key:v,onClick:()=>this.setState({horizon:v,progress:1}),style:{fontFamily:"'Space Grotesk'",fontWeight:hz===v?700:500,fontSize:10,letterSpacing:'.02em',color:hz===v?'#fff':'#6b6e72',background:hz===v?'#16181a':'#fff',border:'1px solid '+(hz===v?'#16181a':'#dcdcd8'),padding:'5px 9px',cursor:'pointer',transition:'all .15s'}},'H'+v)));
  }
  buildSpeedRow(){
    const active=this.state.playbackSpeed;
    const opts=[1,1.5,2,3];
    return h('div',{className:'speed-row'}, opts.map(v=>h('button',{key:v,className:active===v?'active':'',onClick:()=>{
      const wasPlaying=this.state.playing;
      cancelAnimationFrame(this._raf);
      document.querySelectorAll('video').forEach(video=>{
        video.playbackRate = v || 1;
        if(v===0 || !wasPlaying) video.pause();
        else video.play().catch(()=>{});
      });
      this.setState({ playbackSpeed:v, playing:false },()=>{ if(wasPlaying && v>0) this.togglePlay(); });
    }}, v+'x')));
  }
  buildSwatch(key,on){
    const C={speed:'#5ec9c1',flow:'#153f7a',bottlenecks:'#d4724a',predictions:'#d100d8'}[key] || '#9b9b97';
    const op=on?1:.3;
    let inner;
    if(key==='flow') inner=[h('line',{key:'a',x1:0,y1:7,x2:16,y2:7,stroke:C,strokeWidth:1.2,strokeLinecap:'round'}),h('polygon',{key:'b',points:'16,4 22,7 16,10',fill:C})];
    else if(key==='bottlenecks') inner=[h('ellipse',{key:'a',cx:11,cy:7,rx:10,ry:5,fill:C,fillOpacity:.16}),h('ellipse',{key:'b',cx:11,cy:7,rx:5,ry:3,fill:C,fillOpacity:.42})];
    else if(key==='predictions') inner=h('path',{d:'M 0,11 C 7,2 13,2 22,8',fill:'none',stroke:C,strokeWidth:1.4,strokeLinecap:'round',strokeDasharray:'4 3'});
    else inner=h('path',{d:'M 0,11 C 6,8 9,5 13,4 L 22,3',fill:'none',stroke:C,strokeWidth:1.4,strokeLinecap:'round'});
    return h('svg',{width:22,height:14,style:{flexShrink:0,opacity:op}}, inner);
  }
  buildPill(on,key){
    const C={speed:'#5ec9c1',flow:'#153f7a',bottlenecks:'#d4724a',predictions:'#d100d8'}[key] || '#9b9b97';
    return h('div',{style:{width:26,height:14,borderRadius:8,background:on?C:'#ececea',position:'relative',transition:'background .2s',flexShrink:0}},
      h('div',{style:{position:'absolute',top:2,left:on?14:2,width:10,height:10,borderRadius:'50%',background:'#fff',boxShadow:'0 1px 2px rgba(0,0,0,.22)',transition:'left .2s'}}));
  }

  toggleLayer = (key) => {
    this.setState(st=>({layers:{...st.layers,[key]:!st.layers[key]}}), () => {
      if(this.state.layers[key]) this.resetBehaviorLayerVideos(key);
    });
  };

  resetBehaviorLayerVideos(key){
    window.setTimeout(()=>{
      document.querySelectorAll(`video.behavior-map-video[data-layer-key="${key}"]`).forEach(video=>{
        try { video.currentTime = 0; } catch(e) {}
        video.playbackRate = this.state.playbackSpeed || 1;
        if(this.state.playing) video.play().catch(()=>{});
        else video.pause();
      });
    },0);
  }

  startBehaviorLayerVideo = (e) => {
    const media = e.currentTarget;
    try { media.currentTime = 0; } catch(err) {}
    media.playbackRate = this.state.playbackSpeed || 1;
    if(this.state.playing) media.play().catch(()=>{});
    else media.pause();
  };

  startMediaAtBeginning = (e) => {
    const media = e.currentTarget;
    if(media.dataset.mpStartedAtZero) return;
    media.dataset.mpStartedAtZero = '1';
    try { media.currentTime = 0; } catch(err) {}
    const play = media.play?.();
    if(play && typeof play.catch === 'function') play.catch(()=>{});
  };

  portraitVideoStyle(width, height){
    return {
      width: height+'px',
      height: width+'px',
      objectFit:'cover',
      objectPosition:'center center',
      background:'#111',
      display:'block',
      position:'relative',
      left:'50%',
      top:'50%',
      transform:'translate(-50%, -50%) rotate(90deg)',
      transformOrigin:'50% 50%',
    };
  }

  // â”€â”€ playback â”€â”€
  togglePlay = () => {
    const syncVideos=(playing)=>{
      window.setTimeout(()=>{
        document.querySelectorAll('video').forEach(v=>{
          v.playbackRate = this.state.playbackSpeed || 1;
          if(playing) v.play().catch(()=>{});
          else v.pause();
        });
      },0);
    };
    if(this.state.playing){ this.setState({playing:false},()=>syncVideos(false)); cancelAnimationFrame(this._raf); return; }
    if(this.state.playbackSpeed===0){ this.setState({playing:false},()=>syncVideos(false)); return; }
    if(this.state.progress>=1) this.setState({progress:0});
    this.setState({playing:true},()=>syncVideos(true));
    const dur = this.HSEC[this.state.horizon]*1000 / Math.max(.01,this.state.playbackSpeed);
    let last=null;
    const tick=(ts)=>{
      if(last){ const dt=ts-last; this.setState(s=>{ let np=s.progress+dt/dur; if(np>=1){ np=1; } return {progress:np}; }); }
      last=ts;
      if(this.state.progress>=1){ this.setState({playing:false}); return; }
      this._raf=requestAnimationFrame(tick);
    };
    this._raf=requestAnimationFrame(tick);
  };
  rewind = () => {
    cancelAnimationFrame(this._raf);
    document.querySelectorAll('video').forEach(v=>{ v.pause(); v.currentTime=0; });
    this.setState({playing:false,progress:0});
  };
  scrub = (e) => {
    const r=e.currentTarget.getBoundingClientRect();
    const p=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
    cancelAnimationFrame(this._raf);
    this.setState({playing:false,progress:p});
  };

  startDragDetect = (e) => { const p=this.state.panels.detect; this.setState({dragging:{key:'detect',dx:e.clientX-p.x,dy:e.clientY-p.y}}); };
  startDragMetrics = (e) => { const p=this.state.panels.metrics; this.setState({dragging:{key:'metrics',dx:e.clientX-p.x,dy:e.clientY-p.y}}); };
  startDockDetect = (e) => { this._dockClickOk = true; const p=this.state.docks.detect; this.setState({dragging:{kind:'dock',key:'detect',dx:e.clientX-p.x,dy:e.clientY-p.y,sx:e.clientX,sy:e.clientY}}); };
  startDockMetrics = (e) => { this._dockClickOk = true; const p=this.state.docks.metrics; this.setState({dragging:{kind:'dock',key:'metrics',dx:e.clientX-p.x,dy:e.clientY-p.y,sx:e.clientX,sy:e.clientY}}); };

  renderVals(){
    const s=this.state, L=s.layers;
    const dataset = s.dataset || this.buildPlacaDataset();
    const layerDefs=[
      ['speed','Speed','speed population map'],
      ['flow','Flow fields','direction vectors'],
      ['bottlenecks','Bottlenecks','congestion cells'],
      ['predictions','Predictions','future paths'],
    ];
    const layerList=layerDefs.map(([key,label,desc])=>({
      key,label,desc,
      op: L[key]?1:0.45,
      swatch: this.buildSwatch(key,L[key]),
      pill: this.buildPill(L[key],key),
      toggle: ()=>this.toggleLayer(key),
    }));

    const md=s.realData?.metrics || {};
    const metricsData=[
      ['avg speed', md.avgSpeed ? md.avgSpeed.toFixed(2)+' m/s' : '--', Math.min(100,Math.round((md.avgSpeed||0)*25)), '#5ec9c1'],
      ['track count', md.trackCount ? String(md.trackCount) : '--', Math.min(100,Math.round((md.trackCount||0)/8)), '#7ab894'],
      ['flow vectors', md.flowCount ? String(md.flowCount) : '--', Math.min(100,Math.round((md.flowCount||0)/40)), '#7ab894'],
      ['bottlenecks', md.bottleneckCount ? String(md.bottleneckCount) : '--', Math.min(100,Math.round((md.bottleneckCount||0)*7)), '#d4724a'],
    ];
    const metrics=metricsData.map(([label,display,pct,color])=>({label,display,pct,color}));

    const projDefs=[...this.savedProjectDefs(), ...this.projectLibraryDefs()].map(p=>[p.id,p.name,p.meta,p.date||'',p.dot,p.thumb]);
    const projects=projDefs.map(([id,name,meta,date,dot,thumb])=>{
      const active=s.project===id;
      return {id,name,meta,date,dot,
        thumb,
        select:()=>this.setState({project:id}),
        bg: active?'#f4f4f1':'#fff',
        border: active?'#c4c4c0':'#ededea',
      };
    });

    const horizonTicks=[40,60,100,200,400].map((v,i)=>({
      label:'H'+v, pct: i/4*100, color: s.horizon===v?'#16181a':'#bcbcb8'
    }));

    const hsec=this.HSEC[s.horizon];
    const tsec=Math.round(s.progress*hsec);

    return {
      goHome: ()=>this.setState({screen:'home',playing:false}),
      startAnalysis: ()=>this.setState({screen:'upload'}),
      skipToWorkspace: ()=>this.setState({screen:'workspace'}),
      showSpine: s.screen!=='home',
      spine: this.buildSpine(),
      isHome: s.screen==='home',
      isUpload: s.screen==='upload',
      isCalibrate: s.screen==='calibrate',
      isEncode: s.screen==='encode',
      isProcess: s.screen==='process',
      isWorkspace: s.screen==='workspace',
      homeBg: this.buildHomeBg(),

      // profile
      openProfile: ()=>this.setState({profileOpen:true}),
      closeProfile: ()=>this.setState({profileOpen:false}),
      profileOpen: s.profileOpen,
      profileSections: [
        { title:'Recent Projects', items:[
          { name:'Mobility - Passeig Colom', meta:'312 tracks · model C', side:'today', thumb:this.thumb('redbridge') },
          { name:'Pedestrian Angular - Montjuic Study', meta:'287 tracks · model C', side:'yesterday', thumb:this.thumb('montjuic') },
          { name:'Flow Analysis - Placa Catalunya', meta:'441 tracks · 4 layers', side:'3 days', thumb:this.thumb('catalunya') },
        ]},
        { title:'Saved Studies', items:[
          { name:'Crossing Behavior - Montjuic Stairs', meta:'198 tracks · model B', side:'1 week', thumb:this.thumb('stairs') },
          { name:'Mobility - Passeig Colom', meta:'312 tracks · model C', side:'saved', thumb:this.thumb('redbridge') },
        ]},
        { title:'Export History', items:[
          { name:'studio_drawing_pe_morning.svg', meta:'drawing export', side:'today', dot:'#16181a' },
          { name:'trajectories_world.csv', meta:'dataset export', side:'yesterday', dot:'#16181a' },
          { name:'flow_sequence_pe.mp4', meta:'sequence export', side:'3 days', dot:'#16181a' },
        ]},
      ],

      // export drawing + save analysis modals
      openExport: this.openExport, closeExport: this.closeExport,
      exportOpen: s.exportOpen, exportPNG: this.exportPNG, exportSVG: this.exportSVG,
      exportSnapshotSrc: s.exportSnapshotSrc,
      openSave: this.openSave, closeSave: this.closeSave,
      saveOpen: s.saveOpen, saveSelected: s.saveSelected, saveNewName: s.saveNewName, saveConfirm: s.saveConfirm,
      selectSaveProject: this.selectSaveProject, onSaveNewName: this.onSaveNewName, commitSave: this.commitSave,
      projectLibrary: [...this.savedProjectDefs(), ...this.projectLibraryDefs()],

      // upload
      dataset,
      datasetLabel: dataset.dataset_id === 'placa_espanya' ? 'Placa Espanya dataset' : dataset.dataset_id,
      selectedVideoFilename: dataset.selected_video_filename,
      selectedPlanFilename: dataset.selected_plan_filename,
      videoInputRef: this.videoInputRef,
      planInputRef: this.planInputRef,
      onVideoSelected: this.handleVideoSelected,
      onPlanSelected: this.handlePlanSelected,
      uVideo: !!dataset.selected_video_filename,
      uPlan: !!dataset.selected_plan_filename,
      notUVideo: !dataset.selected_video_filename, notUPlan: !dataset.selected_plan_filename,
      canUploadNext: !!dataset.selected_video_filename && !!dataset.selected_plan_filename,
      notCanUploadNext: !(dataset.selected_video_filename && dataset.selected_plan_filename),
      uVideoIcon: s.uVideo ? this.buildCheckIcon('#5ec9c1') : this.buildUploadIcon(),
      uPlanIcon: s.uPlan ? this.buildCheckIcon('#9d88c4') : this.buildUploadIcon(),
      fillVideo: this.triggerVideoPicker, fillPlan: this.triggerPlanPicker, toCalibrate: this.toCalibrate,

      // calibration
      calibVideo: this.buildCalibVideo(),
      calibPlan: this.buildCalibPlan(),
      cpLegend: this.buildCPLegend(),
      calibration: s.calibration,
      cpMatched: s.calibration.matchedPoints,
      rmse: s.calibration.rmse === null ? '--' : s.calibration.rmse.toFixed(2)+' m',
      calibValid: s.calibration.calibrationComplete,
      notCalibValid: !s.calibration.calibrationComplete,
      calibValidLabel: s.calibration.homographyStatus,
      calibValidColor: s.calibration.calibrationComplete ? '#7ab894' : '#d4724a',
      homographyStatus: s.calibration.homographyStatus,
      sharedSpatialCoordinates: dataset.sharedSpatialCoordinates,
      toEncode: this.toEncode,

      // encoding
      encScene: this.buildEncScene(),
      encList: this.buildEncList(),
      encoding: s.encoding,
      encStep: s.encStep,
      encDone: s.encStep>=7,
      notEncDone: !(s.encStep>=7),
      playEncoding: this.playEncoding,
      toProcess: this.toProcess,

      // processing
      procList: this.buildProcList(),
      procPreview: this.buildProcPreview(),
      procDone: s.procDone,
      procTracks: String(s.procDone>=1 ? (s.realData?.metrics?.trackCount || 0) : 0).padStart(3,'0'),
      procPreds: s.procDone>=5 ? '12' : 'â€”',
      procMetrics: [
        ['Tracks Loaded', String(s.realData?.metrics?.trackCount || s.processing.pedestriansTracked).padStart(3,'0')],
        ['Trajectory Count', String(s.realData?.metrics?.trackCount || s.processing.trajectoryCount).padStart(3,'0')],
        ['Average Speed', s.realData?.metrics?.avgSpeed ? s.realData.metrics.avgSpeed.toFixed(2)+' m/s' : '--'],
        ['Flow Vectors', s.realData?.metrics?.flowCount ? String(s.realData.metrics.flowCount) : '--'],
      ],
      procSummary: [
        ['Dataset', 'Placa Espanya'],
        ['Calibration', s.calibration.calibrationComplete ? 'Valid' : 'Pending'],
        ['Encoding', s.encoding.encodingComplete ? 'Complete' : 'Pending'],
        ['Behavior Dataset', s.processing.behaviorDataset],
      ],
      procComplete: s.procDone>=6,

      scene: this.buildScene(),
      canvasTransform: 'none',
      aestheticBar: this.buildAesthetic(),
      horizonRow: this.buildHorizonRow(),
      speedRow: this.buildSpeedRow(),
      layerList, metrics, projects, horizonTicks,

      pDetectX: s.panels.detect.open ? s.panels.detect.x : s.docks.detect.x,
      pDetectY: s.panels.detect.open ? s.panels.detect.y : s.docks.detect.y,
      pMetricsX: s.panels.metrics.open ? s.panels.metrics.x : s.docks.metrics.x,
      pMetricsY: s.panels.metrics.open ? s.panels.metrics.y : s.docks.metrics.y,
      detectOpen: s.panels.detect.open, metricsOpen: s.panels.metrics.open,
      detectCaret: s.panels.detect.open?'-':'+', metricsCaret: s.panels.metrics.open?'-':'+',
      detectIcon: this.buildPanelDockIcon('video'),
      metricsIcon: this.buildPanelDockIcon('metrics'),
      toggleDetect: ()=>{ if(this._dockClickOk === false){ this._dockClickOk = true; return; } this.setState(st=>({panels:{...st.panels,detect:{...st.panels.detect,open:!st.panels.detect.open}}})); },
      toggleMetrics: ()=>{ if(this._dockClickOk === false){ this._dockClickOk = true; return; } this.setState(st=>({panels:{...st.panels,metrics:{...st.panels.metrics,open:!st.panels.metrics.open}}})); },
      railOpen: s.railOpen,
      toggleRail: (key)=>this.setState(st=>({railOpen:{...st.railOpen,[key]:!st.railOpen[key]}})),
      startDragDetect: this.startDragDetect, startDragMetrics: this.startDragMetrics,
      startDockDetect: this.startDockDetect, startDockMetrics: this.startDockMetrics,
      detectVideo: this.buildDetectVideo(),

      togglePlay: this.togglePlay, rewind: this.rewind, scrub: this.scrub,
      playing: s.playing, progressPct: (s.progress*100).toFixed(1),
      clockLabel: 't +'+tsec+'s',
      icoPlay: this.buildPlayIcon(s.playing),
      icoRewind: this.buildRewindIcon(),
    };
  }

  buildPlayIcon(playing){
    return playing
      ? h('svg',{width:11,height:11,viewBox:'0 0 10 10',fill:'currentColor'},h('rect',{x:1,y:0,width:3,height:10}),h('rect',{x:6,y:0,width:3,height:10}))
      : h('svg',{width:11,height:11,viewBox:'0 0 10 10',fill:'currentColor'},h('polygon',{points:'1,0 10,5 1,10'}));
  }
  buildUploadIcon(){
    return h('svg',{width:30,height:30,viewBox:'0 0 24 24',fill:'none',stroke:'#c4c4c0',strokeWidth:1.2},h('path',{d:'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4'}),h('polyline',{points:'17 8 12 3 7 8'}),h('line',{x1:12,y1:3,x2:12,y2:15}));
  }
  buildCheckIcon(c){
    return h('svg',{width:30,height:30,viewBox:'0 0 24 24',fill:'none',stroke:c,strokeWidth:1.4},h('polyline',{points:'20 6 9 17 4 12'}));
  }
  buildRewindIcon(){
    return h('svg',{width:12,height:12,viewBox:'0 0 12 12',fill:'currentColor'},h('polygon',{points:'5,2 5,10 1,6'}),h('polygon',{points:'11,2 11,10 6,6'}));
  }
  buildPanelDockIcon(kind){
    if(kind==='video'){
      return h('svg',{width:17,height:17,viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.8,strokeLinecap:'round',strokeLinejoin:'round'},
        h('rect',{x:3,y:6,width:13,height:12,rx:2}),
        h('path',{d:'M16 10l5-3v10l-5-3z'})
      );
    }
    return h('svg',{width:17,height:17,viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.8,strokeLinecap:'round',strokeLinejoin:'round'},
      h('path',{d:'M4 20L20 4'}),
      h('path',{d:'M7 17l-2-2'}),
      h('path',{d:'M11 13l-2-2'}),
      h('path',{d:'M15 9l-2-2'})
    );
  }
  buildHomeBg(){
    const k=[h('rect',{key:'bg',width:1200,height:700,fill:'#fff'})];
    for(let x=0;x<=1200;x+=40) k.push(h('line',{key:'vx'+x,x1:x,y1:0,x2:x,y2:700,stroke:'#f4f4f2',strokeWidth:.5}));
    for(let y=0;y<=700;y+=40) k.push(h('line',{key:'vy'+y,x1:0,y1:y,x2:1200,y2:y,stroke:'#f4f4f2',strokeWidth:.5}));
    // faint trajectory arcs around a center
    const cx=600,cy=350;
    const cols=['#5ec9c1','#d97a74','#c98d4a','#9d88c4','#7ab894'];
    for(let i=0;i<10;i++){
      const r=120+i*22, a0=i*30, a1=a0+150+i*8;
      let d=`M ${cx+r*Math.cos(a0*Math.PI/180)},${cy+r*Math.sin(a0*Math.PI/180)}`;
      for(let a=a0;a<a1;a+=10) d+=` L ${cx+r*Math.cos(a*Math.PI/180)},${cy+r*Math.sin(a*Math.PI/180)}`;
      k.push(h('path',{key:'tr'+i,d,fill:'none',stroke:cols[i%5],strokeWidth:1,opacity:.10}));
    }
    return h('svg',{width:'100%',height:'100%',viewBox:'0 0 1200 700',preserveAspectRatio:'xMidYMid slice'}, k);
  }

  // â•â•â•â•â•â•â•â• FLOW NAV + DRIVERS â•â•â•â•â•â•â•â•
    toCalibrate = () => this.setState({ screen:'calibrate' });
  toEncode = () => {
    this.setState(s => ({
      screen:'encode',
      calibration: s.calibration,
      dataset: this.buildPlacaDataset({ ...s.dataset, calibration_path: s.dataset.calibration_path || 'demo_assets/placa_espanya/calibration/calib.json' }),
    }));
    this.playEncoding();
  };
  toProcess = () => {
    this.setState({ screen:'process', procDone:0, processing:this.buildProcessingState(0), encoding:this.buildEncodingState(7) });
    clearInterval(this._pt);
    this._pt = setInterval(() => {
      this.setState(s => {
        const nd = s.procDone + 1;
      if(nd >= 6){ clearInterval(this._pt); setTimeout(()=>this.setState({screen:'workspace'}), 1800); }
        const done = Math.min(nd,6);
        return { procDone: done, processing:this.buildProcessingState(done) };
      });
    }, 980);
  };
  playEncoding = () => {
    this.setState({ encStep:0, encPlaying:true, encoding:this.buildEncodingState(0) });
    clearInterval(this._et);
    this._et = setInterval(() => {
      this.setState(s => {
        const ns = s.encStep + 1;
        if(ns >= 7){ clearInterval(this._et); return { encStep:7, encPlaying:false, encoding:this.buildEncodingState(7) }; }
        return { encStep: ns, encoding:this.buildEncodingState(ns) };
      });
    }, 720);
  };
  setHoverCP = (n) => this.setState({ hoverCP:n });

  // ===== Project library (shared by profile, studies rail, save dialog) =====
  get planThumb(){ return 'demo_assets/placa_espanya/plan/placa-espanya.png'; }
  thumb(key){
    const thumbs = {
      espanya: 'demo_assets/project_thumbnails/placa-espanya.png',
      catalunya: 'demo_assets/project_thumbnails/placa-catalunya.png',
      montjuic: 'demo_assets/project_thumbnails/placa-montjuic.png',
      stairs: 'demo_assets/project_thumbnails/stairs-montjuic-1.png',
      redbridge: 'demo_assets/project_thumbnails/red-bridge-combined.png',
    };
    return thumbs[key] || this.planThumb;
  }
  savedProjectDefs(){
    return (this.state.savedStudies || []).map((study,i)=>({
      id: `saved_${i}_${study.name.replace(/\W+/g,'_')}`,
      name: study.name,
      meta: study.meta || 'saved study',
      date: study.when || 'just now',
      thumb: study.thumb || this.thumb('espanya'),
      dot: '#16181a',
    })).reverse();
  }
  projectLibraryDefs(){
    return [
      {id:'colom_mobility', name:'Mobility - Passeig Colom', meta:'312 tracks · model C', date:'today', thumb:this.thumb('redbridge'), dot:'#5ec9c1'},
      {id:'montjuic_angular', name:'Pedestrian Angular - Montjuic Study', meta:'287 tracks · model C', date:'yesterday', thumb:this.thumb('montjuic'), dot:'#d97a74'},
      {id:'cat_flow', name:'Flow Analysis - Placa Catalunya', meta:'441 tracks · 4 layers', date:'3 days', thumb:this.thumb('catalunya'), dot:'#c98d4a'},
      {id:'stairs_crossing', name:'Crossing Behavior - Montjuic Stairs', meta:'198 tracks · model B', date:'1 week', thumb:this.thumb('stairs'), dot:'#9d88c4'},
    ];
  }

  // ===== Export drawing =====
  openExport = () => this.setState({ exportOpen:true, exportSnapshotSrc:null, exportSnapshotSvg:null }, () => this.refreshExportSnapshot());
  closeExport = () => this.setState({ exportOpen:false });
  _downloadBlob(blob, name){
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=name;
    document.body.appendChild(a); a.click();
    setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 1500);
    this.setState({ exportOpen:false });
  }

  readBlobAsDataUrl(blob){
    return new Promise((resolve,reject)=>{
      const fr=new FileReader();
      fr.onload=()=>resolve(fr.result);
      fr.onerror=reject;
      fr.readAsDataURL(blob);
    });
  }

  async inlineImageHref(href){
    if(!href || href.startsWith('data:')) return href;
    const response = await fetch(new URL(href, window.location.href).href);
    return this.readBlobAsDataUrl(await response.blob());
  }

  behaviorFallbackForVideo(src){
    const clean = (src || '').split('?')[0];
    if(clean.includes('speed_population.mp4')) return 'demo_assets/placa_espanya/behavior_maps_final/speed_population_still.png';
    if(clean.includes('flow_fields.mp4')) return 'demo_assets/placa_espanya/behavior_maps_final/flow_fields_still.png';
    if(clean.includes('bottleneck_density.mp4')) return 'demo_assets/placa_espanya/behavior_maps_final/bottleneck_density_still.png';
    return null;
  }

  videoFrameDataUrl(video, width=800, height=520){
    if(!video || video.readyState < 2) return null;
    try {
      const c=document.createElement('canvas');
      c.width=width; c.height=height;
      const ctx=c.getContext('2d');
      ctx.drawImage(video,0,0,width,height);
      return c.toDataURL('image/png');
    } catch(e){ return null; }
  }

  async buildStudioSnapshotSvgString(){
    const svg = document.querySelector('.studio-canvas-shell svg');
    if(!svg){ return null; }
    const clone = svg.cloneNode(true);
    const sourceFOs = Array.from(svg.querySelectorAll('foreignObject'));
    for(const [index,fo] of Array.from(clone.querySelectorAll('foreignObject')).entries()){
      const sourceVideo = sourceFOs[index]?.querySelector('video');
      const cloneVideo = fo.querySelector('video');
      const src = sourceVideo?.currentSrc || cloneVideo?.getAttribute('src') || '';
      const fallback = this.behaviorFallbackForVideo(src);
      let imageHref = this.videoFrameDataUrl(sourceVideo, Number(fo.getAttribute('width')) || 800, Number(fo.getAttribute('height')) || 520);
      if(!imageHref && fallback) imageHref = await this.inlineImageHref(fallback);
      if(!imageHref) { fo.remove(); continue; }
      const img = document.createElementNS('http://www.w3.org/2000/svg','image');
      ['x','y','width','height'].forEach(attr=>img.setAttribute(attr, fo.getAttribute(attr) || '0'));
      img.setAttribute('href', imageHref);
      img.setAttribute('preserveAspectRatio','xMidYMid slice');
      const videoStyle = cloneVideo?.getAttribute('style') || '';
      const foStyle = fo.getAttribute('style') || '';
      img.setAttribute('style', `${foStyle};${videoStyle}`);
      fo.replaceWith(img);
    }
    for(const img of Array.from(clone.querySelectorAll('image'))){
      const href = img.getAttribute('href') || img.getAttributeNS('http://www.w3.org/1999/xlink','href');
      if(href && !href.startsWith('data:')){
        img.setAttribute('href', await this.inlineImageHref(href));
      }
    }
    clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
  }

  refreshExportSnapshot = async () => {
    try {
      const svgString = await this.buildStudioSnapshotSvgString();
      if(!svgString) return;
      const src = 'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svgString);
      this.setState({ exportSnapshotSrc:src, exportSnapshotSvg:svgString });
    } catch(e) {
      console.warn('Could not capture Studio export snapshot', e);
    }
  };

  exportSVG = async () => {
    const data = this.state.exportSnapshotSvg || await this.buildStudioSnapshotSvgString();
    if(!data){ return; }
    this._downloadBlob(new Blob([data],{type:'image/svg+xml;charset=utf-8'}), 'motion_pixels_placa_espanya_drawing.svg');
  };
  exportPNG = async () => {
    try {
      const svgStr = this.state.exportSnapshotSvg || await this.buildStudioSnapshotSvgString();
      if(!svgStr){ return; }
      const W=2400, H=Math.round(2400*520/800);
      const img = new Image();
      img.onload = () => {
        const c=document.createElement('canvas'); c.width=W; c.height=H;
        const ctx=c.getContext('2d'); ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,W,H);
        ctx.drawImage(img,0,0,W,H);
        c.toBlob(b=>this._downloadBlob(b,'motion_pixels_placa_espanya_drawing.png'),'image/png');
      };
      img.onerror = () => this.exportSVG();
      img.src = 'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svgStr);
    } catch(e){ this.exportSVG(); }
  };

  // ===== Save analysis into a project =====
  openSave = () => this.setState({ saveOpen:true, saveSelected:null, saveNewName:'', saveConfirm:null });
  closeSave = () => this.setState({ saveOpen:false });
  selectSaveProject = (id) => this.setState({ saveSelected:id, saveNewName:'' });
  onSaveNewName = (e) => this.setState({ saveNewName:e.target.value, saveSelected:null });
  commitSave = () => {
    const name = (this.state.saveNewName||'').trim();
    const sel = [...this.savedProjectDefs(), ...this.projectLibraryDefs()].find(p=>p.id===this.state.saveSelected);
    const target = name || (sel ? sel.name : 'project');
    this.setState(st=>({
      savedStudies:[...(st.savedStudies||[]), { name:target, when:'just now', meta:sel?.meta || 'saved study', thumb:sel?.thumb || this.thumb('espanya') }],
      saveConfirm:'Saved into \u201c'+target+'\u201d',
      saveSelected:null, saveNewName:'',
      project: `saved_${(st.savedStudies||[]).length}_${target.replace(/\W+/g,'_')}`,
    }));
  };

  get CP(){
    return (this.state.calibrationPairs || []).map((pair,i)=>({
      n:pair.n,
      c:pair.c || this.cpColor(i),
      label:`pair ${String(pair.n).padStart(2,'0')}`,
      v:pair.video,
      p:pair.plan,
    }));
  }

  cpMarker(x,y,n,color,r,fs,dim,onClick){
    return h('g',{key:'cp'+n, onClick:(e)=>{ e.stopPropagation(); if(onClick) onClick(e); }, style:{cursor:'pointer', opacity:dim?0.28:1, transition:'opacity .15s'}},
      h('line',{x1:x-r*1.9,y1:y,x2:x-r*1.05,y2:y,stroke:color,strokeWidth:r*0.13,opacity:.7}),
      h('line',{x1:x+r*1.05,y1:y,x2:x+r*1.9,y2:y,stroke:color,strokeWidth:r*0.13,opacity:.7}),
      h('line',{x1:x,y1:y-r*1.9,x2:x,y2:y-r*1.05,stroke:color,strokeWidth:r*0.13,opacity:.7}),
      h('line',{x1:x,y1:y+r*1.05,x2:x,y2:y+r*1.9,stroke:color,strokeWidth:r*0.13,opacity:.7}),
      h('circle',{cx:x,cy:y,r,fill:color,opacity:.92}),
      h('text',{x,y:y+fs*0.36,textAnchor:'middle',fontSize:fs,fontFamily:"'Space Grotesk'",fontWeight:700,fill:'#fff'},n)
    );
  }

  buildCPLegend(){
    const pairs=this.CP, pending=this.state.pendingCalibPoint;
    const chips=pairs.map(cp=>{
      const hl=this.state.selectedCalibPair===cp.n || this.state.hoverCP===cp.n;
      return h('div',{key:cp.n,onMouseEnter:()=>this.setHoverCP(cp.n),onMouseLeave:()=>this.setHoverCP(null),className:'cp-pair-chip '+(hl?'selected':'')},
        h('svg',{width:16,height:16,viewBox:'0 0 16 16'},h('circle',{cx:8,cy:8,r:7,fill:cp.c,opacity:.92}),h('text',{x:8,y:11,textAnchor:'middle',fontSize:8,fontFamily:"'Space Grotesk'",fontWeight:700,fill:'#fff'},cp.n)),
        h('span',{},cp.label),
        h('button',{onClick:(e)=>{e.stopPropagation(); this.deleteCalibrationPair(cp.n);}},'delete')
      );
    });
    return h('div',{className:'cp-workflow'},
      h('div',{className:'cp-pair-list'},
        chips.length ? chips : h('span',{className:'cp-empty'},pending ? `point selected on ${pending.space} - click the matching ${pending.space==='video'?'plan':'video'} point` : 'click the video still, then click the matching plan point')
      ),
      h('div',{className:'cp-actions'},
        h('button',{className:'btn secondary',onClick:this.undoCalibrationPair,disabled:!pairs.length},'undo last'),
        h('button',{className:'btn secondary',onClick:this.clearCalibrationPairs,disabled:!pairs.length && !pending},'clear all')
      )
    );
  }

  buildCalibVideo(){
    const s=this.state, hover=s.hoverCP, r=this.calibVideoRect;
    const videoSrc=s.dataset?.tracked_video_path || s.dataset?.demo_video_path;
    const k=[
      h('rect',{key:'bg',width:800,height:520,fill:'#151515'}),
      h('foreignObject',{key:'video',x:r.x,y:r.y,width:r.w,height:r.h,style:{overflow:'hidden'}},
        h('video',{className:'calib-tracking-video',src:videoSrc,muted:true,autoPlay:true,loop:true,playsInline:true,preload:'auto',onLoadedMetadata:this.startMediaAtBeginning,style:this.portraitVideoStyle(r.w,r.h)})
      ),
      h('rect',{key:'frame',x:r.x,y:r.y,width:r.w,height:r.h,fill:'none',stroke:'#ffffff',strokeOpacity:.42,strokeWidth:1}),
      h('text',{x:r.x+12,y:r.y+20,fontSize:8,fontFamily:'monospace',fill:'#fff',opacity:.72},'video still / pick point')
    ];
    this.CP.forEach(cp=>{
      const dim=hover!==null&&hover!==cp.n;
      const [vx,vy]=cp.v;
      k.push(h('g',{key:'camcp'+cp.n,onPointerDown:(e)=>this.startCalibrationDrag(e,cp.n,'image')},
        h('circle',{cx:vx,cy:vy,r:17,fill:'none',stroke:cp.c,strokeWidth:1.5,strokeOpacity:.55,style:{animation:'mp-ring 1.8s ease-in-out infinite'}}),
        this.cpMarker(vx,vy,cp.n,cp.c,10,9,dim,()=>this.setState({selectedCalibPair:cp.n})),
        h('text',{x:vx+15,y:vy+5,fontSize:8,fontFamily:"'Space Grotesk'",fontWeight:600,fill:cp.c,opacity:dim?.32:1},'selected')
      ));
    });
    if(s.pendingCalibPoint?.space==='video'){
      const [x,y]=s.pendingCalibPoint.point;
      k.push(h('g',{key:'pending'},h('circle',{cx:x,cy:y,r:12,fill:'none',stroke:'#fff',strokeWidth:1.2,strokeDasharray:'4 4'}),h('text',{x:x+15,y:y-8,fontSize:8,fontFamily:"'Space Grotesk'",fontWeight:600,fill:'#fff'},'point selected')));
    }
    return h('svg',{viewBox:'0 0 320 520',width:'100%',height:'100%',preserveAspectRatio:'xMidYMid meet',style:{display:'block',background:'#111'},onClick:this.calibrationVideoClick},k);
  }
  buildCalibPlan(){
    const s=this.state, hover=s.hoverCP;
    const k=[h('g',{key:'plan'},this.buildPlan())];
    k.push(h('g',{key:'pair-links',opacity:.68},this.CP.map(cp=>{
      const dim=hover!==null&&hover!==cp.n;
      return h('path',{key:'ln'+cp.n,d:`M ${cp.p[0]},${cp.p[1]} C ${cp.p[0]-40},${cp.p[1]-28} ${cp.p[0]+40},${cp.p[1]+28} ${cp.p[0]},${cp.p[1]}`,fill:'none',stroke:cp.c,strokeWidth:1.2,strokeDasharray:'7 5',opacity:dim?.2:.72});
    })));
    this.CP.forEach(cp=>{
      const dim=hover!==null&&hover!==cp.n;
      k.push(h('g',{key:'g'+cp.n,onPointerDown:(e)=>this.startCalibrationDrag(e,cp.n,'plan')},
        h('circle',{cx:cp.p[0],cy:cp.p[1],r:21,fill:'none',stroke:cp.c,strokeWidth:1.3,strokeOpacity:.5,style:{animation:'mp-ring 1.8s ease-in-out infinite'}}),
        this.cpMarker(cp.p[0],cp.p[1],cp.n,cp.c,10,9,dim,()=>this.setState({selectedCalibPair:cp.n})),
        h('text',{x:cp.p[0]+13,y:cp.p[1]-3,fontSize:8,fontFamily:"'Space Grotesk'",fontWeight:500,fill:cp.c,opacity:dim?0.3:0.9},cp.label)
      ));
    });
    if(s.pendingCalibPoint?.space==='plan'){
      const [x,y]=s.pendingCalibPoint.point;
      k.push(h('g',{key:'pending'},h('circle',{cx:x,cy:y,r:12,fill:'none',stroke:'#16181a',strokeWidth:1.2,strokeDasharray:'4 4'}),h('text',{x:x+15,y:y-8,fontSize:8,fontFamily:"'Space Grotesk'",fontWeight:600,fill:'#16181a'},'point selected')));
    }
    return h('svg',{viewBox:'0 0 800 520',width:'100%',height:'100%',preserveAspectRatio:'xMidYMid meet',style:{display:'block',background:'#fff'},onClick:this.calibrationPlanClick},k);
  }

  buildEncScene(){
    const step=this.state.encStep;
    const k=[h('g',{key:'plan'},this.buildPlan())];
    if(step>=2){
      k.push(h('g',{key:'boundary'},
        h('rect',{x:58,y:78,width:688,height:365,fill:'none',stroke:'#5ec9c1',strokeWidth:2,strokeDasharray:'16 9',style:{animation:'mp-flow .9s linear infinite'}}),
        h('path',{d:'M 80,258 L 742,258 M 402,88 L 402,438 M 96,118 L 705,408 M 705,118 L 96,408',stroke:'#5ec9c1',strokeWidth:.8,strokeOpacity:.34,strokeDasharray:'7 8'})
      ));
    }
    if(step>=3){
      k.push(h('g',{key:'walkable'},
        h('rect',{x:58,y:78,width:688,height:365,fill:'#7ab894',fillOpacity:.13,stroke:'#7ab894',strokeWidth:.9,opacity:step===3 ? .78 : 1}),
        h('path',{d:'M 84,260 C 174,214 265,204 355,238 S 546,304 710,255',fill:'none',stroke:'#7ab894',strokeWidth:1.2,strokeOpacity:.55})
      ));
    }
    if(step>=4){
      k.push(h('g',{key:'obstacle-mask'},
        h('image',{href:'demo_assets/placa_espanya/behavior_maps/bottleneck_density_still.png',x:0,y:0,width:800,height:520,preserveAspectRatio:'xMidYMid meet',opacity:.22}),
        h('rect',{x:364,y:230,width:74,height:62,fill:'#d97a74',fillOpacity:.16,stroke:'#d97a74',strokeWidth:1}),
        h('rect',{x:594,y:300,width:58,height:42,fill:'#d97a74',fillOpacity:.12,stroke:'#d97a74',strokeWidth:.8})
      ));
    }
    if(step>=5){
      const pts=[[92,258],[402,92],[402,434],[704,258],[154,116],[654,402],[642,116],[154,402]];
      k.push(h('g',{key:'entrances'},pts.map((p,i)=>h('g',{key:i},h('circle',{cx:p[0],cy:p[1],r:6,fill:'#c98d4a',fillOpacity:.72}),h('line',{x1:p[0]-13,y1:p[1],x2:p[0]+13,y2:p[1],stroke:'#c98d4a',strokeWidth:1.2,strokeLinecap:'round'})))));
    }
    if(step>=6){
      k.push(h('image',{key:'distance-speed',href:'demo_assets/placa_espanya/behavior_maps/speed_population_still.png',x:0,y:0,width:800,height:520,preserveAspectRatio:'xMidYMid meet',opacity:.34}));
    }
    if(step>=7){
      const gl=[]; for(let x=64;x<=736;x+=32) gl.push(h('line',{key:'x'+x,x1:x,y1:76,x2:x,y2:444,stroke:'#7aacc8',strokeWidth:.45,strokeOpacity:.32}));
      for(let y=84;y<=436;y+=32) gl.push(h('line',{key:'y'+y,x1:56,y1:y,x2:744,y2:y,stroke:'#7aacc8',strokeWidth:.45,strokeOpacity:.32}));
      k.push(h('g',{key:'grid'},gl));
    }
    return h('svg',{viewBox:'0 0 800 520',width:'100%',height:'100%',preserveAspectRatio:'xMidYMid meet',style:{display:'block'}},k);
  }
  buildEncList(){
    const step=this.state.encStep;
    const defs=[
      ['Plan Source','real 2D plan loaded','#a6a6a2'],
      ['Boundary Extraction','walkable perimeter trace','#5ec9c1'],
      ['Walkable Surface','passable region mask','#7ab894'],
      ['Obstacle Mask','obstruction and density outputs','#d97a74'],
      ['Entrances','access points on real plan','#c98d4a'],
      ['Distance Field','speed population field','#9d88c4'],
      ['Context Grid','machine-readable real plan cells','#7aacc8']
    ];
    return h('div',{style:{display:'flex',flexDirection:'column',gap:2}}, defs.map((d,i)=>{
      const done=step>=i+1, active=step===i+1;
      return h('div',{key:i,style:{display:'flex',alignItems:'center',gap:11,padding:'9px 0',borderBottom:'1px solid #f0f0ed'}},
        h('span',{style:{width:10,height:10,borderRadius:'50%',background:done?d[2]:'transparent',border:done?'none':'1.4px solid #d4d4d0',flexShrink:0,animation:active?'mp-pulse 1.4s ease-in-out infinite':'none'}}),
        h('div',{style:{flex:1}},
          h('div',{style:{fontFamily:"'Space Grotesk'",fontWeight:done?600:500,fontSize:12,color:done?'#16181a':'#aeaea9'}},d[0]),
          h('div',{style:{fontFamily:"'Space Grotesk'",fontWeight:300,fontSize:9,letterSpacing:'.03em',color:'#bcbcb8',marginTop:1}},d[1])
        ),
        h('span',{style:{fontFamily:"'Space Grotesk'",fontWeight:500,fontSize:10,color:done?d[2]:'#cfcfca'}},done?'encoded':active?'...':'--')
      );
    }));
  }

  buildProcList(){
    const done=this.state.procDone;
    const steps=[
      ['Detection','video frames -> pedestrian detections','128 agents'],
      ['Tracking','ByteTrack IDs -> trajectories','312 tracks'],
      ['Calibration','homography -> shared coordinates','RMSE 0.41 m'],
      ['Encoding','plan layers -> context tensor','5 layers'],
      ['Behavior Metrics','speed + flow + bottleneck maps','3 maps'],
      ['Prediction','reserved for real forecast connection','next pass']
    ];
    return h('div',{style:{display:'flex',flexDirection:'column',width:330}}, steps.map((st,i)=>{
      const isDone=i<done, isActive=i===done;
      return h('div',{key:i,style:{display:'flex',alignItems:'flex-start',gap:13,padding:'11px 0',borderBottom:i<steps.length-1?'1px solid #f0f0ed':'none'}},
        h('span',{style:{width:9,height:9,borderRadius:'50%',background:isDone?'#7ab894':'transparent',border:isDone?'none':'1.5px solid '+(isActive?'#16181a':'#d4d4d0'),flexShrink:0,marginTop:3,animation:isActive?'mp-pulse 1.2s ease-in-out infinite':'none'}}),
        h('div',{style:{flex:1}},
          h('div',{style:{fontFamily:"'Space Grotesk'",fontWeight:isDone||isActive?600:500,fontSize:12.5,color:isDone||isActive?'#16181a':'#aeaea9'}},st[0]),
          h('div',{style:{fontFamily:"'Space Grotesk'",fontWeight:300,fontSize:9.2,letterSpacing:'.03em',color:'#bcbcb8',marginTop:2}},st[1]),
          h('div',{style:{height:2,background:'#efefec',borderRadius:1,marginTop:7,overflow:'hidden'}},h('div',{style:{height:'100%',width:isDone?'100%':isActive?'62%':'0%',background:isDone?'#7ab894':'#16181a',transition:'width .5s'}}))
        ),
        h('span',{style:{fontFamily:"'Space Grotesk'",fontWeight:500,fontSize:9.2,color:isDone?'#7ab894':'#bcbcb8',whiteSpace:'nowrap',marginTop:2}},isDone?st[2]:isActive?'...':'--')
      );
    }));
  }
  buildProcPreview(){
    const done=this.state.procDone;
    // Progressive behavioral-intelligence build: each map layer comes online on top of the
    // last, on a dark stage, so the dataset visibly becomes a layered intelligence surface.
    const layers=[
      ['speed','demo_assets/placa_espanya/behavior_maps_final/speed_population.mp4', 2, 'Speed'],
      ['flow','demo_assets/placa_espanya/behavior_maps_final/flow_fields.mp4', 3, 'Flow Fields'],
      ['bottlenecks','demo_assets/placa_espanya/behavior_maps_final/bottleneck_density.mp4', 4, 'Bottlenecks'],
      ['predictions', this.state?.dataset?.prediction_visual_path || 'demo_assets/placa_espanya/prediction_maps/prediction_anim_placa_espanya_01_plain.mp4', 5, 'Predictions'],
    ];
    const comp = {
      speed: { mixBlendMode:'multiply', opacity:.64, filter:'saturate(1.22) contrast(1.12) brightness(.98)' },
      flow: { mixBlendMode:'multiply', opacity:.76, filter:'invert(1) hue-rotate(180deg) saturate(1.58) contrast(1.18) brightness(.92)' },
      bottlenecks: { mixBlendMode:'multiply', opacity:.7, filter:'saturate(1.26) contrast(1.16) brightness(.98)' },
      predictions: { mixBlendMode:'multiply', opacity:.68, filter:'invert(1) hue-rotate(180deg) saturate(1.5) contrast(1.15) brightness(.96)' },
    };
    const k=[
      h('rect',{key:'bg',width:800,height:520,fill:'#ffffff'}),
      h('g',{key:'plan'},this.buildPlan()),
      h('rect',{key:'veil',width:800,height:520,fill:'#f8f9f7',opacity:.2})
    ];
    layers.forEach(([key,src,threshold])=>{
      if(done>=threshold){
        const built=Math.min(1,(done-threshold)*0.5+0.55);
        const c=comp[key] || {};
        k.push(h('foreignObject',{key:'pv-'+key,x:0,y:0,width:800,height:520,style:{overflow:'hidden',mixBlendMode:c.mixBlendMode||'normal',opacity:built*(c.opacity||1),transition:'opacity .6s ease'}},
          h('video',{src,muted:true,autoPlay:true,loop:true,playsInline:true,preload:'metadata',onLoadedMetadata:this.startMediaMidpoint,style:{width:'800px',height:'520px',objectFit:'cover',display:'block',filter:c.filter||'none'}})
        ));
      }
    });
    const online=layers.filter(l=>done>=l[2]);
    online.forEach((l,i)=>{
      k.push(h('text',{key:'lbl'+l[0],x:18,y:30+i*20,fontFamily:"'Space Grotesk'",fontWeight:600,fontSize:11,letterSpacing:'.04em',fill:'#16181a',opacity:.86},'+ '+l[3]));
    });
    if(done<6) k.push(h('rect',{key:'scan',x:0,y:Math.max(0,470-(done*78)),width:800,height:26,fill:'#16181a',opacity:.04}));
    return h('svg',{viewBox:'0 0 800 520',width:'100%',height:'100%',preserveAspectRatio:'xMidYMid meet',style:{display:'block'}},k);
  }

  startMediaMidpoint = (e) => {
    const media = e.currentTarget;
    if(media.dataset.mpStartedMidpoint) return;
    media.dataset.mpStartedMidpoint = '1';
    if(Number.isFinite(media.duration) && media.duration > 1){
      media.currentTime = media.duration * 0.5;
    }
    const play = media.play?.();
    if(play && typeof play.catch === 'function') play.catch(()=>{});
  };


}

Object.assign(MotionPixelsApp.prototype, window.MotionPixelsRenderMethods);
ReactDOM.createRoot(document.getElementById('root')).render(h(MotionPixelsApp));


