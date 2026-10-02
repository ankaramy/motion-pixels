window.MotionPixelsRenderMethods = {
  render(){
    const v = this.renderVals();
    return h('div',{className:'app-shell'},
      h('nav',{className:'top-nav'},
        h('button',{onClick:v.goHome,className:'brand brand-logo','aria-label':'Motion Pixels home'},
          h('img',{src:'demo_assets/logos/logo3.jpg',alt:'Motion Pixels'})
        ),
        h('div',{style:{flex:1}}),
        h('div',{className:'nav-links'},
          h('span',{},'Dataset'),
          h('span',{},'Method'),
          h('button',{className:'profile-button',onClick:v.openProfile,'aria-label':'profile'},this.renderProfileIcon())
        )
      ),
      v.showSpine ? v.spine : null,
      v.isHome ? this.renderHome(v) : null,
      v.isUpload ? this.renderUpload(v) : null,
      v.isCalibrate ? this.renderCalibrate(v) : null,
      v.isEncode ? this.renderEncode(v) : null,
      v.isProcess ? this.renderProcess(v) : null,
      v.isWorkspace ? this.renderWorkspace(v) : null,
      v.profileOpen ? this.renderProfile(v) : null,
      v.exportOpen ? this.renderExportModal(v) : null,
      v.saveOpen ? this.renderSaveModal(v) : null
    );
  },

  renderHome(v){
    return h('main',{className:'home-screen'},
      h('div',{className:'home-bg'},v.homeBg),
      h('section',{className:'home-content'},
        h('div',{className:'home-logo-glitch','aria-label':'Motion Pixels'},
          h('img',{className:'home-logo home-logo-base',src:'demo_assets/logos/logo1.png',alt:'Motion Pixels',
            onError:(e)=>{ e.currentTarget.style.display='none'; }}),
          h('img',{className:'home-logo home-logo-alt',src:'demo_assets/logos/logo2.png',alt:'','aria-hidden':true})
        ),
        h('h1',{className:'home-title',style:{display:'none'}},'Motion Pixels'),
        h('p',{className:'home-copy'},'Mapping Out Spatial Intelligence'),
        h('div',{className:'home-actions'},
          h('button',{className:'btn primary',onClick:v.startAnalysis},'Start New Study'),
          h('button',{className:'btn secondary',onClick:v.skipToWorkspace},'Open Last Study')
        )
      )
    );
  },

  renderProfile(v){
    return h('div',{className:'profile-overlay',onClick:v.closeProfile},
      h('aside',{className:'profile-drawer',onClick:(e)=>e.stopPropagation()},
        h('header',{className:'profile-head'},
          h('div',{className:'profile-id'},
            h('span',{className:'profile-avatar'},this.renderProfileIcon()),
            h('div',{},h('b',{},'Antoni Gaud\u00ed'),h('small',{},'antoni.gaudi@mp.com'))
          ),
          h('button',{className:'profile-close',onClick:v.closeProfile,'aria-label':'close'},'\u00d7')
        ),
        v.profileSections.map(sec=>h('section',{key:sec.title,className:'profile-section'},
          h('h4',{},sec.title),
          h('div',{className:'profile-list'},sec.items.map((it,i)=>h('button',{key:i,className:'profile-row'+(it.thumb?' has-thumb':'')},
            it.thumb
              ? h('span',{className:'profile-thumb'},h('img',{src:it.thumb,alt:''}))
              : h('span',{className:'profile-dot',style:{background:it.dot||'#c4c4c0'}}),
            h('span',{className:'profile-row-main'},h('b',{},it.name),h('small',{},it.meta)),
            h('span',{className:'profile-row-side'},it.side)
          )))
        ))
      )
    );
  },

  renderUpload(v){
    const uploadBox = (filled, icon, title, hint, onClick, color) => h('button',{className:'upload-box'+(filled?' filled':''),onClick},
      icon,
      filled ? h('span',{className:'upload-file',style:{color}},title) : h('span',{className:'upload-label'},title),
      filled ? h('span',{className:'ready',style:{color}},'ready') : h('span',{className:'upload-hint'},hint)
    );
    return h('main',{className:'step-screen'},
      h('section',{className:'step-heading'},
        h('h2',{},'Pair Movement With Space'),
        h('p',{},"Choose footage and a spatial reference. Selected files are shown here before the workflow continues.")
      ),
      h('input',{ref:v.videoInputRef,type:'file',accept:'.mp4,.mov,video/mp4,video/quicktime',onChange:v.onVideoSelected,className:'native-file-input','aria-label':'upload footage'}),
      h('input',{ref:v.planInputRef,type:'file',accept:'.png,.jpg,.jpeg,.svg,image/png,image/jpeg,image/svg+xml',onChange:v.onPlanSelected,className:'native-file-input','aria-label':'upload spatial reference'}),
      h('div',{className:'upload-pair'},
        uploadBox(v.uVideo,v.uVideoIcon,v.uVideo?v.selectedVideoFilename:'Upload Footage','pedestrian video \u00b7 mp4 \u00b7 mov',v.fillVideo,'#3a9e96'),
        h('div',{className:'upload-join'},h('span',{}),h('b',{},'+'),h('span',{})),
        uploadBox(v.uPlan,v.uPlanIcon,v.uPlan?v.selectedPlanFilename:'Upload Spatial Reference','top-down plan \u00b7 png \u00b7 svg',v.fillPlan,'#7d6aa8')
      ),
      h('div',{className:'step-row'},
        v.canUploadNext ? h('button',{className:'btn primary',onClick:v.toCalibrate},'Continue') : h('button',{className:'btn disabled'},'Continue')
      )
    );
  },

  renderCalibrate(v){
    return h('main',{className:'calib-screen'},
      h('section',{className:'step-heading compact'},
        h('h2',{},'Calibrate Correspondence Points'),
        h('div',{className:'dataset-chip'},v.datasetLabel)
      ),
      h('div',{className:'continuity-strip'},
        h('span',{},'Footage: ',h('b',{},v.selectedVideoFilename || 'not selected')),
        h('span',{},'Plan: ',h('b',{},v.selectedPlanFilename || 'not selected')),
        h('span',{},'Frame: ',h('b',{},v.sharedSpatialCoordinates.id))
      ),
      h('div',{className:'calib-panels'},
        h('section',{className:'calib-panel'},h('header',{},h('span',{},'Video Frame \u00b7 Camera View'),h('b',{},'calibrated region')),h('div',{className:'calib-body'},v.calibVideo)),
        h('div',{className:'homography-label'},'homography H'),
        h('section',{className:'calib-panel'},h('header',{},h('span',{},'Spatial Reference \u00b7 Plan'),h('b',{},'camera FOV')),h('div',{className:'calib-body'},v.calibPlan))
      ),
      h('div',{className:'cp-legend'},v.cpLegend),
      h('div',{className:'step-row wide'},
        h('div',{className:'calib-stats'},
          h('span',{},'Matched Points: ',h('b',{},v.cpMatched)), h('i',{}),
          h('span',{},'RMSE: ',h('b',{},v.rmse)), h('i',{}),
          h('span',{style:{color:v.calibValidColor}},'Homography Status: ',h('b',{},v.homographyStatus))
        ),
        v.calibValid ? h('button',{className:'btn primary',onClick:v.toEncode},'Confirm Calibration') : h('button',{className:'btn disabled'},'Confirm Calibration')
      )
    );
  },

  renderEncode(v){
    return h('main',{className:'encode-screen'},
      h('section',{className:'step-heading compact'},
        h('h2',{},'Encoding Spatial Context'),
        h('div',{className:'dataset-chip'},v.datasetLabel)
      ),
      h('div',{className:'continuity-strip'},
        h('span',{},'Plan source: ',h('b',{},v.selectedPlanFilename || 'Placa Espanya plan')),
        h('span',{},'Coordinate frame: ',h('b',{},v.sharedSpatialCoordinates.id))
      ),
      h('div',{className:'encode-layout'},
        h('div',{className:'encode-scene'},v.encScene),
        h('aside',{className:'encode-list'},
          h('div',{className:'section-title'},'Context Layers'),
          h('div',{className:'encode-list-body'},v.encList),
          h('div',{className:'encode-actions'},
            h('button',{className:'btn secondary',onClick:v.playEncoding},'Replay'),
            v.encDone ? h('button',{className:'btn primary',onClick:v.toProcess},'Run Pipeline') : h('button',{className:'btn disabled'},'Encoding\u2026')
          )
        )
      )
    );
  },

  renderProcess(v){
    return h('main',{className:'process-screen'},
      h('section',{className:'step-heading compact'},
        h('h2',{},'Running Pipeline'),
        h('div',{className:'dataset-chip'},v.datasetLabel)
      ),
      h('div',{className:'process-layout'},
        h('section',{className:'proc-preview'},h('header',{},h('span',{},'Behavioral Intelligence Build'),h('b',{},'active')),v.procPreview,h('footer',{},h('span',{},'frame 312 / 750'),h('span',{},'25 fps'))),
        v.procList,
        h('section',{className:'proc-counters'},
          v.procMetrics.map(m=>h('div',{key:m[0]},h('span',{},m[0]),h('b',{className:m[0]==='Prediction Horizons'?'green':null},m[1]))),
          v.procComplete ? h('div',{className:'pipeline-summary'},
            v.procSummary.map(s=>h('p',{key:s[0]},h('span',{},s[0]),h('b',{},s[1])))
          ) : null
        )
      ),
      v.procComplete ? h('div',{className:'complete-note'},'Pipeline Complete') : null
    );
  },

  renderWorkspace(v){
    return h('main',{className:'workspace studio-v2'},
      h('section',{className:'studio-stage'},
        h('div',{className:'studio-canvas-shell'},
          h('div',{className:'hero-canvas'},h('div',{className:'canvas-transform',style:{transform:v.canvasTransform}},v.scene))
        ),
        h('div',{className:'study-title'},
          h('span',{className:'overline'},'Active Study'),
          h('b',{},'Placa Espanya')
        ),
        this.renderFloatingPanel('detect','01 \u00b7 Video Tracking',v.pDetectX,v.pDetectY,v.startDragDetect,v.startDockDetect,v.detectOpen,v.toggleDetect,v.detectCaret,
          h('div',{className:'tracking-body'},
            h('div',{className:'video-mini'},v.detectVideo)
          ), false, v.detectIcon),
        this.renderFloatingPanel('metrics','02 \u00b7 Metrics',v.pMetricsX,v.pMetricsY,v.startDragMetrics,v.startDockMetrics,v.metricsOpen,v.toggleMetrics,v.metricsCaret,
          h('div',{className:'metrics-body'},
            v.metrics.map(m=>h('div',{key:m.label,className:'metric'},h('div',{},h('span',{},m.label),h('b',{},m.display)),h('em',{},h('i',{style:{width:m.pct+'%',background:m.color}})))),
            h('div',{className:'dataset-row'},h('span',{},'dataset'),h('b',{},v.dataset.dataset_id)),
            h('div',{className:'dataset-row'},h('span',{},'plan'),h('b',{},v.dataset.demo_plan_path ? 'Placa Espanya' : 'not loaded'))
          ), false, v.metricsIcon)
      ),
      this.renderStudiesRail(v),
      this.renderTimeline(v)
    );
  },

  renderProfileIcon(){
    return h('svg',{width:18,height:18,viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.7,strokeLinecap:'round',strokeLinejoin:'round'},
      h('circle',{cx:12,cy:8,r:3.5}),
      h('path',{d:'M 5.5 20 C 6.5 15.8 9 14 12 14 C 15 14 17.5 15.8 18.5 20'})
    );
  },

  renderFloatingPanel(key,title,x,y,onDrag,onDockDrag,open,onToggle,caret,body,live,icon){
    if(!open){
      return h('button',{className:'floating-dock '+key,style:{left:x,top:y},onPointerDown:onDockDrag,onClick:onToggle,title},
        icon,
        h('span',{},key==='detect'?'Video Tracking':'Metrics')
      );
    }
    return h('section',{className:'floating-panel '+key+' open',style:{left:x,top:y}},
      h('header',{onPointerDown:onDrag},
        h('span',{className:'grip'},h('i',{}),h('i',{})), h('b',{},title), h('span',{className:'spacer'}),
        live ? h('span',{className:'live'},h('i',{}),'LIVE') : null,
        h('button',{onClick:(e)=>{e.stopPropagation(); onToggle();}},caret)
      ),
      open ? body : null
    );
  },

  renderStudiesRail(v){
    return h('aside',{className:'studies-rail'},
      h('section',{className:'rail-section studies '+(v.railOpen.saved?'open':'')},
        h('button',{className:'rail-tab',onClick:()=>v.toggleRail('saved')},'Saved'),
        h('h3',{},'Saved Studies'),
        h('div',{className:'project-list'},v.projects.map(p=>h('button',{key:p.id,onClick:p.select,style:{borderColor:p.border,background:p.bg}},p.thumb?h('span',{className:'proj-thumb'},h('img',{src:p.thumb,alt:''})):h('span',{style:{background:p.dot}}),h('b',{},p.name),h('em',{},p.meta),h('small',{},p.date))))
      ),
      h('section',{className:'rail-section '+(v.railOpen.layers?'open':'')},
        h('button',{className:'rail-tab',onClick:()=>v.toggleRail('layers')},'Layers'),
        h('h3',{},'Behavior Layers'),
        h('div',{className:'layer-list'},v.layerList.map(l=>h('button',{key:l.key,onClick:l.toggle},l.swatch,h('span',{},h('b',{style:{opacity:l.op}},l.label),h('small',{},l.desc)),l.pill)))
      ),
      h('section',{className:'rail-section export-panel '+(v.railOpen.export?'open':'')},
        h('button',{className:'rail-tab',onClick:()=>v.toggleRail('export')},'Export'),
        h('h3',{},'Export Controls'),
        h('div',{className:'export-list'},
          h('button',{className:'dark',onClick:v.openExport},h('b',{},'Export Drawing'),h('span',{},'PNG \u00b7 SVG')),
          h('button',{className:'light',onClick:v.openSave},h('b',{},'Save Analysis'),h('span',{},'project')),
          h('button',{},h('b',{},'Export Dataset'),h('span',{},'CSV')),
          h('button',{},h('b',{},'Export Sequence'),h('span',{},'MP4'))
        )
      )
    );
  },

  renderTimeline(v){
    return h('footer',{className:'timeline'},
      h('div',{className:'transport'},h('button',{onClick:v.rewind,title:'rewind'},v.icoRewind),h('button',{className:'play',onClick:v.togglePlay,title:'play / pause'},v.icoPlay)),
      v.speedRow,
      h('div',{className:'clock'},h('b',{},v.clockLabel),h('span',{},'simulation time')),
      h('div',{className:'scrubber',onPointerDown:v.scrub},
        h('div',{className:'ticks'},v.horizonTicks.map(t=>h('span',{key:t.label,style:{left:t.pct+'%',color:t.color}},t.label))),
        h('div',{className:'bar'},h('i',{style:{width:v.progressPct+'%'}}),h('b',{style:{left:v.progressPct+'%'}}))
      ),
      h('div',{className:'horizon'},h('span',{},'Prediction Horizons'),v.horizonRow)
    );
  },

  renderExportModal(v){
    const fmt=(label,sub,onClick)=>h('button',{className:'fmt-card',onClick},
      h('span',{className:'fmt-ext'},label),
      h('span',{className:'fmt-main'},h('b',{},label+' file'),h('small',{},sub)),
      h('span',{className:'fmt-dl'},'Download')
    );
    return h('div',{className:'mp-modal-overlay',onClick:v.closeExport},
      h('div',{className:'mp-modal',onClick:(e)=>e.stopPropagation()},
        h('header',{className:'mp-modal-head'},
          h('div',{},h('b',{},'Export Drawing'),h('small',{},'Placa Espanya · studio drawing')),
          h('button',{className:'mp-modal-close',onClick:v.closeExport},'\u00d7')
        ),
        h('div',{className:'export-preview'},h('img',{src:v.exportSnapshotSrc || 'demo_assets/placa_espanya/plan/placa-espanya.png',alt:''})),
        h('div',{className:'fmt-row'},
          fmt('PNG','raster image \u00b7 2400 px', v.exportPNG),
          fmt('SVG','vector \u00b7 editable layers', v.exportSVG)
        )
      )
    );
  },

  renderSaveModal(v){
    return h('div',{className:'mp-modal-overlay',onClick:v.closeSave},
      h('div',{className:'mp-modal save-modal',onClick:(e)=>e.stopPropagation()},
        h('header',{className:'mp-modal-head'},
          h('div',{},h('b',{},'Save Analysis'),h('small',{},'Save this study into a project')),
          h('button',{className:'mp-modal-close',onClick:v.closeSave},'\u00d7')
        ),
        h('div',{className:'save-section'},
          h('h5',{},'Select Project'),
          h('div',{className:'save-project-list'},v.projectLibrary.map(p=>h('button',{key:p.id,
            className:'save-project'+(v.saveSelected===p.id?' selected':''),onClick:()=>v.selectSaveProject(p.id)},
            p.thumb?h('span',{className:'save-thumb'},h('img',{src:p.thumb,alt:''})):h('span',{className:'save-thumb empty'}),
            h('span',{className:'save-project-main'},h('b',{},p.name),h('small',{},p.meta)),
            h('span',{className:'save-radio'+(v.saveSelected===p.id?' on':'')})
          )))
        ),
        h('div',{className:'save-section'},
          h('h5',{},'Or Create New Project'),
          h('input',{className:'save-input',type:'text',placeholder:'New project name\u2026',value:v.saveNewName,onChange:v.onSaveNewName})
        ),
        h('footer',{className:'save-foot'},
          h('button',{className:'btn secondary',onClick:v.closeSave},'Cancel'),
          (v.saveSelected || (v.saveNewName||'').trim())
            ? h('button',{className:'btn primary',onClick:v.commitSave},'Save Study')
            : h('button',{className:'btn disabled'},'Save Study')
        ),
        v.saveConfirm ? h('div',{className:'save-confirm'},v.saveConfirm) : null
      )
    );
  }
};
