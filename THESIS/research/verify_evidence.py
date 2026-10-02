"""Read-only evidence checks for the manuscript; writes only under THESIS."""
from pathlib import Path
import csv, json, statistics

root = Path(__file__).resolve().parents[2]
exp = root / 'mp-core/trajectory-prediction/experiments'
results = {}
for h, threshold in [(20, 1), (60, 2), (100, 3), (200, 5), (400, 10)]:
    path = exp / 'MODEL_X_HORIZON_SWEEP' / f'H{h}' / 'per_window_metrics.csv'
    with path.open(encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    stats = {k: statistics.mean(float(r[k]) for r in rows if r[k] != '')
             for k in ['ade', 'fde', 'angular_err_deg']}
    stats.update(windows=len(rows), tracks=len({r['trajectory_id'] for r in rows}),
                 success_percent=100*statistics.mean(float(r['fde']) <= threshold for r in rows),
                 endpoint_normalized_percent=100*statistics.mean(
                     max(0, min(1, 1-float(r['fde'])/float(r['gt_net_disp'])))
                     if float(r['gt_net_disp']) > 1e-9 else 0 for r in rows))
    stats['missing_angular_values'] = sum(r['angular_err_deg']=='' for r in rows)
    results[f'H{h}'] = stats
split = exp / 'MODEL_X/splits/model_x_track_split.csv'
with split.open(encoding='utf8') as f:
    rows = list(csv.DictReader(f))
results['model_x_split'] = {s: sum(r['split']==s for r in rows) for s in ['train','val','test']}
results['model_x_split']['unique_tracks'] = len({r['trajectory_id'] for r in rows})
schema = exp/'schema_ablation_bridge/schema_summary.json'
results['bridge_schema'] = json.loads(schema.read_text(encoding='utf8'))
out = root/'THESIS/research/evidence_verification.json'
out.write_text(json.dumps(results, indent=2), encoding='utf8')
print(json.dumps({k:v for k,v in results.items() if k!='bridge_schema'},indent=2))
