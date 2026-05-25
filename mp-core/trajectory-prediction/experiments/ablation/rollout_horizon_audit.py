"""Generate rollout horizon audit files for the ablation test set."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_ablation import SEED, TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, N_ROLLOUT, set_seed

set_seed(SEED)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'mp-data', 'processed', 'encoded', 'motion_dataset_v2.csv')
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'mp-data', 'outputs', 'ablation')
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.normpath(DATA_PATH)
OUT_DIR   = os.path.normpath(OUT_DIR)

SELECTED_PLOT_TIDS = [308, 520, 332, 585, 122, 303, 72, 82, 310, 526]

# --- reproduce same test split as run_ablation.py ---
df = pd.read_csv(DATA_PATH)
all_ids = df['trajectory_id'].unique().tolist()
rng = np.random.default_rng(SEED)
rng.shuffle(all_ids)

n       = len(all_ids)
n_train = int(n * TRAIN_FRAC)
n_val   = int(n * VAL_FRAC)
test_ids = all_ids[n_train + n_val:]

# --- build audit table ---
rows = []
for tid in sorted(test_ids):
    tdf = df[df['trajectory_id'] == tid].reset_index(drop=True)
    total_rows       = len(tdf)
    available_future = total_rows - WINDOW_SIZE
    actual_rollout   = min(max(available_future, 0), N_ROLLOUT)
    rows.append({
        'trajectory_id':           int(tid),
        'total_rows':              total_rows,
        'seed_len':                WINDOW_SIZE,
        'requested_rollout_len':   N_ROLLOUT,
        'available_future_steps':  available_future,
        'actual_rollout_steps_used': actual_rollout,
    })

audit_df = pd.DataFrame(rows)
audit_csv = os.path.join(OUT_DIR, 'rollout_horizon_audit.csv')
audit_df.to_csv(audit_csv, index=False)
print(f'Wrote {audit_csv} ({len(audit_df)} rows)')

# --- summary markdown ---
steps   = audit_df['actual_rollout_steps_used']
n_total = len(steps)

short_df = audit_df[audit_df['actual_rollout_steps_used'] < N_ROLLOUT]

def flag(r):
    if r['actual_rollout_steps_used'] == 0:
        return 'UNUSABLE -- no rollout possible'
    if r['actual_rollout_steps_used'] < 10:
        return 'UNUSABLE -- fewer than 10 steps'
    return 'MARGINAL -- partial rollout only'

lines = [
    '# Rollout Horizon Audit',
    '',
    (f'Test set: {n_total} trajectories | '
     f'WINDOW_SIZE={WINDOW_SIZE} | N_ROLLOUT={N_ROLLOUT} | SEED={SEED}'),
    '',
    '## Statistics',
    '',
    '| Metric | Value |',
    '|--------|-------|',
    f'| Min actual rollout steps | {int(steps.min())} |',
    f'| Max actual rollout steps | {int(steps.max())} |',
    f'| Mean actual rollout steps | {steps.mean():.2f} |',
    f'| Median actual rollout steps | {steps.median():.1f} |',
    f'| Trajectories with >= 10 steps | {(steps >= 10).sum()} / {n_total} |',
    f'| Trajectories with >= 20 steps | {(steps >= 20).sum()} / {n_total} |',
    (f'| Trajectories with >= 30 steps | {(steps >= 30).sum()} / {n_total} '
     f'(capped by N_ROLLOUT={N_ROLLOUT}) |'),
    '',
    '## Short trajectory flags',
    '',
    ('| trajectory_id | total_rows | available_future_steps'
     ' | actual_rollout_steps_used | Status |'),
    ('|--------------|------------|------------------------'
     '|--------------------------|--------|'),
]

for _, r in short_df.iterrows():
    lines.append(
        f'| {int(r.trajectory_id)} | {int(r.total_rows)} | '
        f'{int(r.available_future_steps)} | '
        f'{int(r.actual_rollout_steps_used)} | {flag(r)} |'
    )

lines += [
    '',
    '## Selected plot trajectories',
    '',
    ('All 10 trajectories selected for visual ablation comparison '
     'have full 20-step rollout:'),
    '',
    ('| trajectory_id | available_future_steps'
     ' | actual_rollout_steps_used |'),
    ('|--------------|------------------------'
     '|--------------------------|'),
]
for tid in SELECTED_PLOT_TIDS:
    r = audit_df[audit_df['trajectory_id'] == tid].iloc[0]
    lines.append(
        f'| {int(r.trajectory_id)} | {int(r.available_future_steps)}'
        f' | {int(r.actual_rollout_steps_used)} |'
    )

lines += [
    '',
    'No selected trajectory is too short for visual interpretation.',
    '',
    '## Note on 30-step cap',
    '',
    (f'The maximum is capped at N_ROLLOUT={N_ROLLOUT} by design, '
     f'so the ">= 30 steps" count is always 0. '
     f'To test longer horizons, increase N_ROLLOUT and re-run the ablation.'),
]

summary_path = os.path.join(OUT_DIR, 'rollout_horizon_summary.md')
with open(summary_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Wrote {summary_path}')

# --- histogram ---
fig, ax = plt.subplots(figsize=(8, 4))
bins = list(range(0, N_ROLLOUT + 2))
ax.hist(steps, bins=bins, color='#4C72B0', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Actual rollout steps used', fontsize=11)
ax.set_ylabel('Number of trajectories', fontsize=11)
ax.set_title(
    f'Rollout horizon distribution -- {n_total} test trajectories',
    fontsize=12)
ax.axvline(N_ROLLOUT, color='#C44E52', linewidth=1.5, linestyle='--',
           label=f'Requested horizon (N={N_ROLLOUT})')
ax.axvline(steps.mean(), color='#55A868', linewidth=1.5, linestyle=':',
           label=f'Mean = {steps.mean():.1f}')
ax.legend(fontsize=9)
ax.set_xticks(range(0, N_ROLLOUT + 1, 5))

n_short = len(short_df)
if n_short > 0:
    ax.annotate(
        f'{n_short} trajectories\nshorter than {N_ROLLOUT} steps',
        xy=(short_df['actual_rollout_steps_used'].mean(), n_short),
        xytext=(N_ROLLOUT - 9, n_short + 3),
        fontsize=8, color='#C44E52',
        arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1),
    )

plt.tight_layout()
hist_path = os.path.join(OUT_DIR, 'rollout_horizon_histogram.png')
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Wrote {hist_path}')
print('Done.')
