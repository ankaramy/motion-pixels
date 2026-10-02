"""
build_split.py — MODEL_X mixed-recording track-level split.

Rules (per MODEL_X brief):
  - split by track (trajectory_id), never by row
  - no track in more than one split
  - every recording contributes tracks to train/val/test (stratified per recording)
  - 80 / 10 / 10, fixed seed 42
  - this is a MIXED-RECORDING split, NOT unseen-site generalization

Writes splits/model_x_track_split.csv and prints a per-recording table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from model_x_lib import CONFIG, WINDOW_SIZE, HORIZON  # noqa: E402

SEED = CONFIG["split"]["seed"]
TR, VA = CONFIG["split"]["train"], CONFIG["split"]["val"]
CSV = CONFIG["dataset"]["csv_path"]
OUT = HERE / "model_x_track_split.csv"
MIN_EVAL_LEN = WINDOW_SIZE + HORIZON  # 20 frames needed for an eval rollout window


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"[split] loading {CSV}")
    df = pd.read_csv(CSV, usecols=["recording_id", "trajectory_id", "timestep"])
    lengths = df.groupby("trajectory_id").size()
    rec_of = df.groupby("trajectory_id")["recording_id"].first()

    rng = np.random.default_rng(SEED)
    rows = []
    for rec, recdf in pd.DataFrame({"rec": rec_of, "len": lengths}).groupby("rec"):
        tids = recdf.index.to_numpy()
        rng.shuffle(tids)
        n = len(tids)
        n_tr = int(round(n * TR))
        n_va = int(round(n * VA))
        # guarantee every recording contributes >=1 track to val and test
        n_va = max(n_va, 1)
        n_te = max(n - n_tr - n_va, 1)
        n_tr = n - n_va - n_te
        assign = (["train"] * n_tr) + (["val"] * n_va) + (["test"] * n_te)
        for tid, spl in zip(tids, assign):
            L = int(lengths[tid])
            rows.append({"trajectory_id": tid, "recording_id": rec, "split": spl,
                         "n_rows": L,
                         "n_train_windows": max(L - WINDOW_SIZE, 0),
                         "n_eval_windows": max(L - MIN_EVAL_LEN + 1, 0)})

    out = pd.DataFrame(rows).sort_values(["recording_id", "split", "trajectory_id"])
    out.to_csv(OUT, index=False)

    # ── report ──
    print(f"[split] seed={SEED}  train/val/test = {TR}/{VA}/{1-TR-VA:.2f}")
    print(f"[split] saved -> {OUT}\n")
    print(f"{'recording':26s} {'split':6s} {'tracks':>7} {'rows':>9} "
          f"{'train_win':>10} {'eval_win':>9}")
    print("-" * 75)
    for rec in sorted(out.recording_id.unique()):
        for spl in ("train", "val", "test"):
            s = out[(out.recording_id == rec) & (out.split == spl)]
            print(f"{rec:26s} {spl:6s} {len(s):>7} {s.n_rows.sum():>9} "
                  f"{s.n_train_windows.sum():>10} {s.n_eval_windows.sum():>9}")
    print("-" * 75)
    for spl in ("train", "val", "test"):
        s = out[out.split == spl]
        print(f"{'TOTAL':26s} {spl:6s} {len(s):>7} {s.n_rows.sum():>9} "
              f"{s.n_train_windows.sum():>10} {s.n_eval_windows.sum():>9}")
    print(f"\n[split] total tracks={len(out)}  rows={out.n_rows.sum()}")
    # sanity: no track in >1 split
    assert out.trajectory_id.nunique() == len(out), "duplicate track in split!"
    print("[split] OK: every track in exactly one split.")


if __name__ == "__main__":
    main()
