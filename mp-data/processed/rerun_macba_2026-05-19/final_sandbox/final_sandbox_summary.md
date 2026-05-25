# Final Sandbox — Summary

Isolated overfit-style review experiment using the rerun's calibrated trajectories and v2.1C spatial encoding. Nothing outside `final_sandbox/` was modified.

## Pipeline

| step | script | primary output | status |
|---|---|---|---|
| dataset | `make_final_sandbox_dataset.py` | `mp-data\processed\rerun_macba_2026-05-19\final_sandbox\final_sandbox_dataset.csv` | ✓ |
| train | `train_lstm_final_sandbox.py` | `mp-data\processed\rerun_macba_2026-05-19\final_sandbox\lstm_final.pth` | ✓ |
| predict | `visualize_final_sandbox_predictions.py` | `mp-data\processed\rerun_macba_2026-05-19\final_sandbox\prediction_contact_sheet.png` | ✓ |
| overlay | `overlay_final_sandbox_predictions_on_plan.py` | `mp-data\processed\rerun_macba_2026-05-19\final_sandbox\plan_prediction_overlay_contact_sheet.png` | ✓ |

## Headline output paths

- Dataset: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\final_sandbox_dataset.csv`
- Model:   `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\lstm_final.pth`
- Scaler:  `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\scaler.pkl`
- Loss curve: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\loss_curve.png`
- Training summary: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\training_summary.md`
- Prediction plots: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\prediction_plots/`
- Prediction contact sheet: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\prediction_contact_sheet.png`
- Plan-overlay plots: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\plan_overlay_plots/`
- Plan-overlay contact sheet: `C:\Users\OWNER\Desktop\IAAC\IAAC_Thesis\motion-pixels\mp-data\processed\rerun_macba_2026-05-19\final_sandbox\plan_prediction_overlay_contact_sheet.png`

## Acceptance check

- [x] dataset created from new calibrated Skate 1 data
- [x] trained model + scaler present
- [x] loss curve PNG saved
- [x] prediction contact sheet saved
- [x] plan overlay contact sheet saved

All artefacts under `mp-data\processed\rerun_macba_2026-05-19\final_sandbox`.