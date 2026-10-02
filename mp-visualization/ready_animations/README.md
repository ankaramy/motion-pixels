# ready_animations — final best / worst / stress-test animations

This folder holds the curated GIF/PNG animations of the final model `MODEL_XC_B_CURV_LIGHT` (`best/`, `worst/`, `stress_test/`, `previews/`) and their report, [READY_ANIMATIONS_REPORT.md](READY_ANIMATIONS_REPORT.md). The predictions shown are a deterministic replay of the frozen checkpoint.

## Known missing historical file

`generate_ready_animations.py` imports its drawing and animation helper from
`mp-visualization/motion_pixels_animation_test/generate_d3_style_animation.py`
(module `generate_d3_style_animation`). **That helper file was lost.** It is not in the Git history, the project folders or the author's local archive, and it has not been reconstructed. As a result:

- the outputs in this folder are the surviving originals, produced while the helper still existed;
- `generate_ready_animations.py` is kept for provenance (selection logic, rollout wiring, captions) but **cannot run as-is**;
- the visual rules the helper implemented are documented in [`../MOTION_PIXELS_VISUALIZATION_TEMPLATE.md`](../MOTION_PIXELS_VISUALIZATION_TEMPLATE.md). Any future re-implementation must be labelled as a reconstruction, not as the original.

The underlying rollout does not depend on the lost helper. To reproduce the predictions themselves, use `mp-core/trajectory-prediction/MODEL_XC/predict_example.py` or `mp-visualization/plots_wassim/`.
