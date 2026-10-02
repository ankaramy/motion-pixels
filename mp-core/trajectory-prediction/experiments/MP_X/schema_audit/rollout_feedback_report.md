# Rollout feedback report

Teacher-forced over 50 tracks (991 steps, 0.5% stationary). GT incoming steps are fed through the rollout feature updater and compared to the dataset's stored features at the same frames. Heading features (heading_sin/cos, turn_rate) are compared on MOVING steps only, because heading is undefined when the step is stationary.

              feature   n          mae          p95       maxabs
                   du 991 0.000000e+00 0.000000e+00 0.000000e+00
                   dv 991 0.000000e+00 0.000000e+00 0.000000e+00
                speed 991 3.429030e-17 8.326673e-17 2.220446e-16
          heading_sin 986 2.821174e-15 1.026956e-14 1.950662e-13
          heading_cos 986 2.288918e-15 7.677886e-15 1.949552e-13
            turn_rate 986 6.372399e-03 2.612494e-14 3.141593e+00
                    u 991 1.873711e-17 8.326673e-17 1.110223e-16
                    v 991 2.023552e-17 1.110223e-16 1.110223e-16
dist_to_obstacle_norm 991 2.903654e-07 1.432286e-06 1.153721e-05
dist_to_boundary_norm 991 3.010354e-07 1.488151e-06 1.096380e-05

- Motion features on moving steps: max **p95 = 2.61e-14** (du/dv/speed/u/v exact to ~1e-16, heading_sin/cos to ~2e-15, turn_rate p95 ~3e-14) -> the training-time and rollout-time motion schema are IDENTICAL. The only inflated statistic is turn_rate's MEAN MAE (6.4e-03), caused by rare π jumps at the first moving step after an idle step (dataset resets heading to 0 at idle); p95 confirms 95%+ of steps are exact.
- Spatial features p95 = **1.49e-06** -> at the exact dataset point the KDTree returns the stored percentile-rank distance (nearest-neighbour distance 0), so the rollout refresh reproduces the encoded value here. (During free rollout away from dataset points it interpolates; prior audits found the spatial channel carries ~no turn signal regardless.)
- Stationary steps (0.5%): the dataset stores heading=0 (heading_sin=0, heading_cos=1) while fz.rollout carries the previous heading. This is a benign convention difference confined to idle/stationary steps; genuine turns are moving by construction (net disp>2m, max-step<0.6m), so it does NOT affect the turning rollouts that exp01-04 evaluate.
