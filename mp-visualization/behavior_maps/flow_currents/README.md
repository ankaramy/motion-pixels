# Flow Currents

A seamless animated **flow simulation** over the static Trace + Dust Hero V3
(Plaça Catalunya). The hero is a **static field** — plan, linework, and trace
strings are fully present in every frame, never revealed over time. Glowing
particles flow **along the real trajectory geometry** (the trajectories ARE the
flow field), so the viewer reads continuous **urban currents** — fiber-optic /
river-current / field-line energy — not animated trajectories.

**Visualization-only.** Built from existing tracked trajectories. No retracking,
recalibration, model inference, or source-data modification.

## Run
```
cd mp-visualization/behavior_maps/flow_currents
python generate_flow_currents.py                 # full 24s loop
python generate_flow_currents.py --seconds 6 --preview   # fast check
```

## Layers
1. **Static architectural background** — V3 layered drawing (major Hough geometry
   + faint speckle-cleaned secondary detail), dim.
2. **Static trace strings** — visible throughout, dimmed so currents are active.
3. **Animated flow particles** advecting along real trajectories:
   - **Type A** — small / sharp / bright (active movement).
   - **Type B** — larger / soft / low-opacity (atmospheric dust).
4. **Glow** — particle energy blurred wide so dense corridors brighten through
   accumulation (natural pulsing, no flashing).

## Flow model (no random motion)
Each trajectory is arc-length resampled to a lookup table; a particle position is
`samples[phase·K]` with `phase = (phase0 + f/F·cycles) mod 1`. Particles travel
strictly along the path, continuously entering and exiting — no Brownian motion,
no wandering, no explosions. Colour = local trajectory density (γ 1.4; cyan/blue
sparse → violet/magenta dense). Particles are weighted onto trajectories by
length, so currents concentrate in the real corridors.

## Seamless loop
`cycles` per trajectory is an **integer**, so at frame F every particle returns to
its frame-0 phase → frame F ≡ frame 0. Cycles ≈ `speed·F/length`
(`speed ≈ 3.6 m/s`), giving roughly constant world-speed currents.

## Outputs (`outputs/`)
- `placa_catalunya_flow_currents.mp4` — **24 s seamless loop**, 25 fps, 1920×888,
  H.264 (yuv420p). Loops on repeat.
- `placa_catalunya_flow_currents_still.png` — frame-0 still.
- `placa_catalunya_flow_currents_report.md` — parameters + provenance.

MP4 (not GIF): a 20–30 s HD loop as GIF would be enormous; H.264 keeps it small
and high-quality.

## Key knobs (top of script)
`SECONDS` / `FPS`, `N_A` / `N_B` (particle counts), `A_*` / `B_*` (look),
`WIDE_*` (atmospheric glow), `TARGET_SPEED_MS` (current speed), `BASE_*` (static
field brightness), `DENS_GAMMA` (colour spread).

## Status
Static field + animated currents, single site. Built on the Trace + Dust Hero V3.
