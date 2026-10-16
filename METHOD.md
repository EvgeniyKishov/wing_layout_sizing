# Accepted wing sizing settings

```text
DEFAULT FOKKER SIZING OPTIMIZATION SETUP
Accepted by user: 15 October 2026

Operational entry point: run_layout_candidate.py.
The outer layout optimizer inherits these sizing defaults. CCD remains on hold.
Historical standalone study drivers and archived sources are retained for reproduction.

FSD
Initial total thickness: 0.010 m. Face/core/face fractions: 0.4/0.2/0.4.
Update: t_new = clip(t_old * sigma_filtered / 285e6, 0.000625, 0.15).
No relaxation, square-root damping or internal move limits.
Stress: element face-layer von Mises envelope, BOT/MID/TOP of both metal faces.
Area-weighted conical centroid filter, radius 0.30 m, separate physical surfaces:
upper/lower skins, each spar, each rib. Preserve raw stresses.

Stop FSD at a solved state with filtered peak <=300 MPa AND any one of:
1. Active variable fraction <5% of ALL element thickness variables.
   Active means strictly interior thickness and filtered stress outside inclusive
   285..315 MPa band.
2. Maximum proposed relative thickness change <=1%.
3. Each of three successive absolute relative mass changes is <0.001 (0.1%):
   abs(M[k]-M[k-1])/M[k-1]. All four solved states must have filtered peak <=300 MPa.
   Configurable via --mass-tolerance. This stops FSD, not buckling reinforcement.
Maximum 200 FSD states; a limit is not success.

BUCKLING REINFORCEMENT
Original energy heuristic, all face/core thicknesses grow-only from FSD.
Fresh static prestress is UNFILTERED for geometric stiffness.
Compute first five positive linear buckling modes (search range 1e-6..1000).
Use first-mode element strain energy, normalized separately for skins and webs.
Skin weight: 0.1 + normalized_energy^0.05.
Spar/rib weight: normalized_energy^0.5.
r=min(50000,1/BLF). Usual core multiplier: [1+0.8*(r-1)*weight]^0.7.
First five corrections with BLF<0.2 use uniform multiplier r^0.35.
Core cap: min(0.10 m, 0.15 m - 2*face thickness).
If filtered peak >300 MPa, grow faces by max(1,0.5*(1+sigma_filtered/285e6)),
limited by the remaining total-thickness allowance. No layer may shrink.

EARLY RESTORATION
Switch the NEXT correction to uniform core restoration when, over three
successive corrections (four states), all states have 0.98<=BLF<1 and
filtered peak <=300 MPa, each absolute BLF change <0.002, and each absolute
relative mass change <0.001. Once switched, restoration stays active.
Uniform core multiplier: max(1.02,min(1.25,sqrt(1.02/BLF))) when BLF<1.
Original 40-energy-correction fallback retained; at most 20 restoration-stage
corrections. Stress repair may still be needed after buckling reaches one.
Success requires a solved state with BLF>=1 AND filtered peak <=300 MPa.
Iteration limits or stalled bounds are not success.

SOLVER / SCOPE
SHELL181 full integration, KEYOPT,type,3,2 for all shell types; SMP four cores.
Positive aerodynamic load case; conservative transfer of forces and moments.
Mesh choice is independent of sizing-rule defaults: CLI mesh default remains
coarse. Use --mesh-level extra-coarse for the accepted 3276-element example.
Feasibility is evaluated on the sizing mesh using FILTERED stress; raw stress
may exceed 300 MPa. No automatic finer-mesh feasibility claim.

VERIFIED REFERENCE
runs/layout_extra_coarse_direct_massstop_20261015
20 equally spaced ribs; spar locations 0.18/0.70 chord.
11 FSD states, 17 buckling-stage states; final mass382.339182 kg,
filtered296.491097 MPa, raw319.298069 MPa, BLF1.015695227.
Fresh run example (choose a NEW output folder):
.venv/Scripts/python.exe run_layout_candidate.py --output runs/NEW_CASE --ribs 20 --spars 0.18 0.70 --mesh-level extra-coarse

The reference above is historical and its native run directory is not included in this bundle.

```

## Saved aerodynamic load

The saved Fokker force field uses speed 128 m/s, density 0.660 kg/m³, dynamic pressure 5406.72 Pa, angle of attack 12.721347 degrees and CL 1.237802. Nominal load factor is 2.5 (approximately 2.49 from the resulting force). No additional 2.5 or 1.5 multiplier is applied in structural transfer. Source semi-wing vertical force is 244274.369269 N. Layout changes conserve the same source forces and moments.
