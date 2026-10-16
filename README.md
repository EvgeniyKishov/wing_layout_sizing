# Wing layout and sandwich sizing research software

Python orchestration, MATLAB discrete-vortex aerodynamics, and ANSYS MAPDL shell analysis for Fokker wing layout and thickness sizing. This is a curated research snapshot, not a general aircraft certification tool.

## Contents

- `run_layout_candidate.py`: fixed-layout filtered fully-stressed design (FSD), followed by grow-only buckling reinforcement.
- `run_layout_optimization.py`: quadratic response surfaces and face-centered CCD layout sampling.
- `run_combined_plate_benchmark.py`: monolithic and sandwich plate comparison.
- `run_plate_verification.py`: analytical plate buckling verification.
- `matlab_bridge.py`, `run_aero_batch.py`, `MATLAB/`: file-based aerodynamic interface and source.
- `examples/fokker/`: configuration, airfoil, aerodynamic profile and saved force field.
- `reference_results/`: compact saved numerical evidence; no native solver binaries.
- `tests/`: solver-free checks; MATLAB integration tests are opt-in.

Other Python modules support these entry points. Historical standalone entry points inside support modules can refer to private research-run folders not shipped here; use the commands below for new calculations.

## Setup (PowerShell, from this directory)

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
$env:ANSYS_EXE = 'C:\path\to\ansys.exe'
$env:MATLAB_EXE = 'C:\path\to\matlab.exe'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Licensed external installations are required for new aerodynamic or structural solves. Software versions are omitted by request and can be obtained from the authors. Dependency versions are intentionally unpinned; compatibility on another installation must be checked.

## Reproduction

Geometry and conservative load transfer only (no solver launched):

```powershell
.venv/Scripts/python.exe run_layout_candidate.py --output runs/geometry_check --ribs 16 --spars 0.10 0.70 --mesh-level extra-coarse --max-new-solves 0
```

Single-stage extra-coarse layout sampling (expensive; starts external solver):

```powershell
.venv/Scripts/python.exe run_layout_optimization.py --output runs/ccd_stage1 --rib-counts 2 8 16 25 --mesh-level extra-coarse --max-stages 1 --max-new-evaluations 40
```

Fine sizing of the selected 16-rib layout:

```powershell
.venv/Scripts/python.exe run_layout_candidate.py --output runs/fine_16 --ribs 16 --spars 0.10 0.70 --mesh-level fine --update-rule direct --restoration-policy stagnation --mass-tolerance 0.001
```

For the comparison layouts use 8 ribs or 25 ribs, both with spars at 0.175 and 0.70, and separate fresh output directories. The saved fine masses are 635.265 kg (8 ribs), 548.506 kg (16 ribs), and 610.661 kg (25 ribs). These are semi-wing masses after reinforcement.

Plate comparison:

```powershell
.venv/Scripts/python.exe run_combined_plate_benchmark.py --run-dir runs/plate --meshes 40 --methods variable_monolithic variable_sandwich
```

Use fresh run directories. Saved references are read-only evidence, not restart checkpoints. Native solver results can be regenerated. Retention keeps compact iteration exports and selected native checkpoints; archival uses verified compression receipts. Outputs must remain under `runs/` for the archival safeguards.

To regenerate aerodynamic loads, copy `examples/fokker/for_ansys_1.npy` and `A_prof.txt` into a fresh working directory, then invoke `run_aero_batch.py` by its absolute path from that directory. The structural example deliberately uses the shipped force file; replacing it is an explicit new experiment.

## Interpretation

Feasibility here means filtered face von Mises stress <=300 MPa and lowest positive linear buckling factor >=1 on the sizing mesh. Raw local stresses can exceed 300 MPa. Filtering regularizes the monitored sizing stress, not FE equilibrium or buckling prestress. Single-stage CCD identifies good evaluated candidates, not proven global or continuous optima. Fine resizing can change their ranking. Buckling reinforcement provides feasible designs but is not a minimum-mass guarantee.

See `METHOD.md` for accepted wing sizing rules and load provenance. The plate uses its historical benchmark settings (including a 294 MPa FSD target); it is not numerically identical to the later wing sizing setup.

## Hardware and access

Reference calculations used an Intel Core i9-13900HK, 32 GiB installed RAM, and ANSYS shared-memory execution with four solver cores. Timings are measurements on that machine, not performance guarantees. Compact public data are included here; native solver files and software-version details can be requested from the corresponding author: chertykovceva.vo@ssau.ru.

## Publication and licensing

Authors: Vladislava Chertykovtseva, Nikolay Shevchenko, Oleg Lukyanov, Evgeniy Kishov, and Evgeniy Kurkin; Samara National Research University, Samara, Russia.

Before publishing, the repository owner should confirm redistribution rights for the inherited source and inputs and select a license. No license grant has been invented for this bundle. Add the final article DOI and repository URL when available. No commercial solver or MATLAB installation is redistributed.

## Viewing saved results

Run `python plot_reference_results.py` to regenerate the fine sizing history figure without either external solver. NPZ files contain named arrays and can be read with `numpy.load(..., allow_pickle=False)`. Each fine layout includes its mesh and the final FSD and reinforced fields; JSON histories contain every solved iteration. Plate stage text exports and CCD JSON records are included separately. This compact snapshot does not include every exploratory study or all manuscript figure generators.
