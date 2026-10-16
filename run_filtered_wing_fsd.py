from runtime_config import ansys_executable
"""Experimental fixed-layout FSD with scalar stress filtering and frozen-design audit.

Separate from the historical optimizer: no buckling reinforcement and no claim of
local-strength feasibility. All iterations are retained in a new run directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from stress_filter import ConicalStressFilter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/fokker_mesh_diagnosis_20261014'
FIELDS = ROOT / 'article_results/wing_stress_filter_20261014'


def update_total(total, stress, target=294e6, rule='damped'):
    """Multiplicative FSD with optional damping; fixed layer fractions."""
    total = np.asarray(total, float)
    stress = np.asarray(stress, float)
    if total.shape != stress.shape or not np.isfinite(total).all() or not np.isfinite(stress).all() or (total <= 0).any() or (stress < 0).any():
        raise ValueError('Invalid FSD arrays')
    if not np.isfinite(target) or target <= 0 or rule not in ('damped', 'direct', 'relaxed'):
        raise ValueError('Invalid target or update rule')
    if rule == 'direct':
        return np.clip(total * (stress / target), .000625, .15)
    if rule == 'relaxed':
        return np.clip(.5 * total * (1. + stress / target), .000625, .15)
    return np.clip(total * np.clip(np.sqrt(stress / target), .8, 1.25), .000625, .15)


def active_variables(total, stress):
    """User-defined active set: strictly interior thickness, stress outside band."""
    total = np.asarray(total, float)
    stress = np.asarray(stress, float)
    if total.ndim != 1 or not len(total) or total.shape != stress.shape or not np.isfinite(total).all() or not np.isfinite(stress).all():
        raise ValueError('Invalid active-set arrays')
    inside = (total > .000625) & (total < .15)
    low = inside & (stress < 285e6)
    high = inside & (stress > 315e6)
    count = int(np.sum(low | high))
    return dict(active_variables=count, active_fraction=count/len(total),
                active_below_band=int(low.sum()), active_above_band=int(high.sum()),
                interior_variables=int(inside.sum()))


def stopping_reason(change, peak_stress, active_fraction, active_threshold=None):
    """Stress feasibility AND (old thickness criterion OR enabled active criterion)."""
    if peak_stress > 300e6:
        return None
    if change <= .01:
        return 'thickness_change'
    if active_threshold is not None and active_fraction < active_threshold:
        return 'active_fraction'
    return None


def replace_sections(code, total):
    """Retain original section offsets and quadrature; replace all three layers."""
    pattern = r'(SECTYPE,(\d+),SHELL[^\n]*\n)SECDATA,[^\n]*\nSECDATA,[^\n]*\nSECDATA,[^\n]*\n'
    seen = set()
    def replacement(m):
        i = int(m[2]) - 1
        if i in seen or not 0 <= i < len(total):
            raise ValueError('Unexpected section IDs')
        seen.add(i)
        t = total[i]
        return m[1] + f'SECDATA,{.4*t:.16e},1,0,0\nSECDATA,{.2*t:.16e},2,0,0\nSECDATA,{.4*t:.16e},1,0,1\n'
    result = re.sub(pattern, replacement, code)
    if len(seen) != len(total):
        raise ValueError(f'Replaced {len(seen)} sections, expected {len(total)}')
    return result


def solve(folder, code):
    folder.mkdir(exist_ok=False)
    (folder / 'diagnostic.apdl').write_text(code, encoding='ascii')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    started = time.perf_counter()
    with (folder / 'launcher.log').open('w') as log:
        subprocess.run([ansys_executable(),
                        '-b', '-smp', '-np', '4', '-i', 'diagnostic.apdl', '-o', 'diagnostic.out'],
                       cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=900, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    output = (folder / 'diagnostic.out').read_text(errors='replace')
    counts = re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)', output)
    if not counts or int(counts[-1]) != 0:
        raise RuntimeError(f'ANSYS errors or incomplete solve: {folder}')
    stress = np.loadtxt(folder / 'stress.txt')
    if not np.isfinite(stress).all() or (stress < 0).any():
        raise ValueError('Invalid solver stresses')
    return stress, time.perf_counter() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--radius', type=float, default=.30)
    p.add_argument('--max-iterations', type=int, default=60)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--update-rule', choices=('damped', 'direct', 'relaxed'), default='damped')
    p.add_argument('--target-mpa', type=float, default=294.)
    p.add_argument('--audit-meshes', nargs='+', choices=('coarse', 'fine'), default=['coarse', 'fine'])
    p.add_argument('--active-threshold', type=float, default=None,
                   help='Stop when active fraction is strictly below this value, with stress <=300 MPa')
    args = p.parse_args()
    if not np.isfinite(args.radius) or args.radius <= 0 or args.max_iterations < 1:
        p.error('Positive radius and iteration count required')
    if not np.isfinite(args.target_mpa) or args.target_mpa <= 0:
        p.error('Positive finite sizing target required')
    if args.active_threshold is not None and not 0 < args.active_threshold <= 1:
        p.error('Active threshold must be in (0, 1]')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [SOURCE / f'{mesh}/diagnostic.apdl' for mesh in ('coarse_fullintegration', 'refined_fullintegration')]
    paths += [FIELDS / f'{mesh}_fields.npz' for mesh in ('coarse', 'fine')]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    for name in ('run_filtered_wing_fsd.py', 'stress_filter.py'):
        shutil.copy2(ROOT / name, out / name)
    manifest = dict(radius_m=args.radius, target_Pa=args.target_mpa*1e6, allowable_Pa=300e6,
                    initial_total_m=.01, min_total_m=.000625, max_total_m=.15,
                    layer_fractions=[.4, .2, .4], update_rule=args.update_rule,
                    damping_exponent=None if args.update_rule == 'relaxed' else (.5 if args.update_rule == 'damped' else 1.),
                    relaxation_weight=.5 if args.update_rule == 'relaxed' else None,
                    active_fraction_threshold=args.active_threshold, active_stress_band_Pa=[285e6,315e6],
                    stopping_logic='filtered peak <=300 MPa AND (thickness change <=1% OR enabled active fraction < threshold)',
                    move_limits=[.8, 1.25] if args.update_rule == 'damped' else None,
                    audit_meshes=args.audit_meshes,
                    thickness_tolerance=.01, max_iterations=args.max_iterations,
                    integration='SHELL181 KEYOPT(3)=2', solver='ANSYS MAPDL SMP four cores',
                    criterion='Filtered scalar face stress; raw stress retained for audit',
                    scope='Positive load, fixed layout, stress sizing only; buckling measured without correction',
                    source_sha256=hashes)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    data = np.load(paths[2])
    filt = ConicalStressFilter(data['centers'], data['areas'], data['groups'], args.radius)
    total = np.full(len(data['areas']), .01)
    source = paths[0].read_text()
    assert all(f'KEYOPT,{k},3,2' in source.split('/SOLU', 1)[0] for k in (1, 2, 3, 4))
    static = source.split('rr_iter=0', 1)[0] + '\nFINISH\n/EXIT,NOSAVE\n'
    prepared = replace_sections(static, total)
    (out / 'prepared_initial.apdl').write_text(prepared, encoding='ascii')
    if args.prepare_only:
        print('Prepared inputs only:', out, flush=True)
        return
    history = []
    converged = False
    reason = 'iteration_limit'
    for step in range(args.max_iterations):
        raw, seconds = solve(out / f'iteration_{step:03d}', replace_sections(static, total))
        filtered = filt.apply(raw)
        updated = update_total(total, filtered, target=args.target_mpa*1e6, rule=args.update_rule)
        change = float(np.max(np.abs(updated / total - 1)))
        active = active_variables(total, filtered)
        row = dict(iteration=step, mass_kg=float(data['areas'] @ (total * (2*.4*2700+.2*75))),
                   raw_peak_MPa=float(raw.max()/1e6), filtered_peak_MPa=float(filtered.max()/1e6),
                   next_relative_thickness_change=change, seconds=seconds,
                   minimum_bound_elements=int(np.sum(total <= .000625*(1+1e-9))),
                   maximum_bound_elements=int(np.sum(total >= .15*(1-1e-9))))
        row.update(active)
        history.append(row)
        np.savez(out / f'iteration_{step:03d}' / 'fields.npz', total_m=total,
                 face_m=.4*total, core_m=.2*total, raw_Pa=raw, filtered_Pa=filtered)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        stop = stopping_reason(change, filtered.max(), active['active_fraction'], args.active_threshold)
        if stop is not None:
            converged = True
            reason = stop
            break
        if step < args.max_iterations-1:
            total = updated
    np.save(out / 'final_total_m.npy', total)
    audits = []
    for label, path, fieldpath in zip(('coarse', 'fine'), paths[:2], paths[2:]):
        if label not in args.audit_meshes:
            continue
        code = replace_sections(path.read_text(), total)
        raw, seconds = solve(out / f'{label}_frozen_audit', code)
        field = np.load(fieldpath)
        auditfilter = ConicalStressFilter(field['centers'], field['areas'], field['groups'], args.radius)
        filtered = auditfilter.apply(raw)
        eig = np.loadtxt(out / f'{label}_frozen_audit/restoration_eigenvalues.txt')[:, -1]
        if not np.isfinite(eig).all() or (eig <= 0).any():
            raise ValueError('Invalid positive buckling spectrum')
        inherited = total[field['parents'].astype(int)-1]
        audits.append(dict(mesh=label, elements=len(raw), raw_peak_MPa=float(raw.max()/1e6),
                           filtered_peak_MPa=float(filtered.max()/1e6), buckling_factors=eig.tolist(),
                           mass_kg=float(field['areas'] @ (inherited*(2*.4*2700+.2*75))), seconds=seconds))
        np.savez(out / f'{label}_frozen_audit/fields.npz', total_m=inherited, raw_Pa=raw, filtered_Pa=filtered)
    unchanged = all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == h for f, h in hashes.items())
    assert unchanged
    summary = dict(fsd_converged=converged, iterations=len(history), audits=audits,
                   stopping_reason=reason, final_active_set=active,
                   source_files_unchanged=unchanged, buckling_correction_applied=False,
                   local_strength_verified=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
