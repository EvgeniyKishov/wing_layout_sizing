"""Restartable coarse-layout filtered FSD followed by original buckling correction."""
from pathlib import Path
import argparse, json, hashlib, shutil, sys, time
import numpy as np
from prepare_layout_geometry import prepare, ROOT, CONFIG, FORCES
from run_filtered_wing_fsd import solve, replace_sections, update_total, active_variables, stopping_reason
from run_fine_fsd_buckling import render, grow
from run_fine_fsd_third_check import check_balance
from diagnose_wing_mesh import model
from stress_filter import ConicalStressFilter
from run_localized_buckling import compact
from buckling_restoration_policy import restoration_decision, SETTINGS
from fsd_mass_stopping import mass_stagnation

SOURCES=['runtime_config.py','run_layout_candidate.py','run_wing_mesh_check.py','fsd_mass_stopping.py','buckling_restoration_policy.py','prepare_layout_geometry.py','geom_ansys.py','nagryzka_ansys.py','conservative_loads.py','run_filtered_wing_fsd.py','run_fine_fsd_buckling.py','run_fine_fsd_third_check.py','stress_filter.py','final_restoration.py','diagnose_wing_mesh.py','run_localized_buckling.py','compact_localized_buckling.ps1']


MESH_LEVELS={'fine':dict(num_sections=60,interpolation_density=60,spar_points_multiplier=15),
             'coarse':dict(num_sections=60,interpolation_density=60,spar_points_multiplier=15),
             'extra-coarse':dict(num_sections=30,interpolation_density=30,spar_points_multiplier=8)}

def identity(ribs,spars,restoration_policy='stagnation',mesh_level='coarse',update_rule='direct',mass_tolerance=.001):
    files=[ROOT/n for n in SOURCES]+[CONFIG,FORCES,ROOT/'examples/fokker/naca2412-il.csv']
    return dict(ribs=int(ribs),spars=list(map(float,spars)),mesh_level=mesh_level,quad_subdivision=mesh_level=='fine',mesh_settings=MESH_LEVELS[mesh_level],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                settings=dict(fsd_mass_tolerance=mass_tolerance,fsd_mass_window_changes=3,update_rule=update_rule,initial_total_m=.01,radius_m=.30,target_MPa=285,stress_limit_MPa=300,buckling_limit=1,fsd_max_states=200,energy_corrections=40,restoration_max_corrections=20,restoration_policy=restoration_policy,restoration_trigger=SETTINGS,solver='SMP4',integration='full',scope='Positive aerodynamic load only; filtered face stress and linear buckling'))


def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def evaluate(out,ribs,spars,max_new_solves=None,restoration_policy='stagnation',mesh_level='coarse',update_rule='direct',mass_tolerance=.001):
    mass_stagnation([],mass_tolerance)  # Validate before creating files or launching ANSYS.
    started=time.perf_counter()
    out=Path(out).resolve();manifest=identity(ribs,spars,restoration_policy,mesh_level,update_rule,mass_tolerance)
    if out.exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Cache/source mismatch; use a fresh run directory')
        if (out/'summary.json').exists():
            old=json.loads((out/'summary.json').read_text())
            if old['status']!='checkpoint':return old
    else:
        out.mkdir(parents=True);save(out/'manifest.json',manifest)
        archive=out/'sources';archive.mkdir()
        for name in SOURCES:shutil.copy2(ROOT/name,archive/name)
        prepare(out/'geometry',ribs,spars,mesh_settings=MESH_LEVELS[mesh_level],refine=mesh_level=='fine')
    source=(out/'geometry/template.apdl').read_text();mesh=np.load(out/'geometry/mesh.npz');m=model(out/'geometry/template.apdl')
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    total=np.full(len(mesh['types']),.01);face=.4*total;core=.2*total
    filt=None;areas=None;new=0;status='checkpoint';phase='fsd';step=0;restoration=False;switch_reason=None
    # Reconstruct the next state exclusively from committed exports/history.
    if history:
        last=history[-1];folder=out/last['folder'];d=np.load(folder/'fields.npz');face=d['face_m'];core=d['core_m']
        areas=np.loadtxt(folder/'actual_areas.txt');phase=last['phase'];step=last['iteration']+1
        if phase=='fsd':
            if last['fsd_stop']:phase='buckling';step=0
            else:
                total=update_total(2*face+core,d['filtered_Pa'],285e6,update_rule);face=.4*total;core=.2*total
        else:
            restoration,switch_reason=restoration_decision(history,restoration_policy)
            face,core=grow(face,core,d['filtered_Pa'],d['energy'],mesh['types'],last['BLF'],step,restoration)
    while True:
        if max_new_solves is not None and new>=max_new_solves:break
        if (out/'STOP_AFTER_CURRENT').exists():break
        if phase=='fsd' and step>=200:status='fsd_iteration_limit';break
        if phase=='buckling' and step>60:status='buckling_iteration_limit';break
        if phase=='buckling' and sum(r.get('rule')=='uniform_restoration' for r in history)>=SETTINGS['maximum_restoration_corrections']:status='restoration_iteration_limit';break
        folder=out/phase/f'iteration_{step:03d}';folder.parent.mkdir(exist_ok=True)
        # A partial solver folder is deliberately never silently overwritten.
        code=replace_sections(source,2*face+core) if phase=='fsd' else render(source,face,core)
        raw,seconds=solve(folder,code);new+=1
        current_areas=np.loadtxt(folder/'actual_areas.txt')
        if areas is None:areas=current_areas
        np.testing.assert_allclose(current_areas,areas,rtol=1e-12,atol=1e-14)
        assert raw.shape==areas.shape and (areas>0).all()
        if filt is None:filt=ConicalStressFilter(mesh['centers'],areas,mesh['groups'],.30)
        filtered=filt.apply(raw);balance=check_balance(folder,m)
        load_scale=max(1.,sum(abs(v) for v in m[-1].values()))
        assert np.max(np.abs(balance['force_residual_N']))<load_scale*1e-5
        assert np.max(np.abs(balance['moment_residual_Nm']))<load_scale*np.max(np.abs(mesh['xyz']))*1e-5
        assert np.isfinite(np.loadtxt(folder/'static_u.txt')).all()
        row=dict(phase=phase,iteration=step,folder=str(folder.relative_to(out)),mass_kg=float(areas@(5400*face+75*core)),raw_peak_MPa=float(raw.max()/1e6),filtered_peak_MPa=float(filtered.max()/1e6),seconds=seconds,reaction_balance=balance)
        fields=dict(face_m=face,core_m=core,total_m=2*face+core,raw_Pa=raw,filtered_Pa=filtered)
        if phase=='fsd':
            total=2*face+core;proposed=update_total(total,filtered,285e6,update_rule);change=float(np.max(np.abs(proposed/total-1)));active=active_variables(total,filtered)
            stop=stopping_reason(change,float(filtered.max()),active['active_fraction'],.05)
            mass_stop,mass_changes=mass_stagnation(history+[row],mass_tolerance)
            if stop is None and mass_stop:stop='feasible_mass_stagnation'
            row.update(active,change=change,fsd_stop=stop,mass_window_relative_changes=mass_changes,mass_stagnation=mass_stop)
        else:
            eig=np.loadtxt(folder/'restoration_eigenvalues.txt')[:,-1]
            assert eig.shape==(5,) and np.isfinite(eig).all() and np.all(eig>0) and np.all(np.diff(eig)>=0)
            energy=np.loadtxt(folder/'energy_1.txt');assert energy.shape==core.shape and np.isfinite(energy).all() and energy.sum()>0
            for k in range(1,6):
                mode=np.loadtxt(folder/f'mode_{k}.txt');assert mode.shape==mesh['xyz'].shape and np.isfinite(mode).all()
            fields['energy']=energy;row.update(BLF=float(eig[0]),eigenvalues=eig.tolist(),rule='uniform_restoration' if restoration else 'original_energy',switch_reason=switch_reason)
        np.savez_compressed(folder/'fields.npz',**fields)
        previous=out/history[-1]['folder'] if history and history[-1]['phase']==phase else None
        history.append(row);save(out/'history.json',history)
        compact(folder,out,previous)
        print(json.dumps({k:v for k,v in row.items() if k!='reaction_balance'}),flush=True)
        if phase=='fsd':
            if stop:phase='buckling';step=0;continue
            face=.4*proposed;core=.2*proposed
        else:
            if eig[0]>=1 and filtered.max()<=300e6:status='feasible';break
            restoration,switch_reason=restoration_decision(history,restoration_policy)
            f,c=grow(face,core,filtered,energy,mesh['types'],eig[0],step+1,restoration)
            if np.array_equal(f,face) and np.array_equal(c,core):status='stalled_at_bounds';break
            face,core=f,c
        step+=1
    summary=dict(status=status,feasible=status=='feasible',objective_mass_kg=history[-1]['mass_kg'] if status=='feasible' else None,final=history[-1] if history else None,ribs=ribs,spars=list(spars),completed_solves=len(history),solver_seconds=sum(r['seconds'] for r in history),invocation_wall_seconds=time.perf_counter()-started,restoration_policy=restoration_policy)
    save(out/'summary.json',summary);return summary


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--ribs',type=int,default=20);p.add_argument('--spars',nargs=2,type=float,default=[.18,.70]);p.add_argument('--max-new-solves',type=int);p.add_argument('--restoration-policy',choices=['stagnation','fixed40'],default='stagnation')
    p.add_argument('--mesh-level',choices=list(MESH_LEVELS),default='coarse')
    p.add_argument('--update-rule',choices=['relaxed','direct'],default='direct')
    p.add_argument('--mass-tolerance',type=float,default=.001,help='Relative mass change threshold for each of three consecutive FSD updates (default 0.001 = 0.1 percent)')
    a=p.parse_args();print(json.dumps(evaluate(a.output,a.ribs,a.spars,a.max_new_solves,a.restoration_policy,a.mesh_level,a.update_rule,a.mass_tolerance)))
