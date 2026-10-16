"""Coarse Fokker integer rib sweep and sequential local face-centered CCD."""
from pathlib import Path
import argparse, hashlib, json, sys, shutil, time
import numpy as np
from layout_response_surface import BOUNDS, points, fit_minimum, next_box
from run_layout_candidate import evaluate, identity, save, MESH_LEVELS
from archive_solver_storage import archive_candidate


class EvaluationBudget(Exception):pass


def search(out,rib_counts,max_evaluations=1,max_stages=8,mass_tolerance=.01,stable_stages=3,mesh_level='coarse',seed_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    seed=Path(seed_from).resolve() if seed_from is not None else None
    seed_states={}; seed_hashes={}
    if seed is not None:
        if seed==out:raise ValueError('Seed and output must differ')
        seed_settings=json.loads((seed/'settings.json').read_text())
        if seed_settings['evaluator']!=identity(20,[.18,.70],mesh_level=mesh_level):raise ValueError('Seed sizing settings/source mismatch')
        seed_files=[seed/'settings.json']
        for ribs in rib_counts:
            path=seed/f'ribs_{ribs:02d}.json'; state=json.loads(path.read_text())
            if state['status']!='stage_limit' or not state['best'] or not state['best']['feasible']:raise ValueError('Seed must have a completed limited search and feasible best')
            if len(state['stages'])>=max_stages:raise ValueError('No additional stages requested')
            seed_states[ribs]=state;seed_files.append(path)
        seed_files+=list((seed/'evaluations').glob('*/summary.json'))
        seed_hashes={str(p.relative_to(seed)):hashlib.sha256(p.read_bytes()).hexdigest() for p in seed_files}
    settings=dict(rib_counts=list(rib_counts),max_stages=max_stages,mass_tolerance=mass_tolerance,stable_stages=stable_stages,
        seed_from=str(seed) if seed else None,seed_sha256=seed_hashes,
        evaluator=identity(20,[.18,.70],mesh_level=mesh_level),search_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('run_layout_optimization.py','layout_response_surface.py','archive_solver_storage.py','apply_solver_archives.ps1')})
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Search settings/source mismatch')
    save(out/'settings.json',settings);used=0
    def actual(ribs,xy):
        nonlocal used
        layout=dict(ribs=int(ribs),spars=[float(v) for v in np.round(xy,12)])
        key=hashlib.sha256(json.dumps(layout,sort_keys=True).encode()).hexdigest()[:20]
        folder=out/'evaluations'/key
        if (folder/'summary.json').exists():
            result=json.loads((folder/'summary.json').read_text())
            if result['status']!='checkpoint':return dict(result,path=str(folder))
        if seed is not None and (seed/'evaluations'/key/'summary.json').exists():
            cached=seed/'evaluations'/key
            result=json.loads((cached/'summary.json').read_text())
            if result['status']!='checkpoint':return dict(result,path=str(cached))
        if (out/'STOP_AFTER_CURRENT').exists() or used>=max_evaluations:raise EvaluationBudget()
        free=shutil.disk_usage(out).free
        if free<10*1024**3:
            save(out/'storage_stop.json',dict(free_bytes=free,minimum_free_bytes=10*1024**3))
            raise EvaluationBudget()
        used+=1
        save(out/'progress.json',dict(status='sizing',layout=layout,folder=str(folder),new_evaluations_started=used))
        evaluation_started=time.perf_counter()
        try:result=evaluate(folder,layout['ribs'],layout['spars'],mesh_level=mesh_level)
        except Exception as error:
            result=dict(status='evaluation_error',feasible=False,objective_mass_kg=None,ribs=ribs,spars=layout['spars'],error=repr(error))
            # Preserve partial solver output; a failure is never an objective value.
            folder.mkdir(parents=True,exist_ok=True);save(folder/'summary.json',result)
        if result['status'] not in ('checkpoint','evaluation_error'):
            save(out/'progress.json',dict(status='archiving',layout=layout,folder=str(folder),result=result))
            try:archive_candidate(folder)
            except Exception as error:
                # A retention failure must not erase the verified FE result.
                save(folder/'archive_error.json',dict(error=repr(error)))
                raise RuntimeError('Storage archival failed; stop before accumulating more candidates') from error
        save(folder/'evaluation_timing.json',dict(wall_seconds_including_archival=time.perf_counter()-evaluation_started,free_disk_bytes=shutil.disk_usage(out).free))
        save(out/'progress.json',dict(status='evaluation_completed',layout=layout,folder=str(folder),result=result))
        return dict(result,path=str(folder))
    results=[]
    for ribs in rib_counts:
        path=out/f'ribs_{ribs:02d}.json'
        state=json.loads(path.read_text()) if path.exists() else dict(ribs=ribs,status='running',stages=[],best=None,stable=0,next_box=BOUNDS.tolist())
        if not path.exists() and ribs in seed_states:
            state=json.loads(json.dumps(seed_states[ribs]));state.update(status='running',stable=0)
            save(path,state)
        if state['status'] in ('converged','stage_limit','insufficient_feasible_samples'):
            results.append(state);continue
        try:
            for stage in range(len(state['stages']),max_stages):
                box=np.array(state['next_box']);old=state['best']; samples=[actual(ribs,p) for p in points(box)]
                valid=[r for r in samples if r['feasible']]
                try:candidate,fit=fit_minimum(box,[r['spars'] for r in valid],[r['objective_mass_kg'] for r in valid])
                except ValueError as error:
                    state['status']='insufficient_feasible_samples';state['failure']=str(error);state['partial_samples']=samples
                    available=valid+([old] if old else [])
                    if available:state['best']=min(available,key=lambda r:r['objective_mass_kg'])
                    save(path,state);break
                verified=actual(ribs,candidate)
                available=valid+([verified] if verified['feasible'] else [])+([old] if old else [])
                best=min(available,key=lambda r:r['objective_mass_kg'])
                improvement=None if old is None else (old['objective_mass_kg']-best['objective_mass_kg'])/old['objective_mass_kg']
                box_next,moved=next_box(box,best['spars'],improvement is None or improvement>1e-12)
                stable=(state['stable']+1 if improvement is not None and improvement<mass_tolerance and verified['feasible'] and not moved else 0)
                if verified['feasible']:fit['candidate_relative_prediction_error']=abs(verified['objective_mass_kg']-fit['predicted_mass_kg'])/verified['objective_mass_kg']
                record=dict(stage=stage,box=box.tolist(),samples=samples,candidate=verified,fit=fit,best=best,relative_actual_mass_improvement=improvement,box_translated=moved)
                state['stages'].append(record);state.update(best=best,stable=stable,next_box=box_next.tolist(),status='converged' if stable>=stable_stages else 'running')
                save(path,state);print(json.dumps(dict(ribs=ribs,stage=stage,mass_kg=best['objective_mass_kg'],status=state['status'])),flush=True)
                if state['status']=='converged':break
            if state['status']=='running':state['status']='stage_limit';save(path,state)
        except EvaluationBudget:
            state['status']='checkpoint';save(path,state);results.append(state)
            save(out/'summary.json',dict(status='checkpoint',new_evaluations=used,rib_results=results));return
        results.append(state)
    best=[r['best'] for r in results if r['best']]
    save(out/'summary.json',dict(status='sweep_finished',all_rib_searches_converged=all(r['status']=='converged' for r in results),best_verified=min(best,key=lambda r:r['objective_mass_kg']) if best else None,rib_results=results))
    if (out/'DEFER_NATIVE_ARCHIVAL').exists():
        save(out/'progress.json',dict(status='post_sweep_archiving',rib_counts=list(rib_counts)))
        for completed in sorted((out/'evaluations').glob('*/summary.json')):
            archive_candidate(completed.parent,force=True)
    save(out/'progress.json',dict(status='batch_finished',rib_counts=list(rib_counts),max_stages=max_stages))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--rib-counts',nargs='+',type=int,default=list(range(2,26)));p.add_argument('--max-new-evaluations',type=int,default=1);p.add_argument('--max-stages',type=int,default=8)
    p.add_argument('--mesh-level',choices=list(MESH_LEVELS),default='coarse')
    p.add_argument('--seed-from',type=Path)
    p.add_argument('--mass-tolerance',type=float,default=.01,help='Outer stage-to-stage relative best-mass tolerance')
    a=p.parse_args()
    if any(not 2<=n<=25 for n in a.rib_counts) or len(set(a.rib_counts))!=len(a.rib_counts) or a.max_new_evaluations<1 or a.max_stages<1:p.error('Invalid rib counts or limits')
    if not np.isfinite(a.mass_tolerance) or a.mass_tolerance<=0:p.error('Invalid mass tolerance')
    search(a.output,a.rib_counts,a.max_new_evaluations,a.max_stages,mass_tolerance=a.mass_tolerance,mesh_level=a.mesh_level,seed_from=a.seed_from)
