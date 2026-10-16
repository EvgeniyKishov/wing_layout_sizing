"""Thresholded multi-mode, grow-only core reinforcement; independent comparison."""
from pathlib import Path
import argparse, json, hashlib, shutil, subprocess, gzip
import numpy as np
from run_fine_fsd_buckling import SOURCE, ROOT, FINE, FIELDS, render
from run_filtered_wing_fsd import solve
from stress_filter import ConicalStressFilter
from run_fine_fsd_third_check import check_balance
from diagnose_wing_mesh import model

OLD=ROOT/'runs/fine_fsd_buckling_20261015'

def update(face,core,stress,energies,eigenvalues,areas):
    """Each mode normalized globally; energy density threshold .05, p=1."""
    assert energies.shape==(len(eigenvalues),len(core))
    assert np.isfinite(energies).all() and np.all(areas>0) and np.all(eigenvalues>0)
    density=np.maximum(energies,0)/areas[None,:]
    peak=density.max(axis=1)
    assert np.all(peak>0)
    normalized=density/peak[:,None]
    weights=np.maximum(0,(normalized-.05)/.95)
    deficit=np.maximum(0,1.02/eigenvalues-1)
    increment=np.minimum(.25,.8*np.max(deficit[:,None]*weights,axis=0))
    c=np.maximum(core,np.minimum(core*(1+increment),np.minimum(.1,.15-2*face)))
    f=face.copy()
    if stress.max()>300e6:
        f=np.maximum(face,np.minimum(face*np.maximum(1,.5*(1+stress/285e6)),(.15-c)/2))
    assert np.all(f>=face) and np.all(c>=core) and np.all(2*f+c<=.15+1e-12)
    stats=dict(selected_elements=int(np.sum(increment>0)),core_increased_elements=int(np.sum(c>core)),face_increased_elements=int(np.sum(f>face)),maximum_core_increment=float(np.max(c/core-1)),selected_by_mode=[int(np.sum((weights[k]>0)&(deficit[k]>0))) for k in range(len(deficit))])
    return f,c,stats

def compact(folder,out,previous):
    # Only successful, exported solves enter here. Python writes archives;
    # native PowerShell removes exact verified paths within this run.
    targets=[];receipts=[]
    for name in ('diagnostic.out','file.bat'):
        p=folder/name
        if not p.exists():continue
        original=hashlib.sha256(p.read_bytes()).hexdigest();z=p.with_name(p.name+'.gz')
        with p.open('rb') as src,gzip.open(z,'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
        sha=hashlib.sha256()
        with gzip.open(z,'rb') as src:
            for block in iter(lambda:src.read(4*1024*1024),b''):sha.update(block)
        assert sha.hexdigest()==original
        targets.append(str(p));receipts.append(dict(path=str(p),archive=str(z),sha256=original))
    targets.extend(str(p) for p in folder.glob('*.full'))
    if previous is not None and previous.name!='iteration_000':
        assert (previous/'fields.npz').exists()
        for ext in ('.rst','.db','.esav','.mode'):targets.extend(str(p) for p in previous.glob('*'+ext))
    receipt=out/'pending_retention.json';receipt.write_text(json.dumps(dict(root=str(out),targets=targets,archives=receipts)))
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'compact_localized_buckling.ps1'),'-Receipt',str(receipt)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-iterations',type=int,default=120)
    args=p.parse_args();out=args.output.resolve();out.mkdir(exist_ok=args.resume,parents=True)
    source=(SOURCE/'diagnostic.apdl').read_text();base=np.load(SOURCE/'fields.npz')
    mesh=np.load(FINE/'mesh.npz');g=np.load(FIELDS/'fine_fields.npz');areas=g['areas']
    filt=ConicalStressFilter(g['centers'],areas,g['groups'],.3);geometry=model(SOURCE/'diagnostic.apdl')
    if not args.resume:
        protected=[SOURCE/'fields.npz',SOURCE/'diagnostic.apdl',OLD/'history.json',OLD/'summary.json',OLD/'iteration_041/fields.npz']
        manifest=dict(source=str(SOURCE),baseline_reused_from=str(OLD/'iteration_000'),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},threshold=.05,power=1,gain=.8,maximum_core_increment=.25,buckling_target=1.02,acceptance='BLF>=1 and filtered peak<=300 MPa',core_cap_m=.1,total_cap_m=.15,energy='max(SENE,0)/element area, globally normalized per mode',modes=5,uniform_growth=False,face_restoration='Grow-only relaxed ratio to285 MPa if peak filtered stress>300 MPa',integration='SHELL181 KEYOPT(3)=2',solver='ANSYS SMP four cores',prestress='unfiltered',filter_radius_m=.3)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for name in ('run_localized_buckling.py','run_fine_fsd_buckling.py','run_filtered_wing_fsd.py','stress_filter.py','compact_localized_buckling.ps1'):shutil.copy2(ROOT/name,out/name)
        first=out/'iteration_000';first.mkdir()
        for p in (OLD/'iteration_000').iterdir():
            if p.suffix in ('.txt','.npz','.apdl'):shutil.copy2(p,first/p.name)
        d=np.load(first/'fields.npz');np.testing.assert_array_equal(d['face_m'],base['face_m']);np.testing.assert_array_equal(d['core_m'],base['core_m'])
        history=[json.loads((OLD/'history.json').read_text())[0]]
        history[0]['baseline_reused']=True
        (out/'history.json').write_text(json.dumps(history,indent=2))
    else:history=json.loads((out/'history.json').read_text())
    previous=out/f"iteration_{history[-1]['iteration']:03d}"
    d=np.load(previous/'fields.npz');face=d['face_m'];core=d['core_m'];stress=d['filtered_Pa']
    eigenvalues=np.asarray(history[-1]['eigenvalues']);energies=np.array([np.loadtxt(previous/f'energy_{k}.txt') for k in range(1,6)])
    status='iteration_limit'
    for step in range(history[-1]['iteration']+1,args.max_iterations+1):
        if eigenvalues[0]>=1 and stress.max()<=300e6:status='stress_and_buckling_pass';break
        if (out/'STOP_AFTER_CURRENT').exists():status='checkpoint_requested';break
        f,c,stats=update(face,core,stress,energies,eigenvalues,areas)
        if np.array_equal(f,face) and np.array_equal(c,core):status='stalled_no_growth';break
        folder=out/f'iteration_{step:03d}'
        raw,seconds=solve(folder,render(source,f,c))
        eig=np.loadtxt(folder/'restoration_eigenvalues.txt')[:,-1]
        assert eig.shape==(5,) and np.isfinite(eig).all() and np.all(eig>0) and np.all(np.diff(eig)>=0)
        energy=np.array([np.loadtxt(folder/f'energy_{k}.txt') for k in range(1,6)])
        assert energy.shape==(5,len(core)) and np.isfinite(energy).all()
        for k in range(1,6):
            u=np.loadtxt(folder/f'mode_{k}.txt');assert u.shape==mesh['xyz'].shape and np.isfinite(u).all()
        np.testing.assert_allclose(np.loadtxt(folder/'actual_areas.txt'),areas,rtol=1e-12,atol=1e-14)
        stress=filt.apply(raw)
        assert np.all(f>=face) and np.all(c>=core) and np.all(f>=base['face_m']) and np.all(c>=base['core_m'])
        row=dict(iteration=step,mass_kg=float(areas@(5400*f+75*c)),face_mass_kg=float(areas@(5400*f)),core_mass_kg=float(areas@(75*c)),BLF=float(eig[0]),eigenvalues=eig.tolist(),raw_peak_MPa=float(raw.max()/1e6),filtered_peak_MPa=float(stress.max()/1e6),core_max_mm=float(c.max()*1000),total_max_mm=float((2*f+c).max()*1000),seconds=seconds,reaction_balance=check_balance(folder,geometry),**stats)
        np.savez_compressed(folder/'fields.npz',face_m=f,core_m=c,total_m=2*f+c,raw_Pa=raw,filtered_Pa=stress,energies=energy)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
        compact(folder,out,previous)
        previous=folder;face,core=f,c;eigenvalues=eig;energies=energy
        if eig[0]>=1 and stress.max()<=300e6:status='stress_and_buckling_pass';break
    manifest=json.loads((out/'manifest.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest['source_sha256'].items())
    (out/'summary.json').write_text(json.dumps(dict(status=status,baseline=history[0],final=history[-1]),indent=2));print('STATUS '+status,flush=True)

if __name__=='__main__':main()
