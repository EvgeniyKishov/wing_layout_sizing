"""Monotonic sandwich reinforcement of the accepted fine filtered-FSD design."""
from pathlib import Path
import argparse, json, re, shutil, hashlib
import numpy as np
from run_filtered_wing_fsd import solve
from run_fine_fsd_third_check import SECTION, FINE, FIELDS, check_balance
from diagnose_wing_mesh import model
from stress_filter import ConicalStressFilter
from final_restoration import positive_buckling_commands

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/fine_fsd_relaxed285_active5_20261015/iteration_037'

def grow(face,core,stress,energy,types,blf,iteration,restoration=False):
    # Historical wing energy heuristic, including its first-five-steps rescue.
    effective=np.zeros_like(core)
    for mask,alpha,base in ((np.isin(types,[1,4]),.05,.1),(np.isin(types,[2,3]),.5,0)):
        maximum=energy[mask].max(initial=0)
        if maximum>0: effective[mask]=base+(np.maximum(energy[mask],0)/maximum)**alpha
        else: effective[mask]=base
    c=core.copy();f=face.copy()
    if blf<1:
        req=min(50000.,1/blf)
        factor=np.full_like(core,np.sqrt(req)) if iteration<=5 and blf<.2 else 1+(req-1)*effective*.8
        factor=np.maximum(1,np.minimum(factor**.7,50000.))
        if restoration:
            # Existing final_restoration.py fallback, with 2% target margin.
            factor=np.full_like(core,max(1.02,min(1.25,np.sqrt(1.02/blf))))
        c=np.maximum(core,np.minimum(core*factor,np.minimum(.1,.15-2*face)))
    if stress.max()>300e6:
        f=np.maximum(face,np.minimum(face*np.maximum(1,.5*(1+stress/285e6)),(.15-c)/2))
    assert np.all(f>=face) and np.all(c>=core) and np.all(2*f+c<=.15+1e-12)
    return f,c

def render(source,face,core):
    seen=[]
    def section(m):
        i=int(m[1])-1;seen.append(i)
        return f'SECTYPE,{i+1},SHELL\nSECDATA,{face[i]:.16e},1,0,0\nSECDATA,{core[i]:.16e},2,0,0\nSECDATA,{face[i]:.16e},1,0,1\nSECOFFSET,{m[2]}\n'
    code=re.sub(SECTION,section,source).replace('/EXIT,NOSAVE','')
    np.testing.assert_array_equal(seen,np.arange(len(face)))
    code+='\nrr_iter=0\nrr_ls=1\nrr_valid=1\nrr_blf=1e30\n'+positive_buckling_commands()
    code+='''
/POST1
ALLSEL,ALL
*DIM,mode_energy,ARRAY,ne
*DO,imode,1,5
SET,1,imode
*VGET,du(1,1),NODE,1,U,X
*VGET,du(1,2),NODE,1,U,Y
*VGET,du(1,3),NODE,1,U,Z
*CFOPEN,mode_%imode%,txt
*VWRITE,du(1,1),du(1,2),du(1,3)
(3E24.16)
*CFCLOS
ETABLE,ERAS
ETABLE,MODE_EN,SENE
*VGET,mode_energy(1),ELEM,1,ETAB,MODE_EN
*CFOPEN,energy_%imode%,txt
*VWRITE,mode_energy(1)
(E24.16)
*CFCLOS
*ENDDO
SAVE,diagnostic,db
FINISH
/EXIT,NOSAVE
'''
    return code

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-iterations',type=int,default=40);p.add_argument('--resume',action='store_true')
    p.add_argument('--final-restoration',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=a.resume)
    source=(SOURCE/'diagnostic.apdl').read_text();initial=np.load(SOURCE/'fields.npz')
    face=initial['face_m'].copy();core=initial['core_m'].copy()
    mesh=np.load(FINE/'mesh.npz');g=np.load(FIELDS/'fine_fields.npz')
    filt=ConicalStressFilter(g['centers'],g['areas'],g['groups'],.30)
    assert all(f'KEYOPT,{k},3,2' in source for k in (1,2,3,4))
    manifest=dict(source=str(SOURCE),source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ('fields.npz','diagnostic.apdl')},
        core_cap_m=.1,total_cap_m=.15,filter_radius_m=.3,stress_limit_MPa=300,buckling_limit=1,
        rule='Historical wing heuristic: first five corrections sqrt(1/BLF)^0.7 if BLF<0.2; otherwise [1+0.8*(1/BLF-1)*effective]^0.7; effective=0.1+normalized_energy^0.05 for skins, normalized_energy^0.5 for spars/ribs. Never shrink.',
        face_restoration='Grow-only relaxed ratio to 285 MPa when filtered peak exceeds 300 MPa; core retained.',
        integration='SHELL181 KEYOPT(3)=2',solver='ANSYS SMP 4 cores',prestress='Unfiltered',eigenvalue_search='Five positive modes, range 1e-6..1000')
    manifest['final_restoration']=a.final_restoration
    if a.final_restoration:manifest['fallback']='Existing uniform core factor max(1.02,min(1.25,sqrt(1.02/BLF))); filtered grow-only face restoration unchanged.'
    phase='restoration' if a.final_restoration else 'energy_heuristic'
    manifest_name='restoration_manifest.json' if a.resume else 'manifest.json'
    (out/manifest_name).write_text(json.dumps(manifest,indent=2))
    shutil.copy2(__file__,out/('restoration_driver.py' if a.resume else Path(__file__).name))
    history=json.loads((out/'history.json').read_text()) if a.resume else []
    if history:
        last=history[-1];d=np.load(out/f"iteration_{last['iteration']:03d}/fields.npz")
        if last['BLF']>=1 and last['filtered_peak_MPa']<=300:raise ValueError('Already feasible; no continuation required')
        face,core=grow(d['face_m'],d['core_m'],d['filtered_Pa'],d['energy'],mesh['types'],last['BLF'],last['iteration']+1,a.final_restoration)
    status='iteration_limit'
    for step in range(len(history),a.max_iterations+1):
        folder=out/f'iteration_{step:03d}'
        raw,seconds=solve(folder,render(source,face,core))
        eig=np.loadtxt(folder/'restoration_eigenvalues.txt')[:,-1]
        assert eig.shape==(5,) and np.isfinite(eig).all() and (eig>0).all() and np.all(np.diff(eig)>=0)
        energy=np.loadtxt(folder/'energy_1.txt');assert energy.shape==core.shape and np.isfinite(energy).all() and energy.sum()>0
        for k in range(1,6):
            u=np.loadtxt(folder/f'mode_{k}.txt');assert u.shape==mesh['xyz'].shape and np.isfinite(u).all()
        filtered=filt.apply(raw);areas=np.loadtxt(folder/'actual_areas.txt')
        np.testing.assert_allclose(areas,g['areas'],rtol=1e-12,atol=1e-14)
        assert np.all(face>=initial['face_m']) and np.all(core>=initial['core_m'])
        row=dict(iteration=step,mass_kg=float(areas@(5400*face+75*core)),raw_peak_MPa=float(raw.max()/1e6),filtered_peak_MPa=float(filtered.max()/1e6),BLF=float(eig[0]),eigenvalues=eig.tolist(),core_max_mm=float(core.max()*1000),total_max_mm=float((2*face+core).max()*1000),seconds=seconds,reaction_balance=check_balance(folder,model(SOURCE/'diagnostic.apdl')))
        row['phase']=phase
        np.savez_compressed(folder/'fields.npz',face_m=face,core_m=core,total_m=2*face+core,raw_Pa=raw,filtered_Pa=filtered,energy=energy)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
        if eig[0]>=1 and filtered.max()<=300e6:status='stress_and_buckling_pass';break
        f,c=grow(face,core,filtered,energy,mesh['types'],eig[0],step+1,a.final_restoration)
        if np.array_equal(f,face) and np.array_equal(c,core):status='stalled_at_bounds';break
        face,core=f,c
    (out/'summary.json').write_text(json.dumps(dict(status=status,final=history[-1],baseline=history[0]),indent=2))

if __name__=='__main__':main()
