from runtime_config import ansys_executable
"""Controlled frozen-wing mesh, loading, integration and root-restraint checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from final_restoration import face_stress_commands, positive_buckling_commands

ROOT=Path(__file__).resolve().parent
SOURCES={'coarse':ROOT/'runs/fokker_mesh_control_v2_20261013/mesh_check.apdl',
         'refined':ROOT/'runs/fokker_mesh_refined_20261013/mesh_check.apdl'}

def model(path):
    nodes={};elements={};types={};sections={};fixed=set();loads={}
    kind=sec=0
    for line in path.read_text().split('/SOLU',1)[0].splitlines():
        p=line.split(',')
        if p[0]=='N':nodes[int(p[1])]=tuple(map(float,p[2:5]))
        elif p[0]=='TYPE':kind=int(p[1])
        elif p[0]=='SECNUM':sec=int(p[1])
        elif p[0]=='EN':
            eid=int(p[1]);elements[eid]=list(map(int,p[2:]));types[eid]=kind;sections[eid]=sec
        elif p[0]=='D':
            assert p[2:]==['ALL','0'];fixed.add(int(p[1]))
        elif p[0]=='F':
            assert p[2]=='FZ';loads[int(p[1])]=float(p[3])
    assert set(nodes)==set(range(1,max(nodes)+1))
    return nodes,elements,types,sections,fixed,loads

def audit(path):
    nodes,els,types,secs,fixed,loads=model(path)
    edges=defaultdict(list);usage=defaultdict(set)
    for e,c in els.items():
        for n in c:usage[n].add(types[e])
        for a,b in zip(c,c[1:]+c[:1]):edges[tuple(sorted((a,b)))].append(e)
    root=[n for n,v in nodes.items() if abs(v[1])<1e-10]
    groups=defaultdict(list)
    for n,xyz in nodes.items():groups[tuple(round(x,9) for x in xyz)].append(n)
    xyz=np.array([nodes[n] for n in loads]);f=np.array(list(loads.values()))
    report=dict(nodes=len(nodes),elements=len(els),fixed_nodes=len(fixed),root_nodes=len(root),
        fixed_off_root=[n for n in fixed if abs(nodes[n][1])>1e-10],
        root_unfixed_by_component={str(k):sum(n not in fixed and k in usage[n] for n in root) for k in (1,2,3)},
        fixed_x_range=[min(nodes[n][0] for n in fixed),max(nodes[n][0] for n in fixed)],
        shared_skin_spar_nodes=sum(1 in v and 2 in v for v in usage.values()),
        shared_skin_rib_nodes=sum(1 in v and 3 in v for v in usage.values()),
        duplicate_coordinate_groups=[g for g in groups.values() if len(g)>1],
        edge_incidence_counts=dict(Counter(map(len,edges.values()))),
        loaded_nodes=len(loads),maximum_nodal_Fz=float(f.max()),
        Fz_Mx_My=[float(f.sum()),float(xyz[:,1]@f),float(-xyz[:,0]@f)],
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return report

def prepare(name,out):
    mesh='coarse' if name.startswith('coarse') else 'refined'
    path=SOURCES[mesh];nodes,els,types,secs,fixed,loads=model(path)
    lines=path.read_text().split('/SOLU',1)[0].replace('/FILNAME,mesh_check,1','/FILNAME,diagnostic,1').splitlines()
    if name=='refined_coarse_loads':
        lines=[s for s in lines if not s.startswith('F,')]
        oldloads=model(SOURCES['coarse'])[-1]
        lines+=['/PREP7']+[f'F,{n},FZ,{v:.16e}' for n,v in oldloads.items()]+['FINISH']
    if name.endswith('fullintegration'):
        lines+=['/PREP7']+[f'KEYOPT,{k},3,2' for k in (1,2,3,4)]+['FINISH']
    if name.startswith('refined_allroot'):
        used={n for corners in els.values() for n in corners}
        fixed={n for n,v in nodes.items() if n in used and abs(v[1])<1e-10}
        lines+=['/PREP7']+[f'D,{n},ALL,0' for n in sorted(fixed)]+['FINISH']
    code='\n'.join(lines)+'\n/SOLU\nANTYPE,STATIC\nPSTRES,ON\nSOLVE\nFINISH\n'+face_stress_commands()
    code+='''
/POST1
SET,LAST
ALLSEL,ALL
*CFOPEN,stress,txt
*VWRITE,rr_stress(1)
(E24.16)
*CFCLOS
'''
    code+=f'*DIM,du,ARRAY,{len(nodes)},3\n'
    code+='''
*VGET,du(1,1),NODE,1,U,X
*VGET,du(1,2),NODE,1,U,Y
*VGET,du(1,3),NODE,1,U,Z
*CFOPEN,static_u,txt
*VWRITE,du(1,1),du(1,2),du(1,3)
(3E24.16)
*CFCLOS
*CFOPEN,reactions,txt
'''
    for n in sorted(fixed):
        code+=f'nodeid={n}\n'
        for j,label in enumerate(('FX','FY','FZ','MX','MY','MZ')):
            code+=f'*GET,r{j},NODE,{n},RF,{label}\n'
        code+='*VWRITE,nodeid,r0,r1,r2,r3,r4,r5\n(7E24.16)\n'
    code+='*CFCLOS\nFINISH\nrr_iter=0\nrr_ls=1\nrr_valid=1\nrr_blf=1e30\n'+positive_buckling_commands()
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
    (out/'diagnostic.apdl').write_text(code,encoding='ascii')
    np.savez(out/'mesh.npz',xyz=np.array(list(nodes.values())),corners=np.array(list(els.values())),
             types=np.array(list(types.values())),parents=np.array(list(secs.values())))

def run(name,base):
    out=base/name;out.mkdir(exist_ok=False);prepare(name,out)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    start=time.perf_counter()
    with (out/'launcher.log').open('w') as log:
        subprocess.run([ansys_executable(),
            '-b','-smp','-np','4','-i','diagnostic.apdl','-o','diagnostic.out'],cwd=out,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900,
            startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    text=(out/'diagnostic.out').read_text(errors='replace')
    errors=int(re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)',text)[-1])
    assert errors==0
    stress=np.loadtxt(out/'stress.txt');m=np.load(out/'mesh.npz');u=np.loadtxt(out/'static_u.txt')
    assert stress.shape==(len(m['corners']),) and u.shape==m['xyz'].shape
    centers=m['xyz'][m['corners']-1].mean(axis=1);peak=int(stress.argmax())
    eig=np.loadtxt(out/'restoration_eigenvalues.txt')[:,-1]
    report=dict(case=name,elements=len(stress),stress_MPa=float(stress.max()/1e6),
        peak_element=peak+1,peak_type=int(m['types'][peak]),peak_centroid=centers[peak].tolist(),
        max_uz=float(u[:,2].max()),eigenvalues=eig.tolist(),
        stress_away_from_root_MPa={str(y):float(stress[centers[:,1]>=y].max()/1e6) for y in (.25,.5,1,2)},
        solver_errors=errors,seconds=time.perf_counter()-start)
    for k in range(1,6):
        mode=np.loadtxt(out/f'mode_{k}.txt');assert mode.shape==u.shape and np.isfinite(mode).all()
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['coarse_baseline','refined_baseline','refined_coarse_loads','coarse_fullintegration','refined_fullintegration','refined_allroot'])
    args=p.parse_args();base=args.output.resolve();base.mkdir(exist_ok=True)
    (base/'audit.json').write_text(json.dumps({k:audit(v) for k,v in SOURCES.items()},indent=2))
    shutil.copy2(__file__,base/Path(__file__).name)
    for name in args.cases:run(name,base)

if __name__=='__main__':main()
