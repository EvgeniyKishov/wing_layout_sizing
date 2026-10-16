from runtime_config import ansys_executable
"""Frozen-design quadrilateral subdivision check, with freshly transferred loads.

Every child inherits its parent's final face/core thickness and section offset.
Shared edge midpoint nodes remain shared. Supports extend only along edges
whose two endpoints were constrained (and fully constrained parent faces).
The faceted geometry, constitutive model and shell integration are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from nagryzka_ansys import parse_nodes_from_cdb, parse_elements_from_cdb, element_center
from conservative_loads import transfer
from final_restoration import face_stress_commands, positive_buckling_commands


def subdivide(nodes, elements, supports):
    nodes = dict(nodes)
    fixed = set(supports)
    edges = {}
    children = {}
    parents = {}
    next_node = max(nodes)+1
    for eid, corners in elements.items():
        if len(corners)!=4 or len(set(corners))!=4:
            raise ValueError(f'Expected nondegenerate quadrilateral {eid}')
        mids = []
        for a,b in zip(corners,corners[1:]+corners[:1]):
            key = tuple(sorted((a,b)))
            if key not in edges:
                edges[key] = next_node
                nodes[next_node] = tuple((np.array(nodes[a])+nodes[b])/2)
                if a in supports and b in supports:
                    fixed.add(next_node)
                next_node += 1
            mids.append(edges[key])
        center = next_node
        nodes[center] = tuple(np.mean([nodes[n] for n in corners],axis=0))
        if all(n in supports for n in corners):
            fixed.add(center)
        next_node += 1
        for i in range(4):
            child = len(children)+1
            children[child] = [corners[i],mids[i],center,mids[(i-1)%4]]
            parents[child] = eid
    return nodes,children,fixed,parents


def prepare(run,out,factor):
    cdb = run/'wing_mesh_config_0.cdb'
    nodes = parse_nodes_from_cdb(cdb)
    elements,types = parse_elements_from_cdb(cdb)
    supports = set()
    for line in cdb.read_text(encoding='utf-8').splitlines():
        if line.startswith('D,'):
            parts = line.split(',')
            if parts[2:]!=['ALL','0']:
                raise ValueError('Unsupported non-clamped boundary condition')
            supports.add(int(parts[1]))
    thickness = np.loadtxt(run/'independent_check/diagnostic_elements.txt')
    sections = {int(row[0]):row[2:4] for row in thickness}
    if set(sections)!=set(elements):
        raise ValueError('Final thickness table does not match source mesh')
    skin = [eid for eid in elements if types[eid]==1]
    centers = np.array([element_center(nodes,elements[eid]) for eid in skin])
    target = np.loadtxt(run/'0-0.txt',skiprows=1,ndmin=2)
    distance,index = cKDTree(centers).query(target)
    if distance.max()>2e-6 or len(set(index))!=len(index):
        raise ValueError('Upper skin centers mismatch')
    upper = {skin[i] for i in index}
    if factor==2:
        nodes,elements,supports,parents = subdivide(nodes,elements,supports)
    else:
        parents = {i:i for i in elements}
    upper_nodes = sorted({nid for eid,corners in elements.items() if parents[eid] in upper for nid in corners})
    loads,conservation = transfer(np.loadtxt(run/'0_12.72.txt'),upper_nodes,
                                np.array([nodes[n] for n in upper_nodes]))
    code = ['FINISH','/CLEAR,START','/FILNAME,mesh_check,1','/PREP7','SHPP,OFF',
            'MP,EX,1,69e9','MP,NUXY,1,.33','MP,DENS,1,2700',
            'MP,EX,2,140e6','MP,NUXY,2,.35','MP,DENS,2,75']
    for kind in range(1,5):
        code += [f'ET,{kind},SHELL181',f'KEYOPT,{kind},8,2']
    for nid,xyz in nodes.items():
        code.append(f'N,{nid},'+','.join(f'{v:.16e}' for v in xyz))
    for eid,(face,core) in sections.items():
        offset = {1:'BOT',2:'MID',3:'MID',4:'TOP'}[types[eid]]
        code += [f'SECTYPE,{eid},SHELL',f'SECDATA,{face:.16e},1,0,0',
                 f'SECDATA,{core:.16e},2,0,0',f'SECDATA,{face:.16e},1,0,1',f'SECOFFSET,{offset}']
    for eid,corners in elements.items():
        parent = parents[eid]
        code += [f'TYPE,{types[parent]}','MAT,1',f'SECNUM,{parent}',f'EN,{eid},'+','.join(map(str,corners))]
    code += [f'D,{nid},ALL,0' for nid in sorted(supports)]
    code += [f'F,{nid},FZ,{fz:.16e}' for nid,fz in zip(upper_nodes,loads) if fz!=0]
    code += ['ALLSEL,ALL',f'ne={len(elements)}','*DIM,rr_stress,ARRAY,ne','*DIM,rr_tmp,ARRAY,ne',
             '*DIM,mc_area,ARRAY,ne','*DIM,mc_density,ARRAY,ne']
    # Mass per parent area is unchanged by subdivision; ANSYS supplies child areas.
    code += [f'mc_density({eid})={2*sections[p][0]*2700+sections[p][1]*75:.16e}' for eid,p in parents.items()]
    code += ['*DO,mc_i,1,ne','*GET,mc_area(mc_i),ELEM,mc_i,AREA','*ENDDO',
             '*VOPER,mc_area(1),mc_area(1),MULT,mc_density(1)',
             '*VSCFUN,mc_mass,SUM,mc_area(1)','FINISH','/SOLU','ANTYPE,STATIC','PSTRES,ON','SOLVE','FINISH',
             face_stress_commands(),'/POST1','SET,LAST','*VSCFUN,mc_stress,MAX,rr_stress(1)',
             'NSLE,S','*GET,mc_nn,NODE,0,COUNT','*DIM,mc_uz,ARRAY,mc_nn','mc_nid=0',
             '*DO,mc_i,1,mc_nn','*GET,mc_nid,NODE,mc_nid,NXTH',
             '*GET,mc_uz(mc_i),NODE,mc_nid,U,Z','*ENDDO','*VSCFUN,mc_uzmax,MAX,mc_uz(1)',
             '*CFOPEN,static_summary,txt','*VWRITE,mc_mass,mc_stress,mc_uzmax','(3E24.16)','*CFCLOS',
             'ALLSEL,ALL','FINISH','rr_iter=0','rr_ls=1','rr_valid=1','rr_blf=1e30',
             positive_buckling_commands(),'*CFOPEN,buckling_summary,txt','*VWRITE,rr_valid,rr_blf',
             '(2E24.16)','*CFCLOS','/EXIT,NOSAVE']
    (out/'mesh_check.apdl').write_text('\n'.join(code)+'\n',encoding='ascii')
    (out/'transfer.json').write_text(json.dumps(conservation,indent=2))
    return dict(factor=factor,nodes=len(nodes),elements=len(elements),supports=len(supports),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [cdb,run/'independent_check/diagnostic_elements.txt',run/'0-0.txt',run/'0_12.72.txt']})


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--factor',type=int,choices=[1,2],required=True)
    p.add_argument('--nproc',type=int,default=4)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in ['run_wing_mesh_check.py','conservative_loads.py','final_restoration.py','nagryzka_ansys.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    report = prepare(args.run.resolve(),out,args.factor)
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    started = time.perf_counter()
    with (out/'launcher.log').open('w') as log:
        subprocess.run([ansys_executable(),
                        '-b','-smp','-np',str(args.nproc),'-i','mesh_check.apdl','-o','mesh_check.out'],
                       cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
                       startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    mass,stress,uz = np.loadtxt(out/'static_summary.txt')
    valid,blf = np.loadtxt(out/'buckling_summary.txt')
    log = (out/'mesh_check.out').read_text(errors='replace')
    errors = int(re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)',log)[-1])
    report.update(mass_kg=float(mass),face_stress_Pa=float(stress),max_uz_m=float(uz),
                  positive_buckling_factor=float(blf),solver_errors=errors,
                  elapsed_seconds=time.perf_counter()-started,nproc=args.nproc,mode='shared-memory',
                  feasible=bool(valid==1 and stress<=3e8 and blf>=1))
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors or valid!=1 or not np.isfinite([mass,stress,uz,blf]).all():
        raise ValueError('Invalid mesh check')


if __name__=='__main__':
    main()
