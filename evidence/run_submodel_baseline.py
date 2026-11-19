"""Isolated original-mesh submodel check. No refinement or article changes."""
from pathlib import Path
import argparse, json, subprocess, time, re, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16'
FINAL=PARENT/'buckling/iteration_032'
OUT=ROOT/'runs/skin_front_spar_submodel_baseline_20261117'
ANSYS=Path(r'C:\Program Files\ANSYS Inc\v221\ansys\bin\winx64\ANSYS221.exe')

def execute(folder,code):
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'input.apdl').write_text(code,encoding='ascii')
    si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    start=time.perf_counter()
    with (folder/'launcher.log').open('w') as log:
        p=subprocess.run([str(ANSYS),'-b','-smp','-np','4','-j','baseline','-i','input.apdl','-o','output.out'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW,timeout=900)
    txt=(folder/'output.out').read_text(errors='replace') if (folder/'output.out').exists() else ''
    counts=re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)',txt)
    status=dict(exit_code=p.returncode,seconds=time.perf_counter()-start,error_counts=counts)
    (folder/'execution.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    if p.returncode or not counts or int(counts[-1]):raise RuntimeError(f'ANSYS did not complete cleanly: {folder}')

def inspect():
    db=(FINAL/'diagnostic').as_posix()
    folder=OUT/('inspect_archive_retry' if (OUT/'inspect_archive').exists() else 'inspect_archive')
    execute(folder,f"/CLEAR\nRESUME,'{db}',db\n/POST1\nFILE,'{db}',rst\nSET,LIST\nFINISH\n/EXIT,NOSAVE\n")

def exports(nn,ne):
    code=f'/POST1\nSET,LAST\nALLSEL,ALL\nRSYS,0\n*DIM,all_dof,ARRAY,{nn},6\n'
    for col,(item,comp) in enumerate([(a,b) for a in ('U','ROT') for b in ('X','Y','Z')],1):
        code+=f'*VGET,all_dof(1,{col}),NODE,1,{item},{comp}\n'
    code+='*CFOPEN,dof,txt\n*VWRITE,'+','.join(f'all_dof(1,{i})' for i in range(1,7))+'\n(6E24.16)\n*CFCLOS\n'
    code+=f'*DIM,face_values,ARRAY,{ne},6\n*DIM,raw_values,ARRAY,{ne}\n'
    for col,(layer,surf) in enumerate([(a,b) for a in (1,3) for b in ('BOT','MID','TOP')],1):
        code+=f'LAYER,{layer}\nSHELL,{surf}\nETABLE,ERAS\nETABLE,FACE_EQV,S,EQV\n*VGET,face_values(1,{col}),ELEM,1,ETAB,FACE_EQV\n*VOPER,raw_values(1),raw_values(1),MAX,face_values(1,{col})\n'
    code+='*CFOPEN,stress,txt\n*VWRITE,raw_values(1)\n(E24.16)\n*CFCLOS\n'
    code+='*CFOPEN,face_values,txt\n*VWRITE,'+','.join(f'face_values(1,{i})' for i in range(1,7))+'\n(6E24.16)\n*CFCLOS\n'
    code+=f'LAYER,0\nSHELL,TOP\nETABLE,ERAS\n*DIM,resultants,ARRAY,{ne},8\n'
    for i in range(1,9):code+=f'ETABLE,R{i},SMISC,{i}\n*VGET,resultants(1,{i}),ELEM,1,ETAB,R{i}\n'
    # Two groups of four avoid the APDL formatted-output line-length limit.
    for start in (1,5):
        code+=f'*CFOPEN,resultants_{start},txt\n*VWRITE,'+','.join(f'resultants(1,{i})' for i in range(start,start+4))+'\n(4E24.16)\n*CFCLOS\n'
    code+='FINISH\nSAVE,baseline,db\n/EXIT,NOSAVE\n'
    return code

def parent():
    original=(FINAL/'diagnostic.apdl').read_text()
    pre=original.split('/SOLU',1)[0].replace('/FILNAME,diagnostic,1','/FILNAME,baseline,1')
    code=pre+'/SOLU\nANTYPE,STATIC\nPSTRES,ON\nSOLVE\nFINISH\n'+exports(41126,42216)
    assert code.count('\nSOLVE\n')==1 and 'BUCOPT' not in code
    execute(OUT/'parent_static_replay',code)
    r=np.loadtxt(OUT/'parent_static_replay/stress.txt');old=np.load(FINAL/'fields.npz')['raw_Pa']
    u=np.loadtxt(OUT/'parent_static_replay/dof.txt');old_u=np.loadtxt(FINAL/'static_u.txt')
    metrics=dict(stress_max_absolute_error_Pa=float(abs(r-old).max()),stress_max_error_over_saved_peak=float(abs(r-old).max()/old.max()),
        displacement_max_absolute_error_m=float(abs(u[:,:3]-old_u).max()),displacement_max_error_over_saved_peak=float(abs(u[:,:3]-old_u).max()/abs(old_u).max()))
    assert metrics['stress_max_error_over_saved_peak']<1e-5 and metrics['displacement_max_error_over_saved_peak']<1e-6,metrics
    (OUT/'parent_reproduction.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)

def submodel():
    assert (OUT/'parent_reproduction.json').exists()
    mesh=np.load(PARENT/'geometry/mesh.npz');corners=mesh['corners'];xyz=mesh['xyz'];p=xyz[corners-1]
    cuts=(2.466441,5.426171)
    selected=(p[:,:,1].min(1)>=cuts[0]-1e-8)&(p[:,:,1].max(1)<=cuts[1]+1e-8)
    ids=np.flatnonzero(selected)+1;nodes=np.unique(corners[selected]);nmap={int(n):i+1 for i,n in enumerate(nodes)}
    boundary=nodes[np.isclose(xyz[nodes-1,1],cuts[0],atol=1e-8,rtol=0)|np.isclose(xyz[nodes-1,1],cuts[1],atol=1e-8,rtol=0)]
    assert len(ids)==7808 and len(nodes)==7701
    # Every selected-to-omitted connectivity interface must lie on a prescribed cut.
    interface=np.intersect1d(nodes,np.unique(corners[~selected]));np.testing.assert_array_equal(interface,boundary)
    original=(FINAL/'diagnostic.apdl').read_text();pre=original.split('/SOLU',1)[0]
    blocks={int(a):b for a,b in re.findall(r'SECTYPE,(\d+),SHELL\n(.*?\nEN,[^\n]+\n)',pre,re.S)}
    assert len(blocks)==42216
    lines=pre.splitlines();head=lines[:next(i for i,s in enumerate(lines) if s.startswith('N,'))]
    head=[s.replace('/FILNAME,diagnostic,1','/FILNAME,baseline,1') for s in head]
    head += [f'N,{nmap[int(n)]},'+','.join(f'{v:.16e}' for v in xyz[n-1]) for n in nodes]
    head += [s for s in lines if s.startswith(('ET,','KEYOPT,'))]
    offsets={}
    for new,eid in enumerate(ids,1):
        block=blocks[int(eid)].splitlines();head.append(f'SECTYPE,{new},SHELL')
        for s in block:
            if s.startswith('SECNUM,'):s=f'SECNUM,{new}'
            elif s.startswith('EN,'):s=f'EN,{new},'+','.join(str(nmap[int(n)]) for n in corners[eid-1])
            elif s.startswith('SECOFFSET,'):offsets[s.split(',')[1]]=offsets.get(s.split(',')[1],0)+1
            head.append(s)
    dof=np.loadtxt(OUT/'parent_static_replay/dof.txt')
    for n in boundary:
        for j,label in enumerate(('UX','UY','UZ','ROTX','ROTY','ROTZ')):head.append(f'D,{nmap[int(n)]},{label},{dof[n-1,j]:.16e}')
    retained_loads=[]
    for s in lines:
        if s.startswith('F,'):
            v=s.split(',');n=int(v[1])
            if n in nmap:
                retained_loads.append((n,v[2],float(v[3])));head.append(f'F,{nmap[n]},{v[2]},{v[3]}')
    head+=['FINISH','/SOLU','ANTYPE,STATIC','PSTRES,ON','SOLVE','FINISH']
    code='\n'.join(head)+'\n'+exports(len(nodes),len(ids))
    assert code.count('\nSOLVE\n')==1 and 'BUCOPT' not in code
    np.savez_compressed(OUT/'mapping.npz',parent_element_ids=ids,parent_node_ids=nodes,boundary_parent_nodes=boundary,corners=np.array([[nmap[int(n)] for n in co] for co in corners[selected]]),xyz=xyz[nodes-1],groups=mesh['groups'][selected])
    plan=dict(elements=len(ids),nodes=len(nodes),boundary_nodes=len(boundary),prescribed_dofs=len(boundary)*6,cuts_y_m=cuts,
        offsets=offsets,load_nodes=len(retained_loads),retained_Fz_N=sum(v[2] for v in retained_loads),
        transfer='Direct six-DOF copying at identical nodes; no interpolation',refinement='None',
        comparison_gates=dict(raw_stress_max_error_over_parent_peak=.001,translation_max_error_over_parent_peak=.0001,rotation_max_error_over_parent_peak=.0001))
    (OUT/'BASELINE_PLAN.json').write_text(json.dumps(plan,indent=2))
    execute(OUT/'submodel_original_mesh',code)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','parent','submodel']);a=parser.parse_args()
    if a.action=='inspect':inspect()
    elif a.action=='parent':parent()
    elif a.action=='submodel':submodel()
