"""Standalone linear-static shell T-joint; full integration, no optimization."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import re
import time
import numpy as np

ROOT=Path(__file__).resolve().parent

def model(n):
    if n not in (20,40,80):raise ValueError('supported divisions: 20,40,80')
    nodes=[];lookup={};els=[];groups=[]
    def node(x,y,z):
        key=tuple(round(v,12) for v in (x,y,z))
        if key not in lookup:lookup[key]=len(nodes)+1;nodes.append(key)
        return lookup[key]
    h=1/n
    for j in range(n):
        for i in range(n):
            x=-.5+i*h;y=j*h
            els.append([node(x,y,0),node(x+h,y,0),node(x+h,y+h,0),node(x,y+h,0)]);groups.append(0)
    for k in range(n//2):
        for j in range(n//2):
            y=.25+j*h;z=k*h
            els.append([node(0,y,z),node(0,y+h,z),node(0,y+h,z+h),node(0,y,z+h)]);groups.append(1)
    xyz=np.array(nodes);corners=np.array(els);groups=np.array(groups)
    fixed=np.flatnonzero(np.isclose(abs(xyz[:,0]),.5))+1
    top=np.flatnonzero(np.isclose(xyz[:,2],.5))+1
    loads=np.zeros((len(xyz),3))
    for i in top:
        weight=.5 if np.isclose(xyz[i-1,1],.25) or np.isclose(xyz[i-1,1],.75) else 1
        loads[i-1,2]=20000*h*weight
    return xyz,corners,groups,fixed,loads

def prepare(n,out):
    xyz,els,groups,fixed,loads=model(n);ne=len(els);nn=len(xyz)
    np.savez(out/'mesh.npz',xyz=xyz,corners=els,groups=groups,fixed=fixed,loads=loads)
    lines=['FINISH','/CLEAR,START','/FILNAME,tjoint,1','/PREP7','ET,1,SHELL181','KEYOPT,1,3,2','KEYOPT,1,8,2',
        'SHPP,ON','MP,EX,1,69e9','MP,PRXY,1,.33','MP,DENS,1,2700','SECTYPE,1,SHELL','SECDATA,.01,1,0,5',
        'SECOFFSET,MID','TYPE,1','MAT,1','SECNUM,1']
    lines += [f'N,{i},'+','.join(f'{v:.16e}' for v in p) for i,p in enumerate(xyz,1)]
    lines += [f'EN,{i},'+','.join(map(str,e)) for i,e in enumerate(els,1)]
    lines += [f'D,{i},ALL,0' for i in fixed]
    lines += [f'F,{i+1},FZ,{f:.16e}' for i,f in enumerate(loads[:,2]) if f]
    lines += ['ALLSEL,ALL','FINISH','/SOLU','ANTYPE,STATIC','NLGEOM,OFF','OUTRES,ALL,ALL','SOLVE','FINISH','/POST1','SET,LAST',
        f'*DIM,stress,ARRAY,{ne}',f'*DIM,buf_stress,ARRAY,{ne}','LAYER,1']
    for surf in ('BOT','MID','TOP'):
        lines += [f'SHELL,{surf}','ETABLE,ERAS','ETABLE,EQV,S,EQV','*VGET,buf_stress(1),ELEM,1,ETAB,EQV',
                  '*VOPER,stress(1),stress(1),MAX,buf_stress(1)']
    lines += ['*CFOPEN,raw_stress,txt','*VWRITE,stress(1)','(E24.16)','*CFCLOS',
        'ETABLE,ERAS',f'*DO,ii,1,{ne}','*GET,buf_stress(ii),ELEM,ii,AREA','*ENDDO',
        '*CFOPEN,areas,txt','*VWRITE,buf_stress(1)','(E24.16)','*CFCLOS',
        f'*DIM,disp,ARRAY,{nn},3','*VGET,disp(1,1),NODE,1,U,X','*VGET,disp(1,2),NODE,1,U,Y','*VGET,disp(1,3),NODE,1,U,Z',
        '*CFOPEN,displacements,txt','*VWRITE,disp(1,1),disp(1,2),disp(1,3)','(3E24.16)','*CFCLOS','*CFOPEN,reactions,txt']
    for i in fixed:
        lines += [f'nodeid={i}']+[f'*GET,r{k},NODE,{i},RF,{label}' for k,label in enumerate(('FX','FY','FZ','MX','MY','MZ'))]
        lines += ['*VWRITE,nodeid,r0,r1,r2,r3,r4,r5','(7E24.16)']
    lines += ['*CFCLOS','SAVE,tjoint,db','FINISH','/EXIT,NOSAVE']
    (out/'tjoint.apdl').write_text('\n'.join(lines)+'\n',encoding='ascii')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/tjoint_filter_20261014')
    args=parser.parse_args();base=args.output.resolve();base.mkdir(parents=True,exist_ok=False)
    (base/'manifest.json').write_text(json.dumps(dict(meshes=[20,40,80],element_size_m=[.05,.025,.0125],
        base='x=-0.5..0.5, y=0..1, z=0 m',web='x=0, y=0.25..0.75, z=0..0.5 m',
        joint='Shared translations and rotations at web/base intersection; ideal shell junction without fillet or weld',
        thickness_m=.01,E_Pa=69e9,nu=.33,integration='SHELL181 KEYOPT(3)=2, five thickness points, MID offset',
        constraints='All six DOFs at base edges x=+/-0.5 m',load='Uniform +Fz line load 20000 N/m on web top; total 10000 N',
        analysis='Small-displacement linear static only; no buckling, sizing or plasticity',
        filter='Postprocess equivalent-stress envelope separately on base and web',radii_m=[.075,.15,.25],
        solver='ANSYS 2022 R1, SMP four cores'),indent=2))
    shutil.copy2(__file__,base/Path(__file__).name)
    for n in (20,40,80):
        out=base/f'mesh_{n}';out.mkdir();prepare(n,out)
        si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        start=time.perf_counter()
        with (out/'launcher.log').open('w') as log:
            subprocess.run([r'C:\Program Files\ANSYS Inc\v221\ansys\bin\winx64\ANSYS221.exe','-b','-smp','-np','4','-i','tjoint.apdl','-o','tjoint.out'],
                cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300,startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW)
        text=(out/'tjoint.out').read_text(errors='replace')
        errors=int(re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)',text)[-1]);assert errors==0
        s=np.loadtxt(out/'raw_stress.txt');m=np.load(out/'mesh.npz');area=np.loadtxt(out/'areas.txt')
        assert s.shape==(len(m['corners']),) and np.isfinite(s).all() and (s>=0).all()
        assert np.allclose(area,1/n**2,rtol=1e-8)
        reactions=np.loadtxt(out/'reactions.txt');ids=reactions[:,0].astype(int)-1
        force=reactions[:,1:4].sum(0)+m['loads'].sum(0)
        moment=(np.cross(m['xyz'][ids],reactions[:,1:4])+reactions[:,4:7]).sum(0)+np.cross(m['xyz'],m['loads']).sum(0)
        assert np.max(np.abs(np.r_[force,moment]))<.01
        result=dict(n=n,elements=len(s),solver_errors=errors,peak_MPa=float(s.max()/1e6),
            reaction_residual_force_N=force.tolist(),reaction_residual_moment_Nm=moment.tolist(),seconds=time.perf_counter()-start)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

