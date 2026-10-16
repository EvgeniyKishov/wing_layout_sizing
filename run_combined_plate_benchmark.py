from runtime_config import ansys_executable
"""Three plate design classes: FSD stress sizing followed by buckling reinforcement.

Compression only; equal sandwich faces. Initial stress sizing targets 98% of
allowable. Subsequent updates retain reinforcement and repair overstress.
Results are feasible heuristic designs, not proven minimum-mass solutions.
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
from run_plate_verification import apdl as plate_apdl
from final_restoration import face_stress_commands, positive_buckling_commands
from core_reinforcement import core_growth_factor_commands


def apdl(n,method,load,max_updates,initial=None):
    if method not in ('monolithic','variable_monolithic','uniform_sandwich','variable_sandwich'):
        raise ValueError(method)
    mono = method in ('monolithic','variable_monolithic')
    code = plate_apdl(n).split('/SOLU',1)[0]
    lines=[]
    for line in code.splitlines():
        p=line.split(',')
        if p[0]=='F' and p[2]=='FY':
            line=','.join(p[:3])+f',{float(p[3])*load/1000:.16e}'
        lines.append(line)
    code='\n'.join(lines)+f'''
/PREP7
MP,EX,1,69e9
MP,PRXY,1,.33
MP,DENS,1,2700
MP,EX,2,140e6
MP,PRXY,2,.35
MP,DENS,2,75
KEYOPT,1,8,2
ne={n*n}
*DIM,face,ARRAY,ne
*DIM,core,ARRAY,ne
*VFILL,face(1),RAMP,{.0005 if mono else .00025},0
*VFILL,core(1),RAMP,{0 if mono else .000125},0
*DIM,rr_stress,ARRAY,ne
*DIM,rr_tmp,ARRAY,ne
*DIM,energy,ARRAY,ne
stress_limit=300e6
stress_stage=1
stress_updates=0
buck_updates=0
passed=0
TARGET_BLF=1.05
MAX_REQ_RATIO=50000
DAMP_FACTOR=.8
DAMP_EXP=.7
MAX_FACTOR=2
MAX_INCREASE_RATIO=2
'''
    if initial is not None:
        if initial.shape!=(n*n,2):
            raise ValueError('Frozen field shape mismatch')
        code+='stress_stage=0\n'
        for i,(f,c) in enumerate(initial,1):
            code+=f'face({i})={f:.16e}\ncore({i})={c:.16e}\n'
    code+=f'''
FINISH
*CFOPEN,history,txt
*CFCLOS
*CFOPEN,restoration_eigenvalues,txt
*CFCLOS
*DO,step,0,{max_updates}
/PREP7
total_mass=0
*DO,i,1,ne
SECTYPE,i,SHELL
SECDATA,face(i),1,0,{5 if mono else 0}
'''
    if not mono:
        code+='SECDATA,core(i),2,0,0\nSECDATA,face(i),1,0,1\n'
    code+=f'''
SECOFFSET,MID
EMODIF,i,SECNUM,i
*GET,area_i,ELEM,i,AREA
total_mass=total_mass+area_i*({2700 if mono else 5400}*face(i)+75*core(i))
*ENDDO
FINISH
/SOLU
ANTYPE,STATIC
PSTRES,ON
OUTRES,ALL,ALL
SOLVE
FINISH
*VFILL,rr_stress(1),RAMP,0,0
'''
    code+=face_stress_commands().replace('LAYER,3','LAYER,1') if mono else face_stress_commands()
    code+='''
*VSCFUN,peak_stress,MAX,rr_stress(1)
rr_iter=step
rr_ls=1
rr_valid=1
rr_blf=1e30
'''+positive_buckling_commands()+'''
/POST1
SET,1,1
ETABLE,ERAS
ETABLE,MODE_EN,SENE
*VGET,energy(1),ELEM,1,ETAB,MODE_EN
*VSCFUN,max_energy,MAX,energy(1)
*CFOPEN,history,txt,,APPEND
*VWRITE,step,total_mass,peak_stress,rr_blf,stress_updates,buck_updates
(6E24.16)
*CFCLOS
*CFOPEN,field_%step%,txt
*VWRITE,face(1),core(1),rr_stress(1),energy(1)
(4E24.16)
*CFCLOS
FINISH
*IF,rr_valid,NE,1,THEN
*EXIT
*ENDIF
*IF,peak_stress,LE,stress_limit,AND,rr_blf,GE,1,THEN
passed=1
*EXIT
*ENDIF
'''+f'*IF,step,EQ,{max_updates},THEN\n*EXIT\n*ENDIF\n'+'''
! FSD stage uses membrane stress/thickness scaling with damping.
*IF,stress_stage,EQ,1,AND,peak_stress,LE,stress_limit,THEN
stress_stage=0
*ENDIF
*IF,stress_stage,EQ,1,OR,peak_stress,GT,stress_limit,THEN
stress_updates=stress_updates+1
*DO,i,1,ne
'''
    code+=('local_stress=rr_stress(i)\n' if method.startswith('variable_') else 'local_stress=peak_stress\n')
    code+=f'''
factor=MIN(2,MAX(.5,(local_stress/(.98*stress_limit))**.7))
*IF,stress_stage,EQ,0,THEN
factor=MAX(1,factor)
*ENDIF
face(i)=MAX({.0005 if mono else .00025},MIN(face(i)*factor,(.15-core(i))/{1 if mono else 2}))
*ENDDO
*ELSE
buck_updates=buck_updates+1
min_blf=rr_blf
buck_iter=buck_updates
*DO,i,1,ne
'''
    if method=='variable_monolithic':
        code+='effective_energy=SQRT(MAX(0,energy(i)/max_energy))\n'
        code+='factor=MIN(2,1+((TARGET_BLF/min_blf)**(1./3.)-1)*effective_energy)\nface(i)=MIN(.15,face(i)*factor)\n'
    elif mono:
        code+='factor=MIN(2,(TARGET_BLF/min_blf)**(1./3.))\nface(i)=MIN(.15,face(i)*factor)\n'
    else:
        code+=('effective_energy=SQRT(MAX(0,energy(i)/max_energy))\n'
               if method=='variable_sandwich' else 'effective_energy=1\n')
        code+=core_growth_factor_commands()+'core(i)=MIN(MIN(.1,.15-2*face(i)),core(i)*factor)\n'
    code+='''
*ENDDO
*ENDIF
*ENDDO
*CFOPEN,feasibility,txt
*VWRITE,passed,step,total_mass,peak_stress,rr_blf,stress_updates,buck_updates
(7E24.16)
*CFCLOS
SAVE,final_plate,db
/EXIT,NOSAVE
'''
    # Frozen checks assess physical feasibility; sizing retains a mesh reserve.
    if initial is None:
        code=code.replace('AND,rr_blf,GE,1,THEN', 'AND,rr_blf,GE,1.02,THEN')
    return code


def solve(case,code,nproc):
    case.mkdir()
    (case/'benchmark.apdl').write_text(code,encoding='ascii')
    startup=subprocess.STARTUPINFO()
    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=subprocess.SW_HIDE
    started=time.perf_counter()
    with (case/'launcher.log').open('w') as log:
        subprocess.run([ansys_executable(),
            '-b','-smp','-np',str(nproc),'-i','benchmark.apdl','-o','benchmark.out'],cwd=case,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW)
    text=(case/'benchmark.out').read_text(errors='replace')
    messages={k.lower():int(re.findall(rf'NUMBER OF {k}\s+MESSAGES ENCOUNTERED=\s*(\d+)',text)[-1]) for k in ['WARNING','ERROR']}
    if messages['error']:
        raise ValueError(f'Solver errors: {case}')
    flag,step,mass,stress,blf,stress_steps,buck_steps=np.loadtxt(case/'feasibility.txt')
    if not np.isfinite([flag,step,mass,stress,blf]).all():
        raise ValueError('Nonfinite output')
    field=np.loadtxt(case/f'field_{int(step)}.txt')
    return dict(passed=bool(flag==1 and stress<=3e8 and blf>=1),mass_kg=float(mass),
        stress_Pa=float(stress),buckling_factor=float(blf),updates=int(step),
        stress_updates=int(stress_steps),buckling_updates=int(buck_steps),
        face_min_max_m=[float(field[:,0].min()),float(field[:,0].max())],
        core_min_max_m=[float(field[:,1].min()),float(field[:,1].max())],
        elapsed_seconds=time.perf_counter()-started,messages=messages),field[:,:2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--meshes',nargs='+',type=int,default=[20,40])
    p.add_argument('--load',type=float,default=200000)
    p.add_argument('--max-updates',type=int,default=60)
    p.add_argument('--nproc',type=int,default=4)
    p.add_argument('--methods',nargs='+',choices=['monolithic','variable_monolithic','uniform_sandwich','variable_sandwich'],
                   default=['monolithic','uniform_sandwich','variable_sandwich'])
    args=p.parse_args()
    out=args.run_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources=['run_combined_plate_benchmark.py','run_plate_verification.py','final_restoration.py','core_reinforcement.py']
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    report=dict(edge_compression_N_per_m=args.load,side_m=2,allowable_Pa=3e8,
        sizing_buckling_acceptance=1.02,buckling_growth_target=1.05,
        symmetry='Equal upper and lower face thickness at every element',nproc=args.nproc,
        scope='FSD initial sizing, heuristic buckling growth, stress restoration; first feasible design',
        sources={s:hashlib.sha256((out/s).read_bytes()).hexdigest() for s in sources},cases=[])
    for n in args.meshes:
        for method in args.methods:
            case=out/f'{method}_{n}'
            row,field=solve(case,apdl(n,method,args.load,args.max_updates),args.nproc)
            row.update(method=method,mesh=n)
            print(json.dumps(row),flush=True)
            if row['passed']:
                check,_=solve(out/f'{method}_{n}_check',apdl(n,method,args.load,0,field),args.nproc)
                row['independent_check']=check
                if not check['passed'] or not np.allclose([row['mass_kg'],row['stress_Pa'],row['buckling_factor']],
                            [check['mass_kg'],check['stress_Pa'],check['buckling_factor']],rtol=1e-6):
                    raise ValueError('Independent check disagrees')
            report['cases'].append(row)
            (out/'comparison.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
