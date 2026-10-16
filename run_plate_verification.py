from runtime_config import ansys_executable
"""Verify SHELL181 buckling against a simply supported compressed square plate.

Matches manuscript Table 1: side 2 m, thickness 1 mm, E=200 GPa,
nu=0.3, uniform edge load 1000 N/m. This verifies the analysis, not
the efficiency of a reinforcement heuristic. Historical inputs are untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time


def apdl(n):
    # One longitudinal restraint on the unloaded edge and one lateral anchor
    # permit free Poisson contraction. Edge rotations remain unconstrained.
    code = f'''FINISH
/CLEAR,START
/FILNAME,plate,1
/PREP7
ET,1,SHELL181
KEYOPT,1,3,2
MP,EX,1,2e11
MP,PRXY,1,0.3
SECTYPE,1,SHELL
SECDATA,0.001,1,0,5
SECOFFSET,MID
TYPE,1
MAT,1
SECNUM,1
'''
    for j in range(n+1):
        for i in range(n+1):
            node = j*(n+1)+i+1
            code += f'N,{node},{2*i/n:.16g},{2*j/n:.16g},0\n'
    for j in range(n):
        for i in range(n):
            node = j*(n+1)+i+1
            code += f'E,{node},{node+1},{node+n+2},{node+n+1}\n'
    for j in range(n+1):
        for i in range(n+1):
            node = j*(n+1)+i+1
            if i in (0, n) or j in (0, n):
                code += f'D,{node},UZ,0\n'
            if j == 0:
                code += f'D,{node},UY,0\n'
            if j == n:
                force = -1000*2/n*(.5 if i in (0, n) else 1)
                code += f'F,{node},FY,{force:.16g}\n'
    code += f'''D,1,UX,0
ALLSEL,ALL
FINISH
/SOLU
ANTYPE,STATIC
PSTRES,ON
OUTRES,ALL,ALL
SOLVE
FINISH
/POST1
SET,LAST
ETABLE,SY_AVG,S,Y
*DIM,syvals,ARRAY,{n*n}
*VGET,syvals(1),ELEM,1,ETAB,SY_AVG
*VSCFUN,symin,MIN,syvals(1)
*VSCFUN,symax,MAX,syvals(1)
reaction=0
*DO,k,1,{n+1}
*GET,rf,NODE,k,RF,FY
reaction=reaction+rf
*ENDDO
*CFOPEN,static_check,txt
*VWRITE,symin,symax,reaction
(3E24.16)
*CFCLOS
FINISH
/SOLU
ANTYPE,BUCKLE
BUCOPT,LANB,5,1e-6,1000,RANGE
MXPAND,5,,,YES
SOLVE
FINISH
/POST1
*CFOPEN,eigenvalues,txt
*DO,k,1,5
*GET,eigenval,MODE,k,FREQ
*VWRITE,k,eigenval
(2E24.16)
*ENDDO
*CFCLOS
SET,1,1
*CFOPEN,mode1,txt
*DO,k,1,{(n+1)**2}
*GET,xval,NODE,k,LOC,X
*GET,yval,NODE,k,LOC,Y
*GET,wval,NODE,k,U,Z
*VWRITE,k,xval,yval,wval
(4E24.16)
*ENDDO
*CFCLOS
FINISH
SAVE,verified_plate,db
/EXIT,NOSAVE
'''
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--meshes', nargs='+', type=int, default=[10, 20, 40])
    parser.add_argument('--nproc', type=int, default=4)
    args = parser.parse_args()
    if args.nproc < 1 or any(n < 4 for n in args.meshes):
        parser.error('Use positive thread count and at least four elements per edge')
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/Path(__file__).name)
    ref = 4*math.pi**2*(2e11*.001**3/(12*(1-.3**2)))/2**2/1000
    report = {'parameters': dict(side_m=2, thickness_m=.001, E_Pa=2e11, nu=.3,
                edge_load_N_per_m=1000, total_force_N=2000),
              'analytical_buckling_factor': ref,
              'analytical_reference': 'https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-buckleplate.htm',
              'ansys_parallel_mode': 'shared-memory', 'nproc': args.nproc,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Linear plate buckling verification only; no optimizer comparison', 'meshes': []}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    executable = ansys_executable()
    for n in args.meshes:
        case = out/f'mesh_{n}'
        case.mkdir()
        (case/'plate.apdl').write_text(apdl(n), encoding='ascii')
        started = time.perf_counter()
        with (case/'launcher.log').open('w') as log:
            subprocess.run([executable, '-b', '-smp', '-np', str(args.nproc),
                '-i', 'plate.apdl', '-o', 'plate.out'], cwd=case, stdout=log,
                stderr=subprocess.STDOUT, timeout=600, check=True,
                startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
        eig = [float(s.split()[1]) for s in (case/'eigenvalues.txt').read_text().splitlines()]
        static = list(map(float, (case/'static_check.txt').read_text().split()))
        shape = [list(map(float, s.split())) for s in (case/'mode1.txt').read_text().splitlines()]
        exact = [math.sin(math.pi*r[1]/2)*math.sin(math.pi*r[2]/2) for r in shape]
        correlation = abs(sum(r[3]*v for r,v in zip(shape,exact)))/math.sqrt(
            sum(r[3]**2 for r in shape)*sum(v*v for v in exact))
        text = (case/'plate.out').read_text(errors='replace')
        messages = {kind.lower(): int(re.findall(rf'NUMBER OF {kind}\s+MESSAGES ENCOUNTERED=\s*(\d+)',text)[-1])
                    for kind in ('WARNING','ERROR')}
        row = dict(elements_per_edge=n, elements=n*n, nodes=(n+1)**2,
                   buckling_factors=eig, error_percent=100*(eig[0]/ref-1),
                   mode1_correlation=correlation, static_stress_y_min_max_Pa=static[:2],
                   reaction_y_N=static[2], elapsed_seconds=time.perf_counter()-started,
                   solver_messages=messages)
        report['meshes'].append(row)
        (out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row), flush=True)
        assert messages['error'] == 0 and all(math.isfinite(v) and v > 0 for v in eig)
        assert abs(static[2]-2000) < .01 and all(abs(s+1e6)<100 for s in static[:2])
        assert correlation > .999
    finest = max(report['meshes'], key=lambda r:r['elements_per_edge'])
    report['passed'] = abs(finest['error_percent']) < 1
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
