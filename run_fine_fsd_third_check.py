"""Accepted FSD on independent fine elements, then a frozen third-mesh static audit."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import numpy as np
from run_filtered_wing_fsd import update_total, active_variables, stopping_reason, solve
from stress_filter import ConicalStressFilter, filter_stress_batched
from run_wing_mesh_check import subdivide
from diagnose_wing_mesh import model

ROOT=Path(__file__).resolve().parent
FINE=ROOT/'runs/fokker_mesh_diagnosis_20261014/refined_fullintegration'
THIRD=ROOT/'runs/fokker_integration_check_v2_20261014/third_fullintegration'
FIELDS=ROOT/'article_results/wing_stress_filter_20261014'
SECTION=r'SECTYPE,(\d+),SHELL[^\n]*\nSECDATA,[^\n]*\nSECDATA,[^\n]*\nSECDATA,[^\n]*\nSECOFFSET,([^\n]+)\n'
MARKER='!__FSD_SECTIONS__'


def template(source, assignment):
    """Replace historical sections with a placeholder and explicit variable mapping."""
    code=source.split('rr_iter=0',1)[0]
    matches=list(re.finditer(SECTION,code))
    if not matches:raise ValueError('No original sections')
    offsets={int(m[1]):m[2] for m in matches}
    code=re.sub(SECTION,lambda m: MARKER+'\n' if m.start()==matches[0].start() else '',code)
    seen=[]
    def assign(m):
        eid=int(m[1]);seen.append(eid)
        return f'SECNUM,{assignment[eid-1]}\nEN,{eid},'
    code=re.sub(r'SECNUM,\d+\nEN,(\d+),',assign,code)
    np.testing.assert_array_equal(seen,np.arange(1,len(assignment)+1))
    code=re.sub(r'\*DIM,mc_area,ARRAY,ne.*?\*VSCFUN,mc_mass,SUM,mc_area\(1\)\n','',code,flags=re.S)
    assert code.count(MARKER)==1 and code.count('\nSOLVE\n')==1 and 'ANTYPE,BUCKLE' not in code
    assert all(f'KEYOPT,{k},3,2' in code.split('/SOLU',1)[0] for k in (1,2,3,4))
    return code,offsets


def render(code,total,offsets):
    if len(total)!=len(offsets):raise ValueError('Section length mismatch')
    sections=[]
    for i,(t,offset) in enumerate(zip(total,offsets),1):
        sections.append(f'SECTYPE,{i},SHELL\nSECDATA,{.4*t:.16e},1,0,0\nSECDATA,{.2*t:.16e},2,0,0\nSECDATA,{.4*t:.16e},1,0,1\nSECOFFSET,{offset}')
    return code.replace(MARKER,'\n'.join(sections))+'''
FINISH
/PREP7
ALLSEL,ALL
*DIM,fsd_area,ARRAY,ne
*DO,fsd_i,1,ne
*GET,fsd_area(fsd_i),ELEM,fsd_i,AREA
*ENDDO
*CFOPEN,actual_areas,txt
*VWRITE,fsd_area(1)
(E24.16)
*CFCLOS
FINISH
/EXIT,NOSAVE
'''


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def loads_resultant(m):
    nodes,_,_,_,_,loads=m
    return np.array([sum(loads.values()),sum(nodes[n][1]*f for n,f in loads.items()),-sum(nodes[n][0]*f for n,f in loads.items())])


def check_balance(folder,m):
    nodes=m[0];r=np.loadtxt(folder/'reactions.txt');xyz=np.array([nodes[int(n)] for n in r[:,0]])
    applied=loads_resultant(m)
    return dict(force_residual_N=(r[:,1:4].sum(0)+[0,0,applied[0]]).tolist(),
        moment_residual_Nm=((r[:,4:7]+np.cross(xyz,r[:,1:4])).sum(0)+[applied[1],applied[2],0]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-iterations',type=int,default=200)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.max_iterations<1:p.error('Positive iteration limit required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fine=np.load(FINE/'mesh.npz');third=np.load(THIRD/'mesh.npz')
    fg=np.load(FIELDS/'fine_fields.npz');tg=np.load(FIELDS/'third_fields.npz')
    fm=model(FINE/'diagnostic.apdl');tm=model(THIRD/'diagnostic.apdl')
    # Regenerate subdivision to prove the immediate parent relation, not just assume numbering.
    newnodes,children,fixed,parents=subdivide(fm[0],fm[1],fm[4])
    np.testing.assert_allclose(np.array(list(newnodes.values())),third['xyz'],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(np.array(list(children.values())),third['corners'])
    assert fixed==tm[4]
    parent=np.array(list(parents.values()),int)
    np.testing.assert_array_equal(third['parents'],fine['parents'][parent-1])
    np.testing.assert_array_equal(tg['groups'],fg['groups'][parent-1])
    np.testing.assert_array_equal(np.bincount(parent)[1:],np.full(len(fine['corners']),4))
    np.testing.assert_allclose(loads_resultant(fm),loads_resultant(tm),rtol=1e-9,atol=1e-6)
    for mesh,g in ((fine,fg),(third,tg)):
        np.testing.assert_allclose(mesh['xyz'][mesh['corners']-1].mean(1),g['centers'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.bincount(parent,weights=tg['areas'])[1:],fg['areas'],rtol=1e-10,atol=1e-12)
    fine_template,oldoffsets=template((FINE/'diagnostic.apdl').read_text(),np.arange(1,len(fine['corners'])+1))
    third_template,third_offsets=template((THIRD/'diagnostic.apdl').read_text(),parent)
    assert oldoffsets==third_offsets
    offsets=[oldoffsets[int(i)] for i in fine['parents']]
    np.save(out/'third_to_fine_parent.npy',parent)
    files=[FINE/'diagnostic.apdl',THIRD/'diagnostic.apdl',FINE/'mesh.npz',THIRD/'mesh.npz',FIELDS/'fine_fields.npz',FIELDS/'third_fields.npz']
    hashes={str(f):digest(f) for f in files}
    previous={str(f):digest(f) for f in (ROOT/'article_results').rglob('*') if f.is_file()}
    (out/'previous_artifacts_sha256.json').write_text(json.dumps(previous,indent=2))
    manifest=dict(optimization_elements=len(fine['corners']),verification_elements=len(third['corners']),
        independent_variables=len(fine['corners']),initial_total_m=.01,layer_fractions=[.4,.2,.4],
        update_rule='relaxed',formula='clip(0.5*t*(1+filtered_stress/285e6),0.000625,0.15)',
        target_Pa=285e6,allowable_Pa=300e6,radius_m=.30,active_stress_band_Pa=[285e6,315e6],
        active_fraction_threshold=.05,thickness_tolerance=.01,max_iterations=a.max_iterations,
        stopping_logic='filtered peak <=300 MPa AND (active fraction <5% OR proposed thickness change <=1%)',
        integration='SHELL181 KEYOPT(3)=2',solver='ANSYS SMP four cores',
        third_transfer='Four children inherit each independent fine-element section; no reoptimization',
        analyses='Static only; no buckling correction or buckling solve',source_sha256=hashes,
        fine_applied_Fz_Mx_My=loads_resultant(fm).tolist(),third_applied_Fz_Mx_My=loads_resultant(tm).tolist())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in ('run_fine_fsd_third_check.py','run_filtered_wing_fsd.py','stress_filter.py','run_wing_mesh_check.py','diagnose_wing_mesh.py','final_restoration.py'):
        shutil.copy2(ROOT/name,out/name)
    total=np.full(len(fine['corners']),.01)
    initial=render(fine_template,total,offsets)
    assert initial.count('SECTYPE,')==len(total)
    (out/'prepared_initial.apdl').write_text(initial,encoding='ascii')
    print('Verified independent fine sections, exact third-mesh subdivision, areas, loads and supports.',flush=True)
    if a.prepare_only:return
    filt=ConicalStressFilter(fg['centers'],fg['areas'],fg['groups'],.30)
    history=[];reason='iteration_limit'
    for step in range(a.max_iterations):
        folder=out/f'iteration_{step:03d}'
        raw,seconds=solve(folder,render(fine_template,total,offsets))
        areas=np.loadtxt(folder/'actual_areas.txt')
        np.testing.assert_allclose(areas,fg['areas'],rtol=1e-12,atol=1e-14)
        filtered=filt.apply(raw)
        updated=update_total(total,filtered,285e6,'relaxed')
        change=float(np.max(abs(updated/total-1)))
        active=active_variables(total,filtered)
        row=dict(iteration=step,mass_kg=float(areas@(total*2175)),raw_peak_MPa=float(raw.max()/1e6),
            filtered_peak_MPa=float(filtered.max()/1e6),next_relative_thickness_change=change,seconds=seconds,
            minimum_bound_elements=int(np.sum(total==.000625)),maximum_bound_elements=int(np.sum(total==.15)),**active)
        history.append(row)
        np.savez(folder/'fields.npz',total_m=total,face_m=.4*total,core_m=.2*total,raw_Pa=raw,filtered_Pa=filtered)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        stop=stopping_reason(change,filtered.max(),active['active_fraction'],.05)
        if stop:reason=stop;break
        if step<a.max_iterations-1:total=updated
    np.save(out/'final_total_m.npy',total)
    fine_balance=check_balance(folder,fm)
    final=dict(history[-1],stopping_reason=reason,fsd_converged=reason!='iteration_limit',reaction_balance=fine_balance)
    (out/'fine_summary.json').write_text(json.dumps(final,indent=2))
    # Only the finalized, last-solved design is mapped to the third mesh.
    frozen=out/'third_frozen_static'
    raw3,seconds3=solve(frozen,render(third_template,total,offsets))
    area3=np.loadtxt(frozen/'actual_areas.txt')
    np.testing.assert_allclose(area3,tg['areas'],rtol=1e-12,atol=1e-14)
    filtered3,counts=filter_stress_batched(tg['centers'],area3,tg['groups'],.30,raw3)
    t3=total[parent-1];mass3=float(area3@(t3*2175))
    np.testing.assert_allclose(mass3,final['mass_kg'],rtol=1e-10,atol=1e-8)
    np.savez(out/'third_fields.npz',total_m=t3,face_m=.4*t3,core_m=.2*t3,raw_Pa=raw3,
             filtered_Pa=filtered3,areas=area3,centers=tg['centers'],groups=tg['groups'],fine_parent=parent,neighbours=counts)
    def peak(s,g):
        i=int(s.argmax());return dict(element=i+1,MPa=float(s[i]/1e6),xyz=g['centers'][i].tolist(),group=int(g['groups'][i]))
    assert all(digest(Path(f))==v for f,v in hashes.items())
    assert all(digest(Path(f))==v for f,v in previous.items())
    result=dict(fine=final,third=dict(elements=len(t3),mass_kg=mass3,raw_peak=peak(raw3,tg),filtered_peak=peak(filtered3,tg),
        seconds=seconds3,reaction_balance=check_balance(frozen,tm)),
        fine_filtered_peak=peak(filtered,fg),fine_raw_peak=peak(raw,fg),
        filtered_peak_change_percent=100*(filtered3.max()/filtered.max()-1),
        third_filtered_excess_over_300_percent=100*(filtered3.max()/300e6-1),
        previous_artifacts_verified=len(previous),source_files_unchanged=True,third_reoptimized=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
