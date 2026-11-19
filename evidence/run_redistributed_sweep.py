"""Fixed distributed vertical surface-load field, integrated on four saved meshes."""
from pathlib import Path
import sys,json,hashlib,re,argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from run_submodel_baseline import execute
R=Path(__file__).resolve().parents[2]
OUT=R/'runs/submodel_distributed_load_sweep_20261118'
B=R/'runs/skin_front_spar_submodel_baseline_20261117'
DIRS=[B]+[R/f'runs/skin_front_spar_submodel_refinement{k}_20261117' for k in (1,2,3)]
SOLVES=[B/'submodel_original_mesh']+[p/'static' for p in DIRS[1:]]
PATHS=[B/'mapping.npz']+[p/'mesh.npz' for p in DIRS[1:]]
SOURCE=R/'runs/fokker_aero_reproduction_20261013/0_12.72.txt'
ARTICLE=R/'article_results/layout_sizing_review3_mdpi_draft_20261020/layout_sizing_draft.tex'

def shapes(u,v):
    return np.array([(1-u)*(1-v),(1+u)*(1-v),(1+u)*(1+v),(1-u)*(1+v)]).T/4

def quad(m):
    ids=np.flatnonzero(m['groups']==0);co=m['corners'][ids]-1;p=m['xyz'][co]
    return ids,co,p

def samples(p):
    a,w=np.polynomial.legendre.leggauss(3)
    for i,u in enumerate(a):
        for j,v in enumerate(a):
            n=shapes(u,v);du=np.array([-(1-v),1-v,1+v,-(1+v)])/4;dv=np.array([-(1-u),-(1+u),1+u,1-u])/4
            xu=np.einsum('j,ijk->ik',du,p);xv=np.einsum('j,ijk->ik',dv,p)
            det=xu[:,0]*xv[:,1]-xu[:,1]*xv[:,0]
            assert np.all(abs(det)>1e-15)
            yield n,np.einsum('j,ijk->ik',n,p),abs(det)*w[i]*w[j]

def prepare():
    OUT.mkdir(exist_ok=False)
    meshes=[np.load(p) for p in PATHS];bm=meshes[0];ids,co,p=quad(bm)
    nn=len(bm['xyz']);upper=np.unique(co);xyz=bm['xyz']
    raw=np.loadtxt(SOURCE).reshape(91,15,3);ys=raw[:,0,1]
    edges=(np.arange(16)/15)**2;eta=.25*edges[:-1]+.75*edges[1:]
    # Source x positions are at 3/4 of each aerodynamic panel chord.
    fit=np.linalg.lstsq(np.column_stack([np.ones(15),eta]),raw[:,:,0].T,rcond=None)[0]
    xle,chord=fit;residual=float(abs((xle[:,None]+chord[:,None]*eta)-raw[:,:,0]).max())
    assert residual<2e-4
    dy=float(np.median(np.diff(ys)));density=-raw[:,:,2]/(dy*chord[:,None]*np.diff(edges))
    interp=RegularGridInterpolator((ys,eta),density,bounds_error=True)
    uy=xyz[upper,1];local_eta=(xyz[upper,0]-np.interp(uy,ys,xle))/np.interp(uy,ys,chord)
    q0=np.zeros(nn);q0[upper]=interp(np.column_stack([np.clip(uy,ys[0],ys[-1]),np.clip(local_eta,eta[0],eta[-1])]))
    assert q0[upper].min()>0
    # Freeze a continuous Q1 density field on the baseline skin; all later meshes
    # integrate this SAME field, not samples of a different interpolation.
    A=np.zeros((3,nn));origin=xyz[upper,:2].mean(0);scale=np.ptp(xyz[upper,:2],axis=0)
    for n,x,jac in samples(p):
        phi=np.column_stack([np.ones(len(x)),(x[:,:2]-origin)/scale])
        for k in range(3):np.add.at(A[k],co.ravel(),(jac[:,None]*phi[:,k,None]*n).ravel())
    oldforces=meshes[1]['nodal_forces'][:nn];target=np.array([oldforces[:,2].sum(),xyz[:,0]@oldforces[:,2],xyz[:,1]@oldforces[:,2]])
    target_scaled=np.r_[target[0],(target[1:]-origin*target[0])/scale]
    nodal_phi=np.column_stack([np.ones(len(upper)),(xyz[upper,:2]-origin)/scale])
    def corrected(a):return q0[upper]*np.exp(nodal_phi@a)
    def fun(a):return (A[:,upper]@corrected(a)-target_scaled)/target[0]
    sol=root(fun,np.zeros(3),tol=1e-11);assert sol.success or np.linalg.norm(fun(sol.x))<1e-12
    q=np.zeros(nn);q[upper]=corrected(sol.x)
    assert np.linalg.norm(fun(sol.x))<1e-11
    np.savez_compressed(OUT/'fixed_surface_field.npz',xyz=xyz,corners=co+1,element_ids=ids+1,nodal_density_N_per_projected_m2=q,initial_density=q0)
    sources=[*PATHS,*[s/'input.apdl' for s in SOLVES],SOURCE,ARTICLE,Path(__file__)]
    hashes={str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    meta=dict(method='Continuous Q1 vertical force density on baseline upper skin, consistent 3x3 Gauss integration on each child; projected x-y area; no follower pressure.',source_reconstruction='Saved forces divided by inferred squared-chord aerodynamic panel projected areas; bilinear density interpolation at 3/4-chord source points; constant edge extrapolation; frozen baseline Q1 representation. Positive exponential nodal correction matches saved segment Fz, Mx, My.',source_x_fit_max_error_m=residual,source_span_step_m=dy,correction_coefficients=sol.x.tolist(),density_correction_factor_range=[float((q[upper]/q0[upper]).min()),float((q[upper]/q0[upper]).max())],target_Fz_Mx_My=[target[0],target[2],-target[1]],source_sha256=hashes,boundary_policy='Saved original wing cut motion, unchanged at every mesh; no globally reloaded parent solve.',levels=[])
    for k,m in enumerate(meshes):
        folder=OUT/f'mesh_{k}';folder.mkdir()
        e,c,points=quad(m);parent=e if k==0 else m['baseline_element_indices'][e]
        parentp=xyz[bm['corners'][parent]-1];parentq=q[bm['corners'][parent]-1]
        forces=np.zeros(len(m['xyz']))
        for n,x,jac in samples(points):
            uv=np.zeros((len(e),2))
            for _ in range(7):
                u,v=uv.T;N=shapes(u,v);xx=np.einsum('ij,ijk->ik',N,parentp)
                du=np.column_stack([-(1-v),1-v,1+v,-(1+v)])/4;dv=np.column_stack([-(1-u),-(1+u),1+u,1-u])/4
                xu=np.einsum('ij,ijk->ik',du,parentp);xv=np.einsum('ij,ijk->ik',dv,parentp);diff=x[:,:2]-xx[:,:2];det=xu[:,0]*xv[:,1]-xu[:,1]*xv[:,0]
                uv[:,0]+=(diff[:,0]*xv[:,1]-diff[:,1]*xv[:,0])/det;uv[:,1]+=(xu[:,0]*diff[:,1]-xu[:,1]*diff[:,0])/det
            assert np.max(abs(uv))<=1+1e-8
            N=shapes(*uv.T);np.testing.assert_allclose(np.einsum('ij,ijk->ik',N,parentp),x,atol=2e-12,rtol=0)
            qq=(N*parentq).sum(1);np.add.at(forces,c.ravel(),(jac[:,None]*qq[:,None]*n).ravel())
        resultant=np.array([forces.sum(),m['xyz'][:,1]@forces,-m['xyz'][:,0]@forces])
        np.testing.assert_allclose(resultant,meta['target_Fz_Mx_My'],rtol=1e-10,atol=1e-7)
        assert np.all(forces>=0) and np.count_nonzero(forces)==len(np.unique(c))
        oldcode=(SOLVES[k]/'input.apdl').read_text()
        lines=[line for line in oldcode.splitlines() if not line.startswith('F,')]
        ix=lines.index('/SOLU');loadlines=[f'F,{i+1},FZ,{v:.16e}' for i,v in enumerate(forces) if v]
        # Existing code is in FINISH immediately before /SOLU. Apply loads in /PREP7.
        lines[ix:ix]=['/PREP7']+loadlines+['FINISH']
        code='\n'.join(lines)+'\n';assert code.count('\nSOLVE\n')==1
        clean=lambda s:'\n'.join(t for t in s.splitlines() if not t.startswith('F,') and t not in ('/PREP7','FINISH'))
        assert clean(code)==clean(oldcode)
        (folder/'prepared.apdl').write_text(code,encoding='ascii');np.savez_compressed(folder/'loads.npz',fz_N=forces)
        check=dict(level=k,elements=len(m['corners']),nodes=len(m['xyz']),loaded_nodes=int(np.count_nonzero(forces)),max_nodal_force_N=float(forces.max()),hotspot_node465_force_N=float(forces[464]),Fz_Mx_My=resultant.tolist(),force_moment_residual=(resultant-meta['target_Fz_Mx_My']).tolist(),input_difference='Only F commands and processor transitions; original mesh, sections, DOFs and solve/export commands identical.')
        meta['levels'].append(check);print(json.dumps(check),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2))

def run():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p,h in manifest['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    for k in range(4):
        folder=OUT/f'mesh_{k}'
        print(f'START MESH {k}',flush=True);execute(folder/'static',(folder/'prepared.apdl').read_text())
        s=np.loadtxt(folder/'static/stress.txt');assert len(s)==7808*4**k and np.isfinite(s).all()
        print(json.dumps(dict(level=k,peak_MPa=float(s.max()/1e6))),flush=True)
    for p,h in manifest['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    (OUT/'source_unchanged.json').write_text(json.dumps(manifest['source_sha256'],indent=2))

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--run' in sys.argv:run()
    else:raise SystemExit('Use --prepare or --run')
