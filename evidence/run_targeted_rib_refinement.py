"""One targeted conforming quad refinement with compatibility strips."""
from pathlib import Path
import json,hashlib,re,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from run_submodel_baseline import execute,exports
from run_redistributed_sweep import shapes,samples
R=Path(__file__).resolve().parents[2];BASE=R/'runs/skin_front_spar_submodel_baseline_20261117'
PREV=R/'runs/skin_front_spar_submodel_refinement3_20261117'
DIST=R/'runs/submodel_distributed_load_sweep_20261118'
OUT=R/'runs/rib_endpoint_targeted_refinement_20261118'
def prepare():
    OUT.mkdir(exist_ok=False)
    src=[PREV/'mesh.npz',DIST/'mesh_3/static/input.apdl',DIST/'fixed_surface_field.npz',DIST/'manifest.json',BASE/'mapping.npz',BASE/'submodel_original_mesh/input.apdl',R/'article_results/layout_sizing_review3_mdpi_draft_20261020/layout_sizing_draft.tex',Path(__file__)]
    hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src}
    m=np.load(PREV/'mesh.npz');xyz=m['xyz'];co=m['corners'];oldn=len(xyz)
    edges=np.sort(np.stack([co,np.roll(co,-1,axis=1)],axis=2),axis=2);ue,inv=np.unique(edges.reshape(-1,2),axis=0,return_inverse=True);ie=inv.reshape(-1,4)
    a=ie[:,:2].ravel();b=ie[:,2:].ravel();graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(ue),len(ue))).tocsr();_,lab=connected_components(graph,directed=False)
    center=xyz[co-1].mean(1);origin=xyz[464];seed=np.linalg.norm(center-origin,axis=1)<.025
    marked=np.isin(lab,np.unique(lab[ie[seed]]));split=marked[ie];cnt=split.sum(1);assert np.isin(cnt,[0,2,4]).all()
    mids=np.zeros(len(ue),int);midids=np.flatnonzero(marked);mids[midids]=np.arange(oldn+1,oldn+1+len(midids))
    points=xyz.tolist()+xyz[ue[midids]-1].mean(1).tolist();children=[];parents=[]
    for i,nodes in enumerate(co):
        a,b,c,d=map(int,nodes);ab,bc,cd,da=mids[ie[i]]
        if cnt[i]==0:cc=[[a,b,c,d]]
        elif cnt[i]==4:
            ce=len(points)+1;points.append(xyz[nodes-1].mean(0).tolist());cc=[[a,ab,ce,da],[ab,b,bc,ce],[ce,bc,c,cd],[da,ce,cd,d]]
        elif split[i,0]:cc=[[a,ab,cd,d],[ab,b,c,cd]]
        else:cc=[[a,b,bc,da],[da,bc,c,d]]
        children.extend(cc);parents.extend([i]*len(cc))
    points=np.array(points);children=np.array(children);parents=np.array(parents);bp=m['baseline_element_indices'][parents]
    assert len(children)==588721 and len(np.unique(children))==len(points)
    # Every selected old edge is split on every incident surface; opposite-edge
    # closure guarantees rectangular parametric subdivisions and exact geometry.
    assert np.all(split[:,0]==split[:,2]) and np.all(split[:,1]==split[:,3])
    oldcode=(DIST/'mesh_3/static/input.apdl').read_text();bcvals={};labels=('UX','UY','UZ','ROTX','ROTY','ROTZ')
    for line in oldcode.splitlines():
        p=line.split(',')
        if p[0]=='D':bcvals.setdefault(int(p[1]),{})[p[2]]=float(p[3])
    for eid in midids:
        a,b=ue[eid]
        if a in bcvals and b in bcvals:
            assert abs(xyz[a-1,1]-xyz[b-1,1])<1e-10
            bcvals[int(mids[eid])]={s:(bcvals[a][s]+bcvals[b][s])/2 for s in labels}
    boundary=np.flatnonzero(np.isclose(points[:,1],2.466441,atol=1e-9,rtol=0)|np.isclose(points[:,1],5.426171,atol=1e-9,rtol=0))+1
    np.testing.assert_array_equal(sorted(bcvals),boundary)
    bm=np.load(BASE/'mapping.npz');field=np.load(DIST/'fixed_surface_field.npz');q=field['nodal_density_N_per_projected_m2'];groups=m['groups'][parents]
    skin=np.flatnonzero(groups==0);sp=points[children[skin]-1];parentp=bm['xyz'][bm['corners'][bp[skin]]-1];parentq=q[bm['corners'][bp[skin]]-1];forces=np.zeros(len(points))
    for n,x,jac in samples(sp):
        uv=np.zeros((len(skin),2))
        for _ in range(7):
            u,v=uv.T;N=shapes(u,v);xx=np.einsum('ij,ijk->ik',N,parentp);du=np.column_stack([-(1-v),1-v,1+v,-(1+v)])/4;dv=np.column_stack([-(1-u),-(1+u),1+u,1-u])/4
            xu=np.einsum('ij,ijk->ik',du,parentp);xv=np.einsum('ij,ijk->ik',dv,parentp);diff=x[:,:2]-xx[:,:2];det=xu[:,0]*xv[:,1]-xu[:,1]*xv[:,0]
            uv[:,0]+=(diff[:,0]*xv[:,1]-diff[:,1]*xv[:,0])/det;uv[:,1]+=(xu[:,0]*diff[:,1]-xu[:,1]*diff[:,0])/det
        N=shapes(*uv.T);assert np.max(abs(uv))<=1+1e-8;np.testing.assert_allclose(np.einsum('ij,ijk->ik',N,parentp),x,atol=2e-12,rtol=0)
        qq=(N*parentq).sum(1);np.add.at(forces,(children[skin]-1).ravel(),(jac[:,None]*qq[:,None]*n).ravel())
    resultant=np.array([forces.sum(),points[:,1]@forces,-points[:,0]@forces]);target=json.loads((DIST/'manifest.json').read_text())['target_Fz_Mx_My'];np.testing.assert_allclose(resultant,target,atol=1e-7,rtol=1e-10)
    original=(BASE/'submodel_original_mesh/input.apdl').read_text().split('/SOLU',1)[0];blocks={int(n):b for n,b in re.findall(r'SECTYPE,(\d+),SHELL\n(.*?\nEN,[^\n]+\n)',original,re.S)};assert len(blocks)==7808
    lines=original.splitlines();head=lines[:next(i for i,s in enumerate(lines) if s.startswith('N,'))]
    head += [f'N,{i+1},'+','.join(f'{x:.16e}' for x in p) for i,p in enumerate(points)];head += [s for s in lines if s.startswith(('ET,','KEYOPT,'))];types=[]
    for i in range(1,7809):
        block=blocks[i].splitlines();types.append(int(next(s for s in block if s.startswith('TYPE,')).split(',')[1]));head.append(f'SECTYPE,{i},SHELL');head += [s for s in block if not s.startswith(('TYPE,','MAT,','SECNUM,','EN,'))]
    for i,(nodes,parent) in enumerate(zip(children,bp),1):head += [f'TYPE,{types[parent]}','MAT,1',f'SECNUM,{parent+1}',f'EN,{i},'+','.join(map(str,nodes))]
    for n,vals in sorted(bcvals.items()):head += [f'D,{n},{label},{vals[label]:.16e}' for label in labels]
    head += [f'F,{i+1},FZ,{f:.16e}' for i,f in enumerate(forces) if f]
    head += [f'*DIM,areas,ARRAY,{len(children)}',f'*DO,i,1,{len(children)}','*GET,areas(i),ELEM,i,AREA','*ENDDO','*CFOPEN,actual_areas,txt','*VWRITE,areas(1)','(E24.16)','*CFCLOS','FINISH','/SOLU','ANTYPE,STATIC','PSTRES,ON','SOLVE','FINISH']
    code='\n'.join(head)+'\n'+exports(len(points),len(children));assert code.count('\nSOLVE\n')==1
    for typ in (1,2,3,4):assert f'KEYOPT,{typ},3,2' in code
    (OUT/'prepared.apdl').write_text(code,encoding='ascii')
    np.savez_compressed(OUT/'mesh.npz',xyz=points,corners=children,groups=groups,previous_element_indices=parents,baseline_element_indices=bp,parent_wing_element_ids=m['parent_wing_element_ids'][parents],boundary_nodes=boundary,nodal_fz_N=forces,seed_previous_elements=np.flatnonzero(seed)+1,previous_split_counts=cnt)
    meta=dict(elements=len(children),nodes=len(points),seed_radius_m=.025,seed_element_count=int(seed.sum()),unchanged_previous_elements=int(np.sum(cnt==0)),bisected_previous_elements=int(np.sum(cnt==2)),quartered_previous_elements=int(np.sum(cnt==4)),force_Fz_Mx_My=resultant.tolist(),residual=(resultant-target).tolist(),source_sha256=hashes,mesh_policy='25 mm centroid-distance target around node 465; opposite-edge closure extends directional compatibility strips across connected surfaces. All quadrilaterals, no hanging nodes, no triangles, no constraint equations. Original bilinear geometry and section boundaries retained.',scope='One static solve; same fixed reconstructed load field and inherited cut motion; no article edit.')
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
def run():
    meta=json.loads((OUT/'manifest.json').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    execute(OUT/'static',(OUT/'prepared.apdl').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    (OUT/'source_unchanged.json').write_text(json.dumps(meta['source_sha256'],indent=2))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--run' in sys.argv:run()
