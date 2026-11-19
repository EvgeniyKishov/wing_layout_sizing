"""Spatial/quality audit from saved distributed-load results; no solver calls."""
from pathlib import Path
import json,csv,hashlib,re,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
R=Path(__file__).resolve().parents[2];O=R/'journal_reviews/rib_spatial_quality_audit_20261118';O.mkdir(exist_ok=True)
B=R/'runs/skin_front_spar_submodel_baseline_20261117';RUN=R/'runs/submodel_distributed_load_sweep_20261118'
paths=[B/'mapping.npz']+[R/f'runs/skin_front_spar_submodel_refinement{k}_20261117/mesh.npz' for k in (1,2,3)]
M=[np.load(p) for p in paths];S=[RUN/f'mesh_{k}/static' for k in range(4)]
article=R/'article_results/layout_sizing_review3_mdpi_draft_20261020/layout_sizing_draft.tex'
sources=paths+[p/'stress.txt' for p in S]+[article]
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
origin=M[0]['xyz'][464,[0,2]];rs=np.array([.005,.01,.02,.03,.05,.075,.1,.15,.2])
raydefs={'Into rib, horizontal':np.array([1.,0.]),'Into rib, 30 degrees down':np.array([np.cos(np.pi/6),-.5])}
raw=[np.loadtxt(p/'stress.txt')/1e6 for p in S]
pol=[];ids=[];quality=[];rays=[];patches=[];tails=[];resultants=[];near_skin=[]
skin_nodes=np.intersect1d(np.unique(M[0]['corners'][M[0]['groups']==0]),np.unique(M[0]['corners'][M[0]['groups']==7]))
skin_profile=M[0]['xyz'][skin_nodes-1][:,[0,2]];skin_profile=skin_profile[np.argsort(skin_profile[:,0])]
def area(p):return abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))/2 if len(p)>2 else 0.
def clip(p,axis,bound,keep_above):
    out=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        ia=a[axis]>=bound if keep_above else a[axis]<=bound;ib=b[axis]>=bound if keep_above else b[axis]<=bound
        if ia:out.append(a)
        if ia!=ib:out.append(a+(b-a)*(bound-a[axis])/(b[axis]-a[axis]))
    return np.array(out).reshape(-1,2)
def clipped_area(p,lo,hi):
    for axis in (0,1):
        p=clip(p,axis,lo[axis],True)
        if not len(p):return 0.
        p=clip(p,axis,hi[axis],False)
        if not len(p):return 0.
    return area(p)
def shape_quality(p):
    edges=np.roll(p,-1,axis=1)-p;lens=np.linalg.norm(edges,axis=2)
    cross=edges[:,:,0]*np.roll(edges,-1,axis=1)[:,:,1]-edges[:,:,1]*np.roll(edges,-1,axis=1)[:,:,0]
    signed_area=np.sum(p[:,:,0]*np.roll(p[:,:,1],-1,axis=1)-p[:,:,1]*np.roll(p[:,:,0],-1,axis=1),axis=1)/2
    orientation=np.sign(signed_area)
    sj=cross/(lens*np.roll(lens,-1,axis=1))*orientation[:,None]
    cos=np.sum((-np.roll(edges,1,axis=1))*edges,axis=2)/(np.roll(lens,1,axis=1)*lens)
    angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
    dets=[];conds=[]
    for u,v in [(-1,-1),(1,-1),(1,1),(-1,1),(0,0)]:
        du=np.array([-(1-v),1-v,1+v,-(1+v)])/4;dv=np.array([-(1-u),-(1+u),1+u,1-u])/4
        xu=np.einsum('j,ijk->ik',du,p);xv=np.einsum('j,ijk->ik',dv,p);J=np.stack([xu,xv],axis=2)
        dets.append(np.linalg.det(J)*orientation);sv=np.linalg.svd(J,compute_uv=False);conds.append(sv[:,0]/sv[:,1])
    dets=np.array(dets).T;conds=np.array(conds).T
    return dict(edge_ratio=lens.max(1)/lens.min(1),min_angle_deg=angles.min(1),max_angle_deg=angles.max(1),min_scaled_corner_jacobian=sj.min(1),jacobian_det_ratio=dets.min(1)/dets.max(1),max_jacobian_condition=conds.max(1),min_signed_jacobian=dets.min(1),area_m2=abs(signed_area),max_edge_m=lens.max(1))
for k,m in enumerate(M):
    ii=np.flatnonzero(m['groups']==7);p=m['xyz'][m['corners'][ii]-1][:,:,[0,2]];ids.append(ii);pol.append(p);s=raw[k][ii];q=shape_quality(p)
    peak=int(np.argmax(s));incident=np.any(m['corners'][ii]==465,axis=1);ip=np.flatnonzero(incident)[s[incident].argmax()]
    local=np.linalg.norm(p.mean(1)-origin,axis=1)<.05
    quality.append(dict(level=k,endpoint_element=int(ii[ip]+1),endpoint_stress_MPa=float(s[ip]),endpoint_centroid_distance_mm=float(1000*np.linalg.norm(p[ip].mean(0)-origin)),endpoint={key:float(v[ip]) for key,v in q.items()},within_50mm={key:dict(min=float(v[local].min()),max=float(v[local].max())) for key,v in q.items()},invalid_jacobians_rib=int(np.sum(q['min_signed_jacobian']<=0)),nonconvex_rib=int(np.sum(q['min_scaled_corner_jacobian']<=0))))
    e=np.roll(p,-1,axis=1)-p
    for name,direction in raydefs.items():
        for r in rs:
            point=origin+r*direction;rel=point[None,None,:]-p;cr=e[:,:,0]*rel[:,:,1]-e[:,:,1]*rel[:,:,0]
            found=np.flatnonzero(np.all(cr>=-1e-13,axis=1)|np.all(cr<=1e-13,axis=1));assert len(found)>0
            rays.append(dict(level=k,path=name,distance_mm=float(r*1000),x_m=float(point[0]),z_m=float(point[1]),elements=(ii[found]+1).tolist(),raw_MPa=float(s[found].mean()),minimum_MPa=float(s[found].min()),maximum_MPa=float(s[found].max()),max_element_edge_mm=float(q['max_edge_m'][found].max()*1000)))
    for r in rs:
        xx=origin[0]+r;point=np.array([xx,np.interp(xx,skin_profile[:,0],skin_profile[:,1])-.005]);rel=point[None,None,:]-p;cr=e[:,:,0]*rel[:,:,1]-e[:,:,1]*rel[:,:,0]
        found=np.flatnonzero(np.all(cr>=-1e-13,axis=1)|np.all(cr<=1e-13,axis=1));assert len(found)>0
        near_skin.append(dict(level=k,x_offset_mm=float(r*1000),x_m=float(point[0]),z_m=float(point[1]),raw_MPa=float(s[found].mean()),elements=(ii[found]+1).tolist()))
    for center in (.03,.05,.1,.2):
        lo=origin+np.array([center-.01,-.01]);hi=origin+np.array([center+.01,0.])
        candidate=np.flatnonzero(np.all(p.max(1)>=lo,axis=1)&np.all(p.min(1)<=hi,axis=1))
        aa=np.array([clipped_area(p[j],lo,hi) for j in candidate]);mask=aa>1e-16;aa=aa[mask];jj=candidate[mask]
        assert abs(aa.sum()-.0002)<1e-10
        patches.append(dict(level=k,center_x_offset_mm=center*1000,rectangle_bounds_xz_m=[lo.tolist(),hi.tolist()],area_m2=float(aa.sum()),raw_area_mean_MPa=float(aa@s[jj]/aa.sum()),raw_area_rms_MPa=float(np.sqrt(aa@(s[jj]**2)/aa.sum())),max_MPa=float(s[jj].max()),contributing_elements=len(jj)))
    # Threshold areas confined to a fixed 50-mm vicinity; centroid selection is descriptive.
    for threshold in (200,300,400,500):
        selected=local&(s>threshold)
        tails.append(dict(level=k,threshold_MPa=threshold,area_m2=float(q['area_m2'][selected].sum()),element_count=int(selected.sum())))
    rr=np.column_stack([np.loadtxt(S[k]/'resultants_1.txt'),np.loadtxt(S[k]/'resultants_5.txt')])
    resultants.append(dict(level=k,endpoint_element=int(ii[ip]+1),SMISC1_to_8=rr[ii[ip]].tolist(),note='Saved shell resultants in element axes, preserved ordering. No membrane/bending stress split inferred from equivalent-stress scalars.'))
    print('Audited mesh',k,flush=True)
# Parse actual ANSYS shape warnings without confusing element ids with echoed APDL input.
warn=[]
for k in range(4):
    text=(S[k]/'output.out').read_text(errors='replace')
    matches=re.findall(r'Quadrilateral element\s+(\d+)\s+has an aspect ratio of\s+([\d.E+\-]+)',text)
    eid=quality[k]['endpoint_element'];warn.append(dict(level=k,endpoint_reported_aspect_warning=[float(v) for i,v in matches if int(i)==eid],reported_aspect_warning_count=len(matches),note='No listed warning is not proof of passing; ANSYS may limit repeated messages. Independent geometry metrics reported above.'))
data=dict(origin_xz_m=origin.tolist(),rib_y_m=2.95973,quality=quality,fixed_point_samples=rays,fixed_rectangles=patches,threshold_areas=tails,endpoint_resultants=resultants,solver_warning_audit=warn,source_sha256=hashes,method='Saved raw element face-envelope scalars are piecewise constant. Fixed-point values use containing elements (mean and range if on an edge). Fixed-region means use exact polygon/rectangle intersection areas. No nodal smoothing or solver evaluation.',quality_note='Edge-length ratio and Jacobian condition are independent diagnostics, not the proprietary ANSYS aspect-ratio definition. Scaled corner Jacobian includes corner angle; Jacobian orientation is normalized to the element polygon orientation.')
data['near_skin_5mm_path']=near_skin
(O/'audit.json').write_text(json.dumps(data,indent=2))
for filename,records in [('fixed_point_samples.csv',rays),('fixed_region_samples.csv',patches)]:
    with (O/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
for a,(name,direction) in zip(axs[:2],raydefs.items()):
    for k in range(4):
        rows=[r for r in rays if r['level']==k and r['path']==name];a.plot([r['distance_mm'] for r in rows],[r['raw_MPa'] for r in rows],'o-',label=f'Mesh {k}')
    a.set_title(name);a.set_xlabel('Distance from endpoint (mm)');a.set_ylabel('Containing-element raw stress (MPa)');a.set_xscale('log');a.grid(alpha=.25);a.legend()
for k in range(4):
    rows=[r for r in patches if r['level']==k];axs[2].plot([r['center_x_offset_mm'] for r in rows],[r['raw_area_mean_MPa'] for r in rows],'o-',label=f'Mesh {k}')
axs[2].set_title('Fixed 20 × 10 mm patches');axs[2].set_xlabel('Patch centre x-offset (mm)');axs[2].set_ylabel('Area-mean raw envelope (MPa)');axs[2].grid(alpha=.25);axs[2].legend()
fig.suptitle('Rib 4: fixed physical locations under distributed loading')
fig.savefig(O/'fixed_distance_stresses.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for k in range(4):
    rows=[r for r in near_skin if r['level']==k];ax.plot([r['x_offset_mm'] for r in rows],[r['raw_MPa'] for r in rows],'o-',label=f'Mesh {k}')
ax.set_xlabel('Chordwise offset from endpoint (mm)');ax.set_ylabel('Containing-element raw stress (MPa)');ax.set_title('Fixed path 5 mm vertically below the upper skin');ax.grid(alpha=.25);ax.legend();fig.savefig(O/'near_skin_path.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,6),layout='constrained');p=pol[-1];mask=(p[:,:,0].min(1)<origin[0]+.24)&(p[:,:,1].max(1)>origin[1]-.13)
pc=PolyCollection(p[mask],array=raw[-1][ids[-1]][mask],cmap='turbo',norm=Normalize(0,800),edgecolors='#888',linewidths=.1);ax.add_collection(pc)
for j,(name,d) in enumerate(raydefs.items()):
    pts=origin+rs[:,None]*d;ax.plot(pts[:,0],pts[:,1],'o-',color=['white','black'][j],label=name,markersize=4)
for r in [r for r in patches if r['level']==3]:
    lo,hi=np.array(r['rectangle_bounds_xz_m']);ax.add_patch(plt.Rectangle(lo,*(hi-lo),fill=False,edgecolor='magenta',linewidth=1.5))
ax.scatter(*origin,marker='*',s=100,c='white',edgecolors='black');ax.set_xlim(origin[0]-.01,origin[0]+.24);ax.set_ylim(origin[1]-.13,origin[1]+.17);ax.set_aspect('equal');ax.set_xlabel('Chord x (m)');ax.set_ylabel('Height z (m)');ax.legend(loc='upper left');ax.set_title('Fixed audit paths and patches; magenta rectangles are identical on all meshes');fig.colorbar(pc,ax=ax,label='Raw face stress (MPa)');fig.savefig(O/'audit_locations.png',dpi=180);plt.close(fig)
def table(h,rs):return '<table><tr>'+''.join(f'<th>{x}</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rs)+'</table>'
def image(n):return '<img src="data:image/png;base64,'+base64.b64encode((O/n).read_bytes()).decode()+'">'
report='<!doctype html><html lang="en"><meta charset="utf-8"><title>Rib spatial and element-quality audit</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px}</style><h1>Rib 4 spatial and element-quality audit</h1><p>18 November 2026. Saved distributed-load results only; no new ANSYS calculation or article edit.</p>'+image('audit_locations.png')+image('fixed_distance_stresses.png')
for name in raydefs:
    report+='<h2>'+name+'</h2>'+table(['Distance mm','Baseline MPa','First MPa','Second MPa','Third MPa','Last change %'],[[r,*[f'{next(x for x in rays if x["level"]==k and x["path"]==name and x["distance_mm"]==r)["raw_MPa"]:.3f}' for k in range(4)],f'{100*(next(x for x in rays if x["level"]==3 and x["path"]==name and x["distance_mm"]==r)["raw_MPa"]/next(x for x in rays if x["level"]==2 and x["path"]==name and x["distance_mm"]==r)["raw_MPa"]-1):+.2f}'] for r in rs*1000])
report+='<h2>Fixed patches</h2>'+table(['Centre x-offset mm','Baseline MPa','First MPa','Second MPa','Third MPa','Last change %'],[[r,*[f'{next(x for x in patches if x["level"]==k and x["center_x_offset_mm"]==r)["raw_area_mean_MPa"]:.3f}' for k in range(4)],f'{100*(next(x for x in patches if x["level"]==3 and x["center_x_offset_mm"]==r)["raw_area_mean_MPa"]/next(x for x in patches if x["level"]==2 and x["center_x_offset_mm"]==r)["raw_area_mean_MPa"]-1):+.2f}'] for r in (30.,50.,100.,200.)])
report+='<h2>Endpoint element quality</h2>'+table(['Mesh','Peak MPa','Centroid distance mm','Edge ratio','Min angle deg','Min scaled Jacobian','Jacobian det min/max'],[[q['level'],f'{q["endpoint_stress_MPa"]:.3f}',f'{q["endpoint_centroid_distance_mm"]:.4f}',f'{q["endpoint"]["edge_ratio"]:.3f}',f'{q["endpoint"]["min_angle_deg"]:.3f}',f'{q["endpoint"]["min_scaled_corner_jacobian"]:.4f}',f'{q["endpoint"]["jacobian_det_ratio"]:.4f}'] for q in quality])
report+='<h2>Near-skin check</h2>'+image('near_skin_path.png')+table(['Chordwise offset mm','Baseline MPa','First MPa','Second MPa','Third MPa'],[[r,*[f'{next(x for x in near_skin if x["level"]==k and x["x_offset_mm"]==r)["raw_MPa"]:.3f}' for k in range(4)]] for r in rs*1000])
report+='<h2>Findings</h2><p><strong>Stresses in the sampled rib interior stabilize away from the endpoint, while the endpoint peak continues growing.</strong> Fixed 20×10 mm patches centred at chordwise offsets 30, 50, 100 and 200 mm change by −0.067%, +2.308%, +2.155% and +1.468% on the last refinement. At 50–200 mm on both interior rays, last-step changes are small (less than 2.7% in magnitude). The endpoint stress rises 140.422 → 253.777 → 435.113 → 765.233 MPa while its element centroid approaches the endpoint from 13.675 to 1.730 mm.</p><p>The conclusion is local to the sampled interior. At 5–10 mm there are still large changes; on the path 5 mm below the upper skin some locations remain sensitive (for example 122.06 → 89.52 MPa at 50 mm chordwise offset). Thus the whole attachment strip cannot be called converged. Other localized attachment peaks and coarse sampling affect that path.</p><p>Endpoint edge ratio stays 6.366, minimum angle stays 47.833°, and minimum scaled corner Jacobian stays 0.7412. The within-element Jacobian determinant ratio improves from 0.7800 to 0.9659. All checked rib elements have positive orientation-normalized Jacobians and convex corners. Increasing geometric distortion is therefore not observed as an explanation for peak growth. The mesh remains elongated, so an independent element-shape study would still be needed to exclude all discretization effects.</p><p>Within the fixed 50-mm neighbourhood, the sampled area above 300 MPa decreases from approximately 14.239 mm² on mesh 2 to 8.748 mm² on mesh 3 while the peak increases. This descriptive measure uses centroid inclusion and supports localization. Taken together, growing endpoint stress, stabilizing interior fields and non-deteriorating element shape strengthen the evidence for a local singularity-like response; they do not constitute an analytic proof of an unbounded limit.</p>'
report+='<h2>Interpretation limits</h2><p>The saved stress is an element-level envelope over face layers and through-thickness samples, not a pointwise stress tensor. Values at fixed points therefore change with the containing cell. Coarse cells can span both the hotspot and the observation point. Fixed-patch averages reduce this sampling effect and use identical physical regions with exact overlap areas. They summarize the raw envelope; they are not stresses recomputed from an averaged tensor, nor the sizing filter.</p><p>Rib elements are planar; their warpage is zero to saved-coordinate precision. Jacobian signs are checked against each polygon orientation. Edge ratio, angle, scaled Jacobian and Jacobian condition are independent geometric diagnostics and must not be equated with ANSYS aspect-ratio limits. Persistent distortion could affect stress accuracy even if it does not worsen with refinement.</p><p>Signed membrane and bending stresses were not exported in the saved compact files. Saved shell resultants are retained in audit.json, but a unique stress-tensor decomposition cannot be recovered by subtracting equivalent stresses. No additional native postprocessing run was launched. The cut motion remains inherited from the original wing.</p><p><a href="audit.json">Detailed geometry metrics, samples, resultants and checksums</a> · <a href="fixed_point_samples.csv">Point samples</a> · <a href="fixed_region_samples.csv">Fixed-region samples</a></p></html>'
(O/'spatial_quality_report.html').write_text(report,encoding='utf-8')
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print(json.dumps(dict(quality=quality,patches=patches,rays=rays),indent=2))
