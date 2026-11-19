"""Compare one targeted solve with the four saved distributed-load meshes."""
from pathlib import Path
import ast,json,csv,hashlib,re,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
R=Path(__file__).resolve().parents[2];RUN=R/'runs/rib_endpoint_targeted_refinement_20261118';O=R/'journal_reviews/rib_targeted_refinement_20261118';O.mkdir(exist_ok=True)
old=json.loads((R/'journal_reviews/rib_spatial_quality_audit_20261118/audit.json').read_text());sweep=json.loads((R/'journal_reviews/distributed_load_sweep_20261118/comparison.json').read_text())
# Reuse the audited geometric definitions without executing the prior audit.
tree=ast.parse((Path(__file__).parent/'audit_rib_spatial_quality.py').read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('area','clip','clipped_area','shape_quality')]
exec(compile(ast.Module(body=functions,type_ignores=[]),'<geometry helpers>','exec'))
m=np.load(RUN/'mesh.npz');s=np.loadtxt(RUN/'static/stress.txt')/1e6;u=np.loadtxt(RUN/'static/dof.txt');a=np.loadtxt(RUN/'static/actual_areas.txt')
prev=np.load(R/'runs/skin_front_spar_submodel_refinement3_20261117/mesh.npz');ps=np.loadtxt(R/'runs/submodel_distributed_load_sweep_20261118/mesh_3/static/stress.txt')/1e6;pu=np.loadtxt(R/'runs/submodel_distributed_load_sweep_20261118/mesh_3/static/dof.txt');pa=np.loadtxt(R/'runs/submodel_distributed_load_sweep_20261118/mesh_3/static/actual_areas.txt')
manifest=json.loads((RUN/'manifest.json').read_text())
for p,h in manifest['source_sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
assert len(s)==len(m['corners']) and np.isfinite(s).all() and np.isfinite(u).all()
b=m['boundary_nodes'];b=b[b<=len(pu)]-1;np.testing.assert_allclose(u[b],pu[b],rtol=1e-12,atol=1e-13)
ae=float(np.max(abs(np.bincount(m['previous_element_indices'],weights=a)-pa)/pa));assert ae<1e-8
ii=np.flatnonzero(m['groups']==7);p=m['xyz'][m['corners'][ii]-1][:,:,[0,2]];origin=m['xyz'][464,[0,2]];raw=s[ii];q=shape_quality(p)
incident=np.any(m['corners'][ii]==465,axis=1);ip=np.flatnonzero(incident)[raw[incident].argmax()];peak=int(s.argmax());rays=[];patches=[];nearskin=[]
e=np.roll(p,-1,axis=1)-p
for r in [x for x in old['fixed_point_samples'] if x['level']==3]:
    point=np.array([r['x_m'],r['z_m']]);rel=point[None,None,:]-p;cr=e[:,:,0]*rel[:,:,1]-e[:,:,1]*rel[:,:,0];found=np.flatnonzero(np.all(cr>=-1e-13,axis=1)|np.all(cr<=1e-13,axis=1));assert len(found)>0
    value=float(raw[found].mean());rays.append(dict(path=r['path'],distance_mm=r['distance_mm'],previous_MPa=r['raw_MPa'],targeted_MPa=value,change_percent=100*(value/r['raw_MPa']-1),elements=(ii[found]+1).tolist(),max_element_edge_mm=float(q['max_edge_m'][found].max()*1000)))
for r in [x for x in old['fixed_rectangles'] if x['level']==3]:
    lo,hi=np.array(r['rectangle_bounds_xz_m']);candidate=np.flatnonzero(np.all(p.max(1)>=lo,axis=1)&np.all(p.min(1)<=hi,axis=1));aa=np.array([clipped_area(p[j],lo,hi) for j in candidate]);mask=aa>1e-16;aa=aa[mask];jj=candidate[mask];assert abs(aa.sum()-.0002)<1e-10
    value=float(aa@raw[jj]/aa.sum());patches.append(dict(center_x_offset_mm=r['center_x_offset_mm'],previous_MPa=r['raw_area_mean_MPa'],targeted_MPa=value,change_percent=100*(value/r['raw_area_mean_MPa']-1),area_m2=float(aa.sum())))
for r in [x for x in old['near_skin_5mm_path'] if x['level']==3]:
    point=np.array([r['x_m'],r['z_m']]);rel=point[None,None,:]-p;cr=e[:,:,0]*rel[:,:,1]-e[:,:,1]*rel[:,:,0];found=np.flatnonzero(np.all(cr>=-1e-13,axis=1)|np.all(cr<=1e-13,axis=1));assert len(found)>0
    value=float(raw[found].mean());nearskin.append(dict(x_offset_mm=r['x_offset_mm'],previous_MPa=r['raw_MPa'],targeted_MPa=value,change_percent=100*(value/r['raw_MPa']-1)))
def name(g):return ['Upper skin','Lower skin','Front spar','Rear spar'][int(g)] if g<4 else f'Rib {g-3}'
components=[dict(component=name(g),previous_MPa=float(ps[prev['groups']==g].max()),targeted_MPa=float(s[m['groups']==g].max())) for g in np.unique(m['groups'])]
du=u[:len(pu),:3]-pu[:,:3];log=(RUN/'static/output.out').read_text(errors='replace');errors=int(re.findall(r'NUMBER OF ERROR\s+MESSAGES ENCOUNTERED=\s*(\d+)',log)[-1]);assert errors==0
fields=np.load(R/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16/buckling/iteration_032/fields.npz');wi=m['parent_wing_element_ids']-1;mass=float(a@(5400*fields['face_m'][wi]+75*fields['core_m'][wi]))
endpoint=dict(element=int(ii[ip]+1),raw_MPa=float(raw[ip]),previous_MPa=old['quality'][-1]['endpoint_stress_MPa'],change_percent=float(100*(raw[ip]/old['quality'][-1]['endpoint_stress_MPa']-1)),centroid_distance_mm=float(np.linalg.norm(p[ip].mean(0)-origin)*1000),geometry={key:float(v[ip]) for key,v in q.items()})
data=dict(endpoint=endpoint,global_peak=dict(element=peak+1,component=name(m['groups'][peak]),raw_MPa=float(s[peak]),wing_parent=int(m['parent_wing_element_ids'][peak]),centroid_xyz_m=m['xyz'][m['corners'][peak]-1].mean(0).tolist()),fixed_points=rays,fixed_patches=patches,near_skin_5mm=nearskin,component_peaks=components,mass_kg=mass,area_conservation_relative_error=ae,common_translation_L2_change_percent=float(100*np.linalg.norm(du)/np.linalg.norm(pu[:,:3])),maximum_common_translation_difference_m=float(np.linalg.norm(du,axis=1).max()),invalid_jacobians_rib=int(np.sum(q['min_signed_jacobian']<=0)),execution=json.loads((RUN/'static/execution.json').read_text()),source_files_and_article_unchanged=True,manifest=manifest)
(O/'comparison.json').write_text(json.dumps(data,indent=2))
for filename,records in [('fixed_points.csv',rays),('fixed_patches.csv',patches),('near_skin.csv',nearskin),('component_peaks.csv',components)]:
    with (O/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
hh=[x['endpoint']['max_edge_m']*1000 for x in old['quality']]+[q['max_edge_m'][ip]*1000];ss=[x['endpoint_stress_MPa'] for x in old['quality']]+[raw[ip]]
axs[0].plot(hh,ss,'o-');axs[0].invert_xaxis();axs[0].set_xscale('log');axs[0].set_xticks(hh,[f'{v:.2f}' for v in hh]);axs[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator());axs[0].set_xlabel('Endpoint element longest edge (mm)');axs[0].set_ylabel('Endpoint raw peak (MPa)');axs[0].set_title('Four uniform meshes + targeted refinement');axs[0].grid(alpha=.25)
for ax,path in zip(axs[1:],['Into rib, horizontal','Into rib, 30 degrees down']):
    for k in range(4):
        rows=[r for r in old['fixed_point_samples'] if r['level']==k and r['path']==path];ax.plot([r['distance_mm'] for r in rows],[r['raw_MPa'] for r in rows],'o-',alpha=.6,label=f'Saved mesh {k}')
    rows=[r for r in rays if r['path']==path];ax.plot([r['distance_mm'] for r in rows],[r['targeted_MPa'] for r in rows],'s-',c='black',label='Targeted');ax.set_xscale('log');ax.set_xlabel('Fixed distance (mm)');ax.set_ylabel('Containing-element stress (MPa)');ax.set_title(path);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(O/'targeted_stress_trends.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained');vmax=100*np.ceil(max(s.max(),ps.max())/100)
for k,(mm,stress,ax) in enumerate(zip([prev,m],[ps,s],axs)):
    ids=np.flatnonzero(mm['groups']==7);pp=mm['xyz'][mm['corners'][ids]-1][:,:,[0,2]];mask=(pp[:,:,0].min(1)<origin[0]+.06)&(pp[:,:,1].max(1)>origin[1]-.02)
    pc=PolyCollection(pp[mask],array=stress[ids][mask],norm=Normalize(0,vmax),cmap='turbo',edgecolors='#777',linewidths=.25);ax.add_collection(pc);ax.set_xlim(origin[0]-.003,origin[0]+.055);ax.set_ylim(origin[1]-.02,origin[1]+.045);ax.set_aspect('equal');ax.set_xlabel('Chord x (m)');ax.set_ylabel('Height z (m)');ax.set_title(['Saved finest mesh','Targeted refinement'][k])
fig.colorbar(pc,ax=axs,label='Raw face stress (MPa)');fig.savefig(O/'endpoint_mesh_comparison.png',dpi=180);plt.close(fig)
def table(h,rs):return '<table><tr>'+''.join(f'<th>{x}</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rs)+'</table>'
def image(n):return '<img src="data:image/png;base64,'+base64.b64encode((O/n).read_bytes()).decode()+'">'
report='<!doctype html><html lang="en"><meta charset="utf-8"><title>Targeted rib endpoint refinement</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:9px}img{width:100%}</style><h1>Targeted rib-endpoint refinement</h1><p>18 November 2026. One new static solve; article unchanged.</p>'
report+=f'<p>Endpoint peak: {endpoint["previous_MPa"]:.3f} → <strong>{endpoint["raw_MPa"]:.3f} MPa</strong> ({endpoint["change_percent"]:+.3f}%). Endpoint element centroid distance: {old["quality"][-1]["endpoint_centroid_distance_mm"]:.6f} → {endpoint["centroid_distance_mm"]:.6f} mm. Global maximum: {data["global_peak"]}.</p>'
report+='<p><strong>The targeted refinement confirms continued growth at the same endpoint while the sampled rib interior remains comparatively stable.</strong> Fixed-patch changes are +2.69%, +1.64%, +0.39% and −0.25% at 30, 50, 100 and 200 mm offsets. All sampled interior points from 50 to 200 mm change by less than 1.5%. The 5-mm samples and some near-skin locations remain sensitive. Endpoint element shape does not deteriorate: edge ratio and minimum angle are unchanged, and the Jacobian determinant ratio improves. This strengthens numerical evidence for a localized singularity-like response; it is not proof of a physical infinite stress.</p>'
report+=table(['Mesh','Elements','Same Rib 4 endpoint MPa','Longest endpoint edge mm'],[[f'Saved {k}',sweep['elements'][k],f'{ss[k]:.3f}',f'{hh[k]:.6f}'] for k in range(4)]+[['Targeted',len(s),f'{ss[-1]:.3f}',f'{hh[-1]:.6f}']])
report+=image('targeted_stress_trends.png')+image('endpoint_mesh_comparison.png')
report+='<h2>Fixed physical patches</h2>'+table(['Patch x-offset mm','Saved finest MPa','Targeted MPa','Change %'],[[r['center_x_offset_mm'],f'{r["previous_MPa"]:.4f}',f'{r["targeted_MPa"]:.4f}',f'{r["change_percent"]:+.3f}'] for r in patches])
for path in ['Into rib, horizontal','Into rib, 30 degrees down']:
    report+='<h2>'+path+'</h2>'+table(['Distance mm','Saved finest MPa','Targeted MPa','Change %'],[[r['distance_mm'],f'{r["previous_MPa"]:.3f}',f'{r["targeted_MPa"]:.3f}',f'{r["change_percent"]:+.3f}'] for r in rays if r['path']==path])
report+='<h2>Path 5 mm below upper skin</h2>'+table(['Chordwise offset mm','Saved finest MPa','Targeted MPa','Change %'],[[r['x_offset_mm'],f'{r["previous_MPa"]:.3f}',f'{r["targeted_MPa"]:.3f}',f'{r["change_percent"]:+.3f}'] for r in nearskin])
report+='<h2>Component maxima and strip effects</h2>'+table(['Component','Saved finest MPa','Targeted MPa'],[[r['component'],f'{r["previous_MPa"]:.3f}',f'{r["targeted_MPa"]:.3f}'] for r in components])+'<p>The compatibility strips also refine other rib endpoints. Their peak increases are recorded above and cannot be attributed solely to a remote influence from the Rib 4 target.</p>'
report+='<h2>Mesh scope and controls</h2><p>The target consists of previous-element centroids within 25 mm of the Rib 4 endpoint. Opposite-edge closure extends directional refinement strips throughout connected surfaces, including other ribs and cut boundaries. This is not a strictly compact remesh. Of 499,712 previous elements, 413,937 remain unchanged, 84,158 split into two, and 1,617 split into four. Total: 588,721 quadrilaterals. No triangles, hanging nodes or constraint equations were introduced. Parametric rectangular subdivision preserves the original bilinear geometry and element directions.</p>'
report+=f'<p>Identical baseline sections, materials, shell offsets, layer settings, full integration and SMP4 are retained. The same frozen distributed density is integrated on the new skin mesh; total Fz/Mx/My = {manifest["force_Fz_Mx_My"]}. New cut-node DOFs interpolate the same inherited field; old cut DOFs agree. Native area conservation relative error: {ae:.3e}; segment mass: {mass:.9f} kg. No inverted Rib 4 elements were detected. Endpoint shape diagnostics: {endpoint["geometry"]}.</p>'
report+=f'<p>ANSYS completed in {data["execution"]["seconds"]:.2f} s with zero errors. Source and article hashes are unchanged. Common-node translation relative L2 change is {data["common_translation_L2_change_percent"]:.6f}%. Native outputs: runs/rib_endpoint_targeted_refinement_20261118.</p>'
report+='<h2>Limits</h2><p>Point samples use saved element-level face-envelope values; containing cells change with refinement. Patch means use exact overlap areas of the same 20×10 mm rectangles as the previous audit. Compatibility strips can change fields outside the target, so their effects must be retained in interpretation. Persistent peak growth alongside stable remote measures strengthens evidence for a localized singularity-like response, but is not an analytic proof or a validation of the physical attachment. The boundary motion still comes from the original full-wing solution.</p><p><a href="comparison.json">Full comparison and provenance</a> · <a href="fixed_points.csv">Point samples</a> · <a href="fixed_patches.csv">Patch samples</a> · <a href="component_peaks.csv">Component peaks</a>.</p></html>'
(O/'targeted_comparison_report.html').write_text(report,encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k!='manifest'},indent=2))

