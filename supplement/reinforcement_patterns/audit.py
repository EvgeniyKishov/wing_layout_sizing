"""Read-only postprocessing of saved fields. No solver imports or calls."""
from pathlib import Path
import json, ast, hashlib, gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
P=R/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16'
h=json.loads((P/'history.json').read_text()); rows=[r for r in h if r['phase']=='buckling']
m=dict(np.load(P/'geometry/mesh.npz')); ds=[dict(np.load(P/r['folder']/'fields.npz')) for r in rows]
xyz=m['xyz']; q=xyz[m['corners']-1]; c=m['centers']; g=m['groups']; a=np.loadtxt(P/rows[-1]['folder']/'actual_areas.txt')
skin=g==0; ribs=np.array([np.median(c[g==i,1]) for i in range(4,20)])
# Chord coordinate from actual section extrema of skin mesh nodes.
sn=xyz[np.unique(m['corners'][skin])-1]; sy=np.unique(sn[:,1]); le=np.array([sn[sn[:,1]==y,0].min() for y in sy]); te=np.array([sn[sn[:,1]==y,0].max() for y in sy])
xi=(c[:,0]-np.interp(c[:,1],sy,le))/(np.interp(c[:,1],sy,te)-np.interp(c[:,1],sy,le))
aft=skin&(xi>.7)&(xi<.97)
near=np.min(abs(c[:,1,None]-ribs[None,:]),axis=1)<.10
source=(P/'sources/run_fine_fsd_buckling.py').read_text(); fun=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='grow')
ns={'np':np};exec(compile(ast.Module(body=[fun],type_ignores=[]),'<saved grow function>','exec'),ns)
steps=[]; alloc={k:np.zeros(len(g)) for k in ['initial_uniform_core','energy_core','restoration_core','faces']}
for k,(d,e) in enumerate(zip(ds[:-1],ds[1:])):
 r=rows[k]; rr=rows[k+1]; restore=rr['rule']=='uniform_restoration'
 f,co=ns['grow'](d['face_m'],d['core_m'],d['filtered_Pa'],d['energy'],m['types'],r['BLF'],k+1,restore)
 err=max(float(abs(f-e['face_m']).max()),float(abs(co-e['core_m']).max()));assert err<1e-12
 kind='restoration_core' if restore else ('initial_uniform_core' if k+1<=5 and r['BLF']<.2 else 'energy_core')
 dc=e['core_m']-d['core_m'];df=2*(e['face_m']-d['face_m']);alloc[kind]+=dc;alloc['faces']+=df
 en=d['energy']; peak=int(np.argmax(en));sp=int(np.argmax(np.where(skin,en,-np.inf)))
 steps.append(dict(from_state=k,to_state=k+1,rule=kind,BLF=r['BLF'],filtered_peak_MPa=r['filtered_peak_MPa'],core_mass_added_kg=float(a@dc*75),face_mass_added_kg=float(a@df*2700),replay_error_m=err,energy_peak_group=int(g[peak]),energy_peak_xyz=c[peak].tolist(),upper_energy_peak_xyz=c[sp].tolist(),upper_energy_peak_xi=float(xi[sp]),upper_energy_fraction=float(en[skin].sum()/en.sum())))
def stats(mask):
 out={}
 for name,v in alloc.items():
  out[name]={'area_mean_added_mm':float(np.average(v[mask]*1000,weights=a[mask])),'max_added_mm':float(v[mask].max()*1000),'added_mass_kg':float(a[mask]@v[mask]*(2700 if name=='faces' else 75))}
 for name in ['face_m','core_m','total_m']:
  out[name]={'initial_mean_mm':float(np.average(ds[0][name][mask]*1000,weights=a[mask])),'final_mean_mm':float(np.average(ds[-1][name][mask]*1000,weights=a[mask])),'final_range_mm':[float(ds[-1][name][mask].min()*1000),float(ds[-1][name][mask].max()*1000)]}
 return out
out={'scope':'Saved 16-rib reference, 33 buckling states; postprocessing only. Aft diagnostic region 0.70<xi<0.97; near rib means span distance <0.10 m. Thresholds are descriptive.', 'steps':steps,'regions':{name:stats(mask) for name,mask in [('all',np.ones(len(g),bool)),('upper',skin),('upper_aft',aft),('upper_aft_near_ribs',aft&near),('upper_aft_away_ribs',aft&~near)]},'rib_span_m':ribs.tolist(),'replayed_all_32_transitions':True}
(O/'audit.json').write_text(json.dumps(out,indent=2))
def panel(ax,v,title,mask=skin,cmap='viridis',vmin=None,vmax=None):
 pc=PolyCollection(q[mask][:,:,[1,0]],array=v[mask],cmap=cmap,edgecolors='none',clim=(vmin,vmax));ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set(xlabel='Span y (m)',ylabel='Chord x (m)')
 for y in ribs:ax.axvline(y,color='black',lw=.35,alpha=.35)
 plt.colorbar(pc,ax=ax,pad=.01,shrink=.8)
fig,axs=plt.subplots(5,1,figsize=(13,12),layout='constrained')
for ax,v,title in zip(axs,[ds[0]['total_m']*1000,ds[-1]['total_m']*1000,alloc['energy_core']*1000,alloc['faces']*1000,ds[-1]['core_m']*1000],['After FSD: total thickness (mm)','Final: total thickness (mm)','Core added by energy-weighted steps (mm)','Both faces: total addition during reinforcement (mm)','Final core thickness (mm)']):panel(ax,v,title)
fig.savefig(O/'thickness_decomposition.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(4,2,figsize=(15,10),layout='constrained')
for j,k in enumerate([2,5,14,28]):
 en=ds[k]['energy'];norm=en/np.max(en[np.isin(m['types'],[1,4])]);weight=.1+np.maximum(norm,0)**.05
 panel(axs[j,0],np.log10(np.maximum(norm,1e-12)),f'State {k}: log10 normalized modal energy',vmin=-12,vmax=0)
 dc=(ds[k+1]['core_m']-ds[k]['core_m'])*1000
 panel(axs[j,1],dc,f'Following core increment {k} to {k+1} (mm)')
fig.savefig(O/'energy_and_increment.png',dpi=160);plt.close(fig)
# Aft chord strip profiles, shown with actual rib stations.
fig,axs=plt.subplots(3,1,figsize=(13,9),layout='constrained')
sel=skin&(xi>.78)&(xi<.88); yvals=np.unique(np.round(c[sel,1],6))
for ax,key,label in zip(axs,['core_m','face_m','total_m'],['Core','One face','Total']):
 for k in [0,2,5,14,30,32]:
  yy=[];vv=[]
  for y in yvals:
   mm=sel&(abs(c[:,1]-y)<1e-6)
   if mm.any():yy.append(y);vv.append(np.average(ds[k][key][mm]*1000,weights=a[mm]))
  ax.plot(yy,vv,label=f'State {k}',lw=1)
 for y in ribs:ax.axvline(y,color='black',lw=.5,alpha=.4)
 ax.set(ylabel=label+' thickness (mm)',xlabel='Span y (m)');ax.grid(alpha=.2)
axs[0].legend(ncol=6);fig.suptitle('Upper-skin aft strip: 0.78 < local chord fraction < 0.88; vertical lines = ribs')
fig.savefig(O/'aft_profiles.png',dpi=160);plt.close(fig)
zoom=skin&(c[:,1]<8)&(xi>.7)&(xi<.95)
fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained')
for ax,v,title in zip(axs,[ds[0]['core_m']*1000,ds[-1]['core_m']/ds[0]['core_m'],ds[-1]['core_m']*1000],['Aft upper skin: core thickness already present after FSD (mm)','Cumulative core multiplication factor (final / after FSD)','Final core thickness (mm)']):
 panel(ax,v,title,mask=zoom);ax.set_xlim(-.1,8.15)
fig.savefig(O/'aft_pattern_origin.png',dpi=170);plt.close(fig)
# Inspect saved first-mode displacements without exporting or solving anything.
normals=np.cross(q[:,1]-q[:,0],q[:,3]-q[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
mode_stats=[];fig,axs=plt.subplots(3,2,figsize=(14,9),layout='constrained')
for j,k in enumerate([2,14,32]):
 path=P/rows[k]['folder']/'mode_1.txt'
 if not path.exists():path=path.with_suffix('.txt.gz')
 u=np.loadtxt(path);assert u.shape==xyz.shape
 un=np.einsum('ij,ij->i',u[m['corners']-1].mean(axis=1),normals);un/=np.max(abs(un[skin]))
 en=ds[k]['energy']/ds[k]['energy'][skin].max()
 panel(axs[j,0],un,f'State {k}: signed normal mode displacement / upper-skin max',cmap='coolwarm',vmin=-1,vmax=1)
 panel(axs[j,1],np.log10(np.maximum(en,1e-12)),f'State {k}: log10 normalized upper-skin modal energy',vmin=-12,vmax=0)
 ii=np.argmax(np.where(skin,abs(un),-1));jj=np.argmax(np.where(skin,en,-1))
 mode_stats.append(dict(state=k,upper_normal_displacement_max_xyz=c[ii].tolist(),upper_energy_max_xyz=c[jj].tolist()))
fig.savefig(O/'saved_modes.png',dpi=150);plt.close(fig)
out['saved_mode_inspection']=mode_stats
out['aft_initial_final_core_pearson']=float(np.corrcoef(ds[0]['core_m'][aft],ds[-1]['core_m'][aft])[0,1])
out['aft_core_multiplier_range']=[float((ds[-1]['core_m']/ds[0]['core_m'])[aft].min()),float((ds[-1]['core_m']/ds[0]['core_m'])[aft].max())]
out['rib_trailing_endpoints_xyz']=[q[g==i].reshape(-1,3)[np.argmax(q[g==i].reshape(-1,3)[:,0])].tolist() for i in range(4,20)]
(O/'audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'regions':out['regions'],'energy_peak_groups':[x['energy_peak_group'] for x in steps],'max_replay_error_m':max(x['replay_error_m'] for x in steps)},indent=2))
