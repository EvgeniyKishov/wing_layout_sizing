"""Geometric postprocessing of saved fields only. No solver imports or execution."""
from pathlib import Path
import numpy as np,json,re,hashlib
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16';FINAL=RUN/'buckling/iteration_032'
print('Loading saved fields',flush=True)
m=np.load(RUN/'geometry/mesh.npz');d=np.load(FINAL/'fields.npz');corners=m['corners'];q=m['xyz'][corners-1];g=m['groups'];c=m['centers'];t=d['total_m'];f=d['face_m'];core=d['core_m']
a=q[:,1]-q[:,0];b=q[:,3]-q[:,0]
# Bilinear surface normal at element centre.
a=(q[:,1]+q[:,2]-q[:,0]-q[:,3])/2;b=(q[:,3]+q[:,2]-q[:,0]-q[:,1])/2
n=np.cross(a,b);n/=np.linalg.norm(n,axis=1)[:,None]
s=(FINAL/'diagnostic.apdl').read_text();blocks=re.findall(r'SECTYPE,(\d+),SHELL\n(.*?\nEN,[^\n]+)',s,re.S);assert len(blocks)==len(t)
print('Parsed sections',len(blocks),flush=True)
off=[]
for eid,block in blocks:
 i=int(eid)-1;layers=[float(x) for x in re.findall(r'^SECDATA,([^,]+)',block,re.M)];assert max(abs(layers[0]-f[i]),abs(layers[1]-core[i]),abs(layers[2]-f[i]))<1e-12
 off.append(re.search(r'SECOFFSET,(\w+)',block)[1]);en=list(map(int,re.search(r'EN,([^\n]+)',block)[1].split(',')));assert en[0]==i+1;assert en[1:]==list(corners[i])
off=np.array(off);assert np.all(off[g<2]=='BOT') and np.all(off[g>=2]=='MID');assert np.all(n[g==0,2]<0) and np.all(n[g==1,2]>0)
print('Verified layers and offsets',flush=True)
base=np.where(g<2,0,-t/2);verts=np.concatenate([q+n[:,None,:]*base[:,None,None],q+n[:,None,:]*(base+t)[:,None,None]],axis=1)
edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
def cut(v,y,es=edges):
 pts=[]
 for i,j in es:
  if (v[i,1]-y)*(v[j,1]-y)<0:
   u=(y-v[i,1])/(v[j,1]-v[i,1]);pts.append((v[i]+u*(v[j]-v[i]))[[0,2]])
  elif abs(v[i,1]-y)<1e-10:pts.append(v[i,[0,2]])
 if len(pts)<3:return None
 p=np.unique(np.round(pts,12),axis=0)
 if len(p)<3:return None
 return p[ConvexHull(p).vertices]
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def clip(subject,clipping):
 p=subject
 for a,b in zip(clipping,np.roll(clipping,-1,axis=0)):
  if len(p)==0:return np.empty((0,2))
  result=[]
  for u,v in zip(p,np.roll(p,-1,axis=0)):
   du=cross(b-a,u-a);dv=cross(b-a,v-a)
   if du>=-1e-12:result.append(u)
   if (du>0)!=(dv>0) and abs(du-dv)>1e-15:result.append(u+(v-u)*du/(du-dv))
  p=np.array(result)
 return p
def area(p):return abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2 if len(p)>2 else 0.
def section(y):
 ids=np.flatnonzero((g<4)&(verts[:,:,1].min(1)<=y)&(verts[:,:,1].max(1)>=y));pol=[]
 for i in ids:
  p=cut(verts[i],y)
  if p is not None:pol.append((int(i),p))
 overlaps=[]
 for i,p in pol:
  if g[i]!=0:continue
  for j,r in pol:
   if g[j]!=1 or np.any(p.max(0)<r.min(0)) or np.any(r.max(0)<p.min(0)):continue
   z=clip(p,r);ar=area(z)
   if ar>1e-10:overlaps.append((i,j,ar,z))
 return pol,overlaps
ys=np.unique(np.round(m['xyz'][:,1],7));samples=(ys[:-1]+ys[1:])/2
records=[]
for number,y in enumerate(samples):
 if number%20==0:print('Section',number,'of',len(samples),flush=True)
 pol,ov=section(y);records.append({'y_m':float(y),'skin_overlap_area_mm2':float(sum(x[2] for x in ov)*1e6),'pairs':len(ov)})
print('sections',len(samples),'overlap',sum(x['pairs']>0 for x in records),'worst',max(records,key=lambda z:z['skin_overlap_area_mm2']),flush=True)
# Local reference depth, interpolated at the maximum-thickness centroid.
def reference(y):
 curves=[]
 for group in (0,1):
  points=[]
  for i in np.flatnonzero((g==group)&(q[:,:,1].min(1)<=y)&(q[:,:,1].max(1)>=y)):
   for a,b in [(0,1),(1,2),(2,3),(3,0)]:
    if (q[i,a,1]-y)*(q[i,b,1]-y)<0:
     u=(y-q[i,a,1])/(q[i,b,1]-q[i,a,1]);points.append((q[i,a]+u*(q[i,b]-q[i,a]))[[0,2]])
  p=np.unique(np.round(points,12),axis=0);curves.append(p[np.argsort(p[:,0])])
 return curves
stats=[]
for group in range(4):
 ids=np.flatnonzero(g==group);i=ids[np.argmax(t[ids])];curves=reference(c[i,1]);depth=np.interp(c[i,0],curves[0][:,0],curves[0][:,1])-np.interp(c[i,0],curves[1][:,0],curves[1][:,1])
 stats.append(dict(component=['Upper skin','Lower skin','Front spar','Rear spar'][group],total_min_mm=float(t[ids].min()*1000),total_max_mm=float(t[i]*1000),each_face_min_mm=float(f[ids].min()*1000),each_face_max_mm=float(f[ids].max()*1000),core_min_mm=float(core[ids].min()*1000),core_max_mm=float(core[ids].max()*1000),max_element=int(i+1),max_location_m=c[i].tolist(),face_at_max_mm=float(f[i]*1000),core_at_max_mm=float(core[i]*1000),reference_depth_at_max_mm=float(depth*1000),normal_at_max=n[i].tolist(),offset=str(off[i])))
ids=np.flatnonzero(g>=4);stats.append(dict(component='All ribs',total_min_mm=float(t[ids].min()*1000),total_max_mm=float(t[ids].max()*1000),each_face_min_mm=float(f[ids].min()*1000),each_face_max_mm=float(f[ids].max()*1000),core_min_mm=float(core[ids].min()*1000),core_max_mm=float(core[ids].max()*1000),offset='MID'))
# Plot actual section bands, equal axes; no thickness exaggeration.
worst=max(records,key=lambda z:z['skin_overlap_area_mm2']);ymax=float(c[t.argmax(),1]);ymid=float(samples[np.argmin(abs(samples-7.4))]);ytip=float(samples[-1]);chosen=[ymax,ymid,ytip]
colors=['#2166ac','#67a9cf','#de9a28','#925d12']
fig,axs=plt.subplots(3,1,figsize=(13,9),layout='constrained')
for ax,y in zip(axs,chosen):
 pol,ov=section(y)
 for i,p in pol:ax.add_patch(Polygon(p,facecolor=colors[g[i]],edgecolor='none',alpha=.85))
 for i,j,ar,p in ov:ax.add_patch(Polygon(p,facecolor='#d7191c',edgecolor='none'))
 for p in reference(y):ax.plot(p[:,0],p[:,1],color='black',lw=.8)
 ax.autoscale_view();ax.set_aspect('equal');ax.set_xlabel('Chord x (m)');ax.set_ylabel('Height z (m)');ax.set_title(f'y = {y:.4f} m | upper/lower skin overlap area = {sum(z[2] for z in ov)*1e6:.1f} mm^2')
fig.suptitle('Saved selected wing: reconstructed thickness bands at true geometric scale',fontsize=15)
fig.supxlabel('Blue: skin sections; ochre: spar sections; black: saved skin reference surface; red: upper/lower skin intersection.\nElement-centre-normal extrusion of saved sections; geometric diagnostic, not a solid or contact analysis.',fontsize=10)
fig.savefig(OUT/'panel_cross_sections.png',dpi=180);plt.close(fig)
# Zoom worst overlap and attachment.
y=worst['y_m'];pol,ov=section(y)
fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for ax in axs:
 for i,p in pol:ax.add_patch(Polygon(p,facecolor=colors[g[i]],edgecolor='#555555',lw=.2,alpha=.8))
 for i,j,ar,p in ov:ax.add_patch(Polygon(p,facecolor='#d7191c',edgecolor='none'))
 for p in reference(y):ax.plot(p[:,0],p[:,1],color='black',lw=1)
 ax.set_aspect('equal');ax.set_xlabel('Chord x (m)');ax.set_ylabel('Height z (m)')
if ov:
 v=np.concatenate([o[3] for o in ov if o[3][:,0].mean()>np.mean(reference(y)[0][:,0])]);lo=v.min(0);hi=v.max(0);axs[0].set_xlim(lo[0]-.04,hi[0]+.04);axs[0].set_ylim(lo[1]-.04,hi[1]+.04)
spar=np.concatenate([p for i,p in pol if g[i]==2]);xx=spar[:,0].mean();zz=spar[:,1].max();axs[1].set_xlim(xx-.11,xx+.11);axs[1].set_ylim(zz-.18,zz+.04)
axs[0].set_title('Upper/lower skin collision');axs[1].set_title('Front spar extends into upper skin band')
fig.suptitle(f'Geometric intersections at y = {y:.4f} m (true scale)',fontsize=15);fig.savefig(OUT/'panel_fit_details.png',dpi=180);plt.close(fig)
provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'geometry/mesh.npz',FINAL/'fields.npz',FINAL/'diagnostic.apdl']}
out=dict(selected_run=str(RUN),final_iteration=32,component_thicknesses=stats,offset_counts={x:int((off==x).sum()) for x in np.unique(off)},skin_normal_z_range={str(k):[float(n[g==k,2].min()),float(n[g==k,2].max())] for k in (0,1)},section_scan=records,worst_section=worst,source_sha256=provenance,method='Undeformed quad corners extruded along element-centre normals using saved BOT/MID offsets; convex sections intersected in spanwise planes. Approximate geometric reconstruction, no solver or contact calculation. 110 bay-midpoint sections, no CAD tolerance/manufacturing certification.')
(OUT/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(stats,indent=2))

# Vertical clearance at the thickest skin station and thickness overlap at the worst section.
def interval(p,x):
 z=[]
 for a,b in zip(p,np.roll(p,-1,axis=0)):
  if min(a[0],b[0])-1e-12<=x<=max(a[0],b[0])+1e-12 and abs(b[0]-a[0])>1e-14:z.append(float(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0])))
 return (min(z),max(z)) if z else None
x=float(c[t.argmax(),0]);pol,_=section(ymax)
up=[interval(p,x) for i,p in pol if g[i]==0];up=[z for z in up if z]
low=[interval(p,x) for i,p in pol if g[i]==1];low=[z for z in low if z]
clearance=min(z[0] for z in up)-max(z[1] for z in low)
pol,ov=section(worst['y_m']);te=[o for o in ov if o[3][:,0].mean()>np.mean(reference(worst['y_m'])[0][:,0])]
penetration=0
for i,j,ar,p in te:
 for x in np.unique(p[:,0]):
  z=interval(p,x)
  if z:penetration=max(penetration,z[1]-z[0])
out['clearance_at_max_thickness_mm']=clearance*1000
out['worst_section_trailing_edge_overlap_max_vertical_mm']=penetration*1000
out['worst_section_trailing_edge_overlap_x_range_m']=[float(min(o[3][:,0].min() for o in te)),float(max(o[3][:,0].max() for o in te))]
# Shared skin-edge topology: ribs and spars reach skin reference surfaces.
from collections import defaultdict
adj=defaultdict(set)
for i,co in enumerate(corners):
 for aa,bb in zip(co,np.roll(co,-1)):adj[tuple(sorted((int(aa),int(bb))))].add(int(g[i]))
attachment={}
for group in range(2,20):attachment[str(group)]={str(skin):sum(group in gs and skin in gs for gs in adj.values()) for skin in (0,1)}
out['shared_skin_edges_by_group']=attachment
(OUT/'audit.json').write_text(json.dumps(out,indent=2))
print('CLEARANCE_MM',clearance*1000,'TRAILING_EDGE_OVERLAP_MM',penetration*1000,flush=True)
