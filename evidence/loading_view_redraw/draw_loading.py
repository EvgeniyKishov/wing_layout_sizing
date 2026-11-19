from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
here=Path(__file__).resolve().parent
root=here.parents[3];out=here.parents[1]
source=here/'aerodynamic_loading_data.npz'
meshfile=root/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16/geometry/mesh.npz'
d=np.load(source);m=np.load(meshfile)
polys=d['vertices_xyz'].copy();force=d['Fz_N'];load=d['vertical_load_kPa'];xy=d['source_xy'];idx=d['arrow_indices'];area=d['projected_area_m2']
xyz=m['xyz'];corners=m['corners']-1;un=np.unique(corners[m['groups']==0]);interp=LinearNDInterpolator(xyz[un,:2],xyz[un,2]);near=NearestNDInterpolator(xyz[un,:2],xyz[un,2])
def project(p):
 z=interp(p);missing=~np.isfinite(z)
 z[missing]=near(p[missing]);return z,int(missing.sum())
z,nedge=project(polys[:,:,:2].reshape(-1,2));polys[:,:,2]=z.reshape(polys.shape[:2]);bases,nb=project(xy[idx]);assert nb==0
assert np.allclose(load*1000*area,force,rtol=1e-12)
L=xyz[:,1].max();scale=2./float(force[idx].max()) # graphical height per N, same for all selected forces
maxheight=max(2.,float(force[idx].max()*scale+.4))
plt.rcParams.update({'font.size':10})
fig=plt.figure(figsize=(10.5,9.5));axes=[];norm=Normalize(0,40);colors=plt.get_cmap('jet')(norm(load))
for k in range(2):
 ax=fig.add_axes([.05,.53-k*.45,.80,.40],projection='3d',computed_zorder=False);axes.append(ax)
 ax.add_collection3d(Poly3DCollection(polys[:,:,[1,0,2]],facecolors=colors,edgecolors='none',zorder=1))
 ax.scatter(xy[idx,1],xy[idx,0],bases+.015,s=3,c='black',depthshade=False,zorder=3)
 if k:ax.quiver(xy[idx,1],xy[idx,0],bases,0,0,force[idx]*scale,color='#222222',arrow_length_ratio=.13,linewidth=.8,zorder=4)
 ax.set(xlim=(-.3,L+.3),ylim=(4.1,-.2),zlim=(-.4,maxheight));ax.set_xticks([0,4,8,12]);ax.set_yticks([0,2,4]);ax.set_zticks([])
 ax.set_xlabel('Span y (m)',labelpad=9);ax.set_ylabel('Chord x (m)',labelpad=9)
 ax.set_box_aspect((L,4,3),zoom=1.45);ax.view_init(elev=55,azim=-70);ax.set_proj_type('ortho')
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
 ax.grid(False)
 ax.text(0,4.0,-.1,'Root',fontsize=9);ax.text(L,2.3,-.1,'Tip',fontsize=9)
 ax.text(6,-.6,-.12,'Leading edge',fontsize=8);ax.text(6,3.7,-.12,'Trailing edge',fontsize=8)
 ax.text2D(.04,.98,'(a) Loading intensity; dots mark sampled panels' if not k else '(b) Same field with selected panel-force vectors',transform=ax.transAxes,fontsize=11)
# Explicit vector scale, drawn in the same 3-D coordinates and camera as the data arrows.
ax=axes[1];ax.quiver(13.5,4.0,0,0,0,200*scale,color='#222222',arrow_length_ratio=.13,linewidth=1.2,zorder=5);ax.text(13.5,4.0,200*scale+.13,'200 N',fontsize=10)
cax=fig.add_axes([.88,.29,.022,.45]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='jet'),cax=cax,extend='max');cb.set_label(r'$p_{z,a}=F_{z,a}/A_{xy,a}$ (kPa)')
fig.text(.87,.23,'Display cap: 40 kPa\nPeak: 91.657 kPa',fontsize=9,ha='center')
fig.text(.5,.975,'Same saved loading | V = 128 m/s | Semi-wing resultant = 244.274 kN',ha='center',fontsize=12)
fig.text(.5,.035,r'$F_{z,a}=p_{z,a}A_{xy,a}$: variable panel area means colour and arrow length are not proportional.',ha='center',fontsize=10)
fig.text(.5,.012,'All 1365 panels in colour; selected source-panel forces shown before structural load transfer.',ha='center',fontsize=9)
fig.savefig(out/'figures/aerodynamic_loading.pdf');fig.savefig(out/'figures/aerodynamic_loading.png',dpi=200);plt.close(fig)
audit={'saved_source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'saved_mesh':str(meshfile),'mesh_sha256':hashlib.sha256(meshfile.read_bytes()).hexdigest(),'source_panels':len(force),'arrow_count':len(idx),'selected_indices':idx.tolist(),'resultant_N':float(force.sum()),'max_intensity_kPa':float(load.max()),'display_projection':'Linear interpolation onto saved upper-skin nodes; nearest boundary-node height for aerodynamic corners outside the triangulated skin footprint. Planform coordinates and source force/intensity values unchanged.','boundary_vertex_fallback_count':nedge,'source_anchor_fallback_count':nb,'force_arrow_display_m_per_N':scale,'camera':'Identical orthographic elev=55 azim=-70, axis limits and box aspect','new_scientific_solves':False}
(here/'redraw_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Saved-data figure rendered; boundary display vertices:',nedge)
