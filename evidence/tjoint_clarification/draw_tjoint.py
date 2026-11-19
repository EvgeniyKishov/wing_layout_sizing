from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
import numpy as np
out=Path(__file__).resolve().parents[2]
fig,axs=plt.subplots(1,2,figsize=(11,6));navy='#183b56';red='#b52a3c';blue='#2674aa'
a,b=axs
for ax in axs:ax.axis('off')
def P(x,y,z):return np.array([2.5*x+1.3*y, .75*x+1.15*y+3.2*z])
def line3(points,**kw):a.plot(*np.array([P(*p) for p in points]).T,**kw)
def text3(x,y,z,text,**kw):a.text(*P(x,y,z),text,color=navy,**kw)
base=[(-.5,0,0),(.5,0,0),(.5,1,0),(-.5,1,0)]
a.add_patch(Polygon([P(*p) for p in base],facecolor='#dce9f2',edgecolor=navy,lw=1.2))
web=[(0,.25,0),(0,.75,0),(0,.75,.5),(0,.25,.5)]
a.add_patch(Polygon([P(*p) for p in web],facecolor='#f1d49f',edgecolor=navy,lw=1.2))
for x in [-.5,.5]:
 line3([(x,0,0),(x,1,0)],color=red,lw=3)
 for y in np.linspace(0,1,9):line3([(x,y,0),(x+.045,y-.035,-.035)],color=red,lw=1)
line3([(0,.25,0),(0,.75,0)],color=navy,lw=3)
for y in np.linspace(.25,.75,6):a.annotate('',xy=P(0,y,.70),xytext=P(0,y,.5),arrowprops=dict(arrowstyle='->',color=blue,lw=1.5))
a.text(-.1,3.45,'Uniform upward line load',color=blue,fontsize=10,ha='center');a.text(-.1,3.22,'20 kN/m; 10 kN total',color=blue,fontsize=10,ha='center')
text3(0,.5,.25,'Web',ha='center',fontsize=11);text3(.23,.28,0,'Base',fontsize=11)
a.annotate('Shared nodes',xy=P(0,.4,0),xytext=(-1.35,.9),fontsize=10,color=navy,arrowprops=dict(arrowstyle='->',color=navy))
a.annotate('0.5 m height',xy=P(0,.25,.5),xytext=(-1.45,2.0),fontsize=10,color=navy,arrowprops=dict(arrowstyle='->',color=navy))
for v,label in [((.27,0,0),'x'),((0,.28,0),'y'),((0,0,.24),'z')]:
 origin=np.array([-1.2,-.9]);vec=P(*v);a.annotate('',xy=origin+vec,xytext=origin,arrowprops=dict(arrowstyle='->',color=navy));a.text(*(origin+vec),label,fontsize=10)
a.set(xlim=(-1.8,2.8),ylim=(-1.2,3.65));a.set_title('(a) Geometry, load and restraints',loc='left',fontsize=12,fontweight='bold',color=navy)
# Plan: x horizontal, y vertical, direct coordinates.
b.add_patch(Rectangle((-.5,0),1,1,facecolor='#eef5fa',edgecolor=navy,lw=1.2))
b.add_patch(Rectangle((-.1,.15),.2,.7,facecolor='#f1d49f',alpha=.6,edgecolor='#c27e20',linestyle='--',lw=1.4))
for x in [-.5,.5]:
 b.plot([x,x],[0,1],color=red,lw=3)
 for y in np.linspace(0,1,11):b.plot([x,x-.035],[y,y-.025],color=red,lw=.8)
b.plot([0,0],[.25,.75],color=navy,lw=3);b.scatter([0,0],[.25,.75],s=40,c=navy,zorder=4)
b.annotate('Web footprint',xy=(0,.52),xytext=(.16,.57),fontsize=10,color=navy,arrowprops=dict(arrowstyle='->',color=navy))
b.annotate('Junction endpoint',xy=(0,.75),xytext=(.14,.92),fontsize=9,color=navy,arrowprops=dict(arrowstyle='->',color=navy))
def dim(x,y,X,Y,label,tx,ty):
 b.annotate('',xy=(X,Y),xytext=(x,y),arrowprops=dict(arrowstyle='<->',color=navy,lw=.8));b.text(tx,ty,label,fontsize=9,color=navy,ha='center',va='center')
dim(-.5,-.1,.5,-.1,'1.0 m',0,-.15)
for lo,hi,label in [(0,.25,'0.25'),(.25,.75,'0.50'),(.75,1,'0.25')]:dim(-.65,lo,-.65,hi,label,-.77,(lo+hi)/2)
b.text(-.77,1.08,'m',fontsize=9,color=navy,ha='center')
b.annotate('Monitored region',xy=(-.1,.38),xytext=(-.46,-.3),fontsize=10,color='#94601c',arrowprops=dict(arrowstyle='->',color='#94601c'))
b.text(-.46,-.39,r'$|x|\leq0.1$, $0.15\leq y\leq0.85$, $z\leq0.1$ m',fontsize=9,color=navy)
b.annotate('',xy=(.70,.2),xytext=(.70,0),arrowprops=dict(arrowstyle='->',color=navy));b.text(.74,.17,'y',fontsize=10)
b.annotate('',xy=(.9,0),xytext=(.70,0),arrowprops=dict(arrowstyle='->',color=navy));b.text(.9,-.06,'x',fontsize=10)
b.set(xlim=(-.95,1.),ylim=(-.48,1.2),aspect='equal');b.set_title('(b) Plan view at z = 0',loc='left',fontsize=12,fontweight='bold',color=navy)
fig.text(.5,.08,'Red edges: all three translations and all three rotations fixed. Other external edges have no prescribed restraints.',ha='center',fontsize=10,color=red)
fig.text(.5,.035,'Base: 1 x 1 m | Web: 0.5 x 0.5 m | Both plates: 10 mm aluminium | Schematics; no weld or fillet modeled',ha='center',fontsize=10,color=navy)
fig.subplots_adjust(left=.035,right=.98,top=.9,bottom=.18,wspace=.08)
fig.savefig(out/'figures/tjoint_geometry.pdf');fig.savefig(out/'figures/tjoint_geometry.png',dpi=200);plt.close(fig)
