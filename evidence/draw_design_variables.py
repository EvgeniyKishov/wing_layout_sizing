from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle,ConnectionPatch
import numpy as np
out=Path(__file__).resolve().parents[1]/'figures'
fig,ax=plt.subplots(figsize=(10,6.2));ax.set(xlim=(0,10),ylim=(0,6.2));ax.axis('off')
navy='#183b56';orange='#d17a24';green='#319374'
def txt(x,y,t,size=10,**kw):ax.text(x,y,t,fontsize=size,color=navy,**kw)
def double(x,y,X,Y):ax.annotate('',xy=(X,Y),xytext=(x,y),arrowprops=dict(arrowstyle='<->',lw=.9,color=navy))
txt(.2,5.95,'(a) Layout variables: prescribed wing outline',11,fontweight='bold')
txt(5.35,5.95,'(b) Element thickness variables',11,fontweight='bold')
# Plan-view span runs left to right; chord coordinate increases downward.
x0,x1=.55,4.45
def le(x):return 5.03-.35*(x-x0)/(x1-x0)
def te(x):return 2.15+.65*(x-x0)/(x1-x0)
ax.add_patch(Polygon([(x0,le(x0)),(x1,le(x1)),(x1,te(x1)),(x0,te(x0))],facecolor='#f1f5f8',edgecolor=navy,lw=1.3))
for x in np.linspace(x0,x1,7):ax.plot([x,x],[le(x),te(x)],color='#879bad',lw=.9)
for fraction,color in [(.22,orange),(.70,green)]:ax.plot([x0,x1],[le(x0)-fraction*(le(x0)-te(x0)),le(x1)-fraction*(le(x1)-te(x1))],color=color,lw=2)
txt(1.7,5.25,'Leading edge',9);txt(1.7,2.07,'Trailing edge',9)
txt(.25,3.5,'Root',9,rotation=90,ha='center');txt(4.66,3.65,'Tip',9,rotation=90,ha='center')
txt(.78,4.22,r'Front spar: $\xi_1$',9);txt(.78,2.65,r'Rear spar: $\xi_2$',9)
x=3.8;double(x,le(x),x,te(x));txt(x-.12,3.65,r'$c(y)$',10,ha='right')
ax.annotate('',xy=(4.4,1.76),xytext=(.6,1.76),arrowprops=dict(arrowstyle='->',color=navy));txt(2.5,1.52,r'Spanwise direction $y$',9,ha='center')
ax.add_patch(Rectangle((2.5,3.55),.42,.42,facecolor='#f8d39b',edgecolor=orange,lw=1.4))
ax.annotate('',xy=(5.5,4.9),xytext=(2.92,3.97),arrowprops=dict(arrowstyle='->',color=orange,lw=1,connectionstyle='arc3,rad=-.25'))
txt(.3,1.18,r'$x_{\mathrm{spar},j}-x_{\mathrm{LE}}=\xi_j c(y)$',11)
txt(.3,.86,r'$n_r$: equally spaced ribs, including root and tip',9)
txt(.3,.58,'Illustrated rib count is schematic.',9)
# Mesh patch and representative area.
for i in range(4):
 for j in range(2):
  ax.add_patch(Rectangle((5.55+i*.65,4.45+j*.45),.65,.45,facecolor='#f8d39b' if (i,j)==(1,0) else '#eef5fa',edgecolor='#879bad',lw=.8))
txt(6.52,4.66,r'$A_i$',11,ha='center',va='center');txt(8.35,4.9,'Representative\nshell mesh',9,va='center')
ax.annotate('',xy=(6.5,3.88),xytext=(6.52,4.45),arrowprops=dict(arrowstyle='->',color=orange))
# Enlarged section: dimensions illustrative, not initial layer fractions.
for y,h,color in [(2.65,.3,'#9ebbd0'),(2.95,.64,'#f1d49f'),(3.59,.3,'#9ebbd0')]:ax.add_patch(Rectangle((5.55,y),2.6,h,facecolor=color,edgecolor=navy,lw=.9))
for y,h,label in [(2.65,.3,r'$f_i$'),(2.95,.64,r'$c_i$'),(3.59,.3,r'$f_i$')]:
 double(8.3,y,8.3,y+h);txt(8.45,y+h/2,label,11,va='center')
double(9.25,2.65,9.25,3.89);txt(9.5,3.27,r'$t_i$',11,va='center')
txt(6.85,3.74,'Metal face',9,ha='center',va='center');txt(6.85,3.27,'Core',9,ha='center',va='center');txt(6.85,2.80,'Metal face',9,ha='center',va='center')
txt(5.55,2.25,r'$t_i=2f_i+c_i$; equal metal-face thicknesses',10)
txt(5.55,1.84,'Layer definitions apply to skins, spars and ribs.',9)
txt(5.55,1.51,'Section and mesh are not to scale.',9)
txt(5.55,1.18,'FSD: fixed layer fractions.',9)
txt(5.55,.86,'Reinforcement: core growth and face repair.',9)
ax.plot([.2,9.8],[.35,.35],color='#879bad',lw=.7)
txt(5,.12,'Objective: semi-wing mass | Assessment: filtered face stress, linear buckling and thickness bounds',9,ha='center')
txt(5,-.1,'One prescribed aerodynamic load case',9,ha='center')
fig.subplots_adjust(left=.01,right=.99,bottom=.055,top=.99)
fig.savefig(out/'design_variables.pdf');fig.savefig(out/'design_variables.png',dpi=200);plt.close(fig)
