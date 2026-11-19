from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
out=Path(__file__).resolve().parents[1]/'figures'
fig,ax=plt.subplots(figsize=(10,9.2));ax.set(xlim=(0,10),ylim=(0,9.2));ax.axis('off')
navy='#183b56'; blue='#eef5fa'; green='#eef6ef'
def box(x,y,w,h,title,body,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035',linewidth=.8,edgecolor=navy,facecolor=color))
 ax.text(x+w/2,y+h-.17,title,ha='center',va='top',fontsize=10,fontweight='bold',color=navy)
 ax.text(x+w/2,y+h-.43,body,ha='center',va='top',fontsize=9,linespacing=1.35)
def arrow(x,y,X,Y,dashed=False):
 ax.annotate('',xy=(X,Y),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=navy,lw=1.1,linestyle='--' if dashed else '-'))
box(.15,7.72,9.7,1.3,'INPUTS AND DESIGN VARIABLES','Fixed aerodynamic source forces (MATLAB); wing geometry and materials\nLayout: rib count {2, 8, 16, 25} and two spar positions within prescribed bounds\nThickness: metal faces and core; prescribed thickness limits')
ax.text(.2,7.39,'A. Layout screening and comparison',fontsize=11,fontweight='bold',color=navy)
ax.text(5.45,7.39,'B. Repeated structural evaluator',fontsize=11,fontweight='bold',color=navy)
box(.2,5.94,4.1,1.13,'Extra-coarse screening','One CCD stage per rib count; size samples\nFit response surface; verify its candidate')
box(.2,4.75,4.1,.86,'Retain evaluated candidates','Best accepted layout at each rib count')
box(.2,3.34,4.1,1.08,'Independent fine sizing','Four selected layouts; uniform 10 mm starts\nRepeat thickness sizing on the fine mesh')
box(.2,2.02,4.1,.99,'Compare accepted fine masses','Report lightest evaluated candidate\nFailed / incomplete cases remain identified')
for y,Y in [(5.94,5.61),(4.75,4.42),(3.34,3.01)]:arrow(2.25,y,2.25,Y)
box(5.45,6.05,4.35,1.02,'Generate model and transfer loads','Layout + mesh -> conservative force mapping\nGeometry / transfer / iteration control: Python')
box(5.45,4.85,4.35,.9,'Stress-based thickness sizing','Static solve -> filtered stress -> thickness update')
box(5.45,3.55,4.35,1.,'Buckling reinforcement when needed','Buckling solve -> core growth / face repair\nRepeat with stress rechecking')
box(5.45,2.02,4.35,1.23,'Check stopping and acceptance','Stress, buckling and thickness criteria satisfied:\naccepted mass + thicknesses; otherwise flag failure\nStatic / buckling solutions: ANSYS MAPDL')
for y,Y in [(6.05,5.75),(4.85,4.55),(3.55,3.25)]:arrow(7.62,y,7.62,Y)
arrow(4.3,6.5,5.45,6.5);ax.plot([4.3,4.85,4.85],[3.88,3.88,6.25],color=navy,lw=1.1);arrow(4.85,6.25,5.45,6.25)
ax.text(4.87,6.65,'evaluate',ha='center',fontsize=8)
ax.text(4.55,4.03,'call',ha='center',fontsize=8)
ax.text(5.45,1.7,'Evaluator returns accepted records or failure status to stage A.',fontsize=8.5,color=navy)
arrow(.45,7.72,.45,7.48);arrow(.45,7.23,.45,7.08);arrow(9.55,7.72,9.55,7.48);arrow(9.55,7.23,9.55,7.08)
box(.2,.18,9.6,1.11,'SEPARATE SUPPORTING ASSESSMENT','Plate / T-joint implementation checks; 16-rib local-stress, radius and thick-panel studies\nEvidence applies to the tested models and assumptions, not automatically to the lightest candidate',green)
arrow(2.25,2.02,2.25,1.31,True)
ax.text(.35,1.49,'Assess selected models',fontsize=8,color=navy)
ax.text(5,0,'One-way loading: structural deformation is not fed back to aerodynamics.',ha='center',fontsize=9,fontweight='bold',color=navy)
fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.99)
fig.savefig(out/'workflow.pdf');fig.savefig(out/'workflow.png',dpi=200);plt.close(fig)
