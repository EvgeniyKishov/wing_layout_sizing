"""Saved fine 16-rib thickness fields: no solver runs or interpolation."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm,Normalize
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/fokker_fine_layout_comparison_20261016/evaluations/ribs_16'
OUT=ROOT/'article_results/selected_wing_thickness_20261016'

def main():
    OUT.mkdir(exist_ok=True)
    h=json.loads((RUN/'history.json').read_text())
    rows=[next(r for r in reversed(h) if r['phase']=='fsd'),h[-1]]
    paths=[RUN/'geometry/mesh.npz']+[RUN/r['folder']/'fields.npz' for r in rows]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    mesh=np.load(paths[0]);data=[np.load(p) for p in paths[1:]]
    groups=mesh['groups'];pol=mesh['xyz'][mesh['corners']-1]
    assert all(np.all(data[1][k]>=data[0][k]) for k in ['face_m','core_m'])
    np.testing.assert_allclose(data[0]['total_m'],2*data[0]['face_m']+data[0]['core_m'])
    np.testing.assert_allclose(data[1]['total_m'],2*data[1]['face_m']+data[1]['core_m'])
    components={'skin':([0,1],[1,0]),'spars':([2,3],[1,2]),'ribs':(list(range(4,20)),[0,2])}
    stats={}
    for comp,(ids,dims) in components.items():
        stats[comp]={};selection=np.isin(groups,ids)
        for key,label in [('total_m','Total sandwich thickness'),('face_m','One metal face-sheet thickness'),('core_m','Core thickness')]:
            values=[d[key]*1000 for d in data];lo=min(v[selection].min() for v in values);hi=max(v[selection].max() for v in values)
            logarithmic=hi/lo>8
            norm=LogNorm(lo,hi) if logarithmic else Normalize(0 if hi==lo else lo,hi*2 if hi==lo else hi)
            ticks=[x for x in [.125,.25,.5,.625,1,2,5,10,20,50,100,150] if lo-1e-8<=x<=hi+1e-8] if logarithmic else None
            if comp=='ribs':fig,axes=plt.subplots(8,4,figsize=(15,14),layout='constrained')
            else:fig,axes=plt.subplots(2,2,figsize=(13,6.5),layout='constrained')
            for j,g in enumerate(ids):
                mask=groups==g
                name=['Upper skin','Lower skin','Front spar (10%)','Rear spar (70%)'][g] if g<4 else f'Rib {g-3}, y={mesh["centers"][mask,1].mean():.2f} m'
                for col,stage in enumerate(['After FSD','After buckling reinforcement']):
                    ax=axes[j//2,2*(j%2)+col] if comp=='ribs' else axes[j,col]
                    artist=PolyCollection(pol[mask][:,:,dims],array=values[col][mask],cmap='viridis',norm=norm,edgecolors='none',rasterized=True)
                    ax.add_collection(artist);ax.autoscale_view();ax.margins(.02)
                    ax.set_title(f'{name} | {"FSD" if col==0 else "Reinforced"}\n{values[col][mask].min():.3f}–{values[col][mask].max():.3f} mm',fontsize=9 if comp=='ribs' else 11)
                    ax.set_xlabel('Chord x (m)' if comp=='ribs' else 'Span y (m)',fontsize=9)
                    ax.set_ylabel('Chord x (m)' if comp=='skin' else 'Height z (m)',fontsize=9)
                    ax.tick_params(labelsize=8);ax.grid(alpha=.12)
                    if comp=='skin':ax.invert_yaxis()
            cb=fig.colorbar(artist,ax=axes.ravel().tolist(),shrink=.82,pad=.015,label=label+' (mm)'+(' — logarithmic scale' if logarithmic else ''),ticks=ticks)
            if logarithmic:cb.ax.yaxis.set_major_formatter(ScalarFormatter());cb.minorticks_off()
            fig.suptitle(f'Selected 16-rib wing: {comp} — {label.lower()}\nFSD iteration 23 (308.84 kg) → reinforced iteration 32 (548.51 kg)',fontsize=14)
            filename=f'{comp}_{key.removesuffix("_m")}_thickness'
            fig.savefig(OUT/(filename+'.png'),dpi=160)
            if key=='total_m':fig.savefig(OUT/(filename+'.pdf'))
            plt.close(fig)
            stats[comp][key]=dict(FSD_min_mm=float(values[0][selection].min()),FSD_max_mm=float(values[0][selection].max()),reinforced_min_mm=float(values[1][selection].min()),reinforced_max_mm=float(values[1][selection].max()),logarithmic_color_scale=bool(logarithmic))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    (OUT/'summary.json').write_text(json.dumps(dict(source_sha256=hashes,statistics=stats,iterations=[r['iteration'] for r in rows],definition='Total = two equal metal faces + lightweight core. Face plots show ONE face. Shared color scale across stages within each figure. Unsmoothed element thicknesses; projected geometry; height enlarged in spar/rib views.'),indent=2))
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
