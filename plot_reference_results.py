"""Regenerate sizing-history plots from bundled data without external solvers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def main():
    output = ROOT/'reference_plots'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for n in (8, 16, 25):
        rows = json.loads((ROOT/f'reference_results/fine/ribs_{n:02d}/history.json').read_text())
        for ax, key in zip(axes, ('mass_kg', 'filtered_peak_MPa', 'BLF')):
            pairs = [(i, r[key]) for i, r in enumerate(rows) if key in r]
            ax.plot(*zip(*pairs), label=f'{n} ribs')
        transition = next(i for i,r in enumerate(rows) if r['phase']=='buckling')
        axes[0].plot(transition, rows[transition]['mass_kg'], 'o', color=axes[0].lines[-1].get_color())
    for ax,title in zip(axes, ('Semi-wing mass (kg)', 'Filtered peak stress (MPa)', 'Lowest positive buckling factor')):
        ax.set(xlabel='Solved state index', ylabel=title)
        ax.grid(alpha=.25)
    axes[1].axhline(300, color='black', linestyle='--', linewidth=1)
    axes[2].axhline(1, color='black', linestyle='--', linewidth=1)
    axes[0].legend()
    fig.suptitle('Fine-mesh sizing histories; circles mark the start of buckling reinforcement')
    fig.savefig(output/'fine_sizing_history.png', dpi=180)
    plt.close(fig)
    print(output/'fine_sizing_history.png')

if __name__=='__main__': main()
