"""Generate MTL curriculum sampling proportions figure for thesis."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

phases = ['P0\nBalanced', 'P1\nB2 Step-up', 'P1\nB2 Ramp', 'P2\nAll-recover']

data = {
    'Flat (A1, A2)': [35, 25, 10, 15],
    'Rough (B1)': [35, 35, 10, 25],
    'Stairs (B2)': [15, 25, 75, 40],
    'Gap (C2)': [15, 15,  5, 20],
}

colors = ['#7FC97F', '#FDB462', '#386CB0', '#BEAED4']

fig, ax = plt.subplots(figsize=(8, 3.6))
y = np.arange(len(phases))
left = np.zeros(len(phases))

for (label, vals), color in zip(data.items(), colors):
    vals = np.array(vals)
    ax.barh(y, vals, left=left, color=color, height=0.52,
            edgecolor='white', linewidth=0.8)
    for j, (v, l) in enumerate(zip(vals, left)):
        if v >= 8:
            ax.text(l + v / 2, j, f'{v}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
    left += vals

ax.set_yticks(y)
ax.set_yticklabels(phases, fontsize=10)
ax.set_xlabel('Terrain sampling proportion (%)', fontsize=10)
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.tick_params(axis='x', labelsize=9)

legend_handles = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(colors, data.keys())
]
ax.legend(handles=legend_handles, loc='upper center',
          bbox_to_anchor=(0.5, -0.22),
          fontsize=9, framealpha=0.9, ncol=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out = 'scripts/curriculum_sampling.pdf'
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f'Saved to {out}')

out_png = 'scripts/curriculum_sampling.png'
plt.savefig(out_png, bbox_inches='tight', dpi=300)
print(f'Saved to {out_png}')

plt.show()
