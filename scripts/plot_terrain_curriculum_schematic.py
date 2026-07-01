"""Terrain difficulty curriculum schematic for thesis (stair-climbing example).

B2 config: step_height_range = (0.04, 0.16), 10 levels in practice.
Figure shows 5 levels for visual clarity.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

N_LEVELS = 5 # simplified; actual B2 has 10 levels
N_COLS = 7
step_heights = np.linspace(0.04, 0.16, N_LEVELS)
cmap = plt.cm.Blues

# helper

def stair_polygon(x_base, y_base, total_w, total_h, n_steps=3):
    """Return (xs, ys) for a stair outline + base, suitable for ax.fill."""
    step_w = total_w / n_steps
    step_h = total_h / n_steps
    xs = [x_base]
    ys = [y_base]
    for s in range(n_steps):
        xs += [x_base + s * step_w,
               x_base + s * step_w,
               x_base + (s + 1) * step_w]
        ys += [y_base + s * step_h,
               y_base + (s + 1) * step_h,
               y_base + (s + 1) * step_h]
    # close the bottom
    xs += [x_base + n_steps * step_w, x_base + n_steps * step_w, x_base]
    ys += [ys[-1], y_base, y_base]
    return xs, ys


fig, ax = plt.subplots(figsize=(12, 5))
ax.set_aspect('equal')
ax.axis('off')

# two representative profiles (Level 0 and Level 4)
P_X = -4.6 # left edge of profiles
P_W = 1.5 # profile width
P_H = 1.55 # profile height (each occupies ~1.5 rows)

# level 0
xs0, ys0 = stair_polygon(P_X, 0.15, P_W, P_H * (step_heights[0] / step_heights[-1]))
ax.fill(xs0, ys0,
        facecolor=cmap(0.15 + 0.65 * 0 / (N_LEVELS - 1)),
        edgecolor='#445', linewidth=1.2, alpha=0.9, zorder=2)
ax.text(P_X + P_W / 2, -0.05,
        f'h = {step_heights[0]:.2f} m', ha='center', va='top', fontsize=8, color='#444')

# dots separator
ax.text(P_X + P_W / 2, N_LEVELS / 2,
        '⋮', ha='center', va='center', fontsize=15, color='#888')

# level 4
xs4, ys4 = stair_polygon(P_X, N_LEVELS - P_H - 0.1, P_W, P_H)
ax.fill(xs4, ys4,
        facecolor=cmap(0.15 + 0.65 * (N_LEVELS - 1) / (N_LEVELS - 1)),
        edgecolor='#445', linewidth=1.2, alpha=0.9, zorder=2)
ax.text(P_X + P_W / 2, N_LEVELS + 0.1,
        f'h = {step_heights[-1]:.2f} m', ha='center', va='bottom', fontsize=8, color='#444')

# shared label
ax.text(P_X + P_W / 2, N_LEVELS + 0.42,
        'stair profile', ha='center', va='bottom', fontsize=9, color='#444', style='italic')

# level labels
for row in range(N_LEVELS):
    ax.text(-0.15, row + 0.5, f'Level {row}',
            ha='right', va='center', fontsize=8.5)

# grid cells
ax.text(N_COLS / 2, N_LEVELS + 0.2, 'Terrain instances (columns)',
        ha='center', va='bottom', fontsize=9, color='#555', style='italic')

for row in range(N_LEVELS):
    for col in range(N_COLS):
        color = cmap(0.15 + 0.65 * row / (N_LEVELS - 1))
        rect = plt.Rectangle((col, row), 1, 1,
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)

# robot: rightmost column, level 2
robot_row, robot_col = 2, N_COLS - 1

highlight = plt.Rectangle(
    (robot_col, robot_row), 1, 1,
    facecolor=cmap(0.15 + 0.65 * robot_row / (N_LEVELS - 1)),
    edgecolor='#e05c00', linewidth=2.8, zorder=3,
)
ax.add_patch(highlight)
ax.text(robot_col + 0.5, robot_row + 0.5, 'robot',
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=4)

# annotations outside the grid
ANN_X = N_COLS + 0.8
mid_y = robot_row + 0.5

ax.plot([N_COLS, ANN_X], [mid_y, mid_y],
        color='#bbb', lw=1.0, ls='--', zorder=2)

# success -> promoted
ax.annotate('',
            xy=(ANN_X, robot_row + 2.05),
            xytext=(ANN_X, mid_y),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.3), zorder=5)
ax.text(ANN_X + 0.2, (mid_y + robot_row + 2.05) / 2,
        'success -> promoted', ha='left', va='center', fontsize=8.5, color='#2ca02c')

# failure -> demoted
ax.annotate('',
            xy=(ANN_X, robot_row - 1.05),
            xytext=(ANN_X, mid_y),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.3), zorder=5)
ax.text(ANN_X + 0.2, (mid_y + robot_row - 1.05) / 2,
        'failure -> demoted', ha='left', va='center', fontsize=8.5, color='#d62728')

ax.set_xlim(P_X - 0.6, N_COLS + 5.5)
ax.set_ylim(-0.7, N_LEVELS + 0.9)

plt.tight_layout()

for ext in ('pdf', 'png'):
    out = f'scripts/terrain_curriculum_schematic.{ext}'
    plt.savefig(out, bbox_inches='tight', dpi=300)
    print(f'Saved {out}')

plt.show()
