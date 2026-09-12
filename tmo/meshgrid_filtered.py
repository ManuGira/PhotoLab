import numpy as np
import matplotlib.pyplot as plt
from math import gcd

# Number of points along each axis
N = 10

# Create coordinates: i, j in range(N)
x = np.arange(N)
y = np.arange(N)

# Create the N x N meshgrid
X, Y = np.meshgrid(x, y)

# Filter: keep only points where x and y are coprimes and y/x in [0.5, 2]
filtered_x = []
filtered_y = []

for i in range(N):
    for j in range(N):
        x_val = X[j, i]
        y_val = Y[j, i]
        
        # Skip if x is 0 (can't compute ratio or gcd meaningfully)
        if x_val == 0:
            continue
        
        # Check if x and y are coprimes (gcd = 1)
        if gcd(x_val, y_val) != 1:
            continue
        
        # Check if y/x is in range [0.5, 2]
        ratio = y_val / x_val
        if not (0.5 <= ratio <= 2):
            continue
        
        filtered_x.append(x_val)
        filtered_y.append(y_val)

# Create plot
fig, ax = plt.subplots(figsize=(12, 12))

# Add axis lines (x=0 and y=0)
ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)

# Add curves y = k/x for k in range(1, 10)
x_range = np.linspace(0.1, 10, 1000)
colors_curves = plt.cm.Blues(np.linspace(0.3, 0.9, 9))
for k in range(1, 10):
    y_curve = k / x_range
    ax.plot(x_range, y_curve, color=colors_curves[k-1], alpha=0.5, linewidth=1, label=f'y = {k}/x')

# Add reference lines
x_line = np.linspace(0, 10, 100)
ax.plot(x_line, x_line, 'g--', linewidth=1.5, alpha=0.6, label='y = x')
ax.plot(x_line, 2*x_line, 'b--', linewidth=1.5, alpha=0.6, label='y = 2x')
ax.plot(x_line, x_line/2, 'r--', linewidth=1.5, alpha=0.6, label='y = x/2')

# Plot the scatter points
ax.scatter(filtered_x, filtered_y, s=50, color='red', alpha=0.9, zorder=5, label='Filtered points')
ax.scatter(X, Y, s=20, alpha=0.2, color='gray', label='All points')

# Draw circles for filtered points (green color)
for x_val, y_val in zip(filtered_x, filtered_y):
    radius = 1 / (x_val * y_val)
    circle = plt.Circle((x_val, y_val), radius, fill=False, edgecolor='green', alpha=0.8, linewidth=2)
    ax.add_patch(circle)

# Set viewport cropped around points
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.set_xlabel("x (i)", fontsize=12)
ax.set_ylabel("y (j)", fontsize=12)
ax.set_title(f"Filtered Meshgrid: Circles with radius = 1/(x*y)\n({len(filtered_x)} points out of {N*N})", fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8)

# Save to file
output_path = 'meshgrid_filtered.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
