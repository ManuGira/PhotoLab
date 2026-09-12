import numpy as np
import matplotlib.pyplot as plt
from math import gcd, log2

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

# Transform coordinates: (x, y) -> (12*log2(y/x), 1/(x*y))
transformed_x = []
transformed_y = []

for x_val, y_val in zip(filtered_x, filtered_y):
    new_x = 12 * log2(y_val / x_val)
    new_y = 1 / (x_val * y_val)
    transformed_x.append(new_x)
    transformed_y.append(new_y)

# Create plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the scatter points
ax.scatter(transformed_x, transformed_y, s=50, color='red', alpha=0.9, zorder=5, label='Transformed points')

# Draw circles for transformed points (green color, same radius in new coordinates)
for tx_val, ty_val in zip(transformed_x, transformed_y):
    radius = 0.05  # Small fixed radius in transformed space for visibility
    circle = plt.Circle((tx_val, ty_val), radius, fill=False, edgecolor='green', alpha=0.8, linewidth=2)
    ax.add_patch(circle)

ax.set_xlabel("12*log₂(y/x)", fontsize=12)
ax.set_ylabel("1/(x*y)", fontsize=12)
ax.set_title(f"Transformed Meshgrid: {len(filtered_x)} filtered points\nTransformation: (x,y) → (12*log₂(y/x), 1/(x*y))", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# Save to file
output_path = 'meshgrid_transformed.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
print(f"\nTransformed points ({len(transformed_x)}):")
for i, (x_val, y_val, tx_val, ty_val) in enumerate(zip(filtered_x, filtered_y, transformed_x, transformed_y)):
    print(f"  ({x_val}, {y_val}) → ({tx_val:.2f}, {ty_val:.4f})")
