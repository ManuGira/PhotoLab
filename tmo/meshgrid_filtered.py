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

# Plot the filtered points
plt.figure(figsize=(8, 8))
plt.scatter(filtered_x, filtered_y, s=100, alpha=0.6, color='red')
plt.scatter(X, Y, s=30, alpha=0.2, color='gray', label='All points')

plt.xlabel("x (i)", fontsize=12)
plt.ylabel("y (j)", fontsize=12)
plt.title(f"Filtered Meshgrid: Coprimes with 0.5 ≤ y/x ≤ 2\n({len(filtered_x)} points out of {N*N})", fontsize=14)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()

# Save to file
output_path = 'meshgrid_filtered.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
