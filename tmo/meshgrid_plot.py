import numpy as np
import matplotlib.pyplot as plt

# Number of points along each axis
N = 10

# Create coordinates: i, j in range(N)
x = np.arange(N)
y = np.arange(N)

# Create the N x N meshgrid
X, Y = np.meshgrid(x, y)

# Plot the points
plt.figure(figsize=(8, 8))
plt.scatter(X, Y, s=100, alpha=0.6)

plt.xlabel("x (i)", fontsize=12)
plt.ylabel("y (j)", fontsize=12)
plt.title(f"{N} × {N} 2D Meshgrid", fontsize=14)
plt.axis("equal")
plt.grid(True, alpha=0.3)

# Save to file
output_path = 'meshgrid_plot.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
