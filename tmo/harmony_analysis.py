import numpy as np
from math import gcd, log2

def format_ratio(ratio):
    """Convert ratio to 12*log2(ratio) format with 2 decimal places"""
    return f"{12 * log2(ratio):.2f}"

# Number of points along each axis
N = 20

# Create coordinates: i, j in range(N)
x = np.arange(N)
y = np.arange(N)

# Create the N x N meshgrid
X, Y = np.meshgrid(x, y)

# Filter: keep only points where x and y are coprimes and y/x in [0.5, 2]
points = []
frequency_map = {}  # Map: formatted frequency -> point

for i in range(N):
    for j in range(N):
        x_val = int(X[j, i])
        y_val = int(Y[j, i])
        
        # Skip if x is 0
        if x_val == 0:
            continue
        
        # Check if x and y are coprimes (gcd = 1)
        if gcd(x_val, y_val) != 1:
            continue
        
        # Check if y/x is in range [0.5, 2]
        ratio = y_val / x_val
        if not (0.5 <= ratio <= 2):
            continue
        
        # Calculate harmony: log2(1/(x*y))
        harmony = log2(1 / (x_val * y_val))
        
        # Filter: keep only points where harmony > -7
        if harmony > -7:
            point_dict = {'x': x_val, 'y': y_val, 'harmony': harmony, 'ratio': ratio}
            points.append(point_dict)
            
            # Create frequency map with formatted key
            freq_key = f"{ratio:.5f}"
            if freq_key not in frequency_map:
                frequency_map[freq_key] = point_dict

# Calculate frequencies
freqs = [p['y']/p['x'] for p in points]

# Find products of frequency pairs that match existing frequencies
threshold = 1e-4
mylist = []

for p1 in points:
    for p2 in points:
        f = (p1['y']/p1['x']) * (p2['y']/p2['x'])
        
        # Check if f matches any frequency in freqs with tolerance
        for freq in freqs:
            if abs(f - freq) < threshold:
                # Find the matched point using the frequency map
                freq_key = f"{freq:.5f}"
                matched_point = frequency_map.get(freq_key)
                
                mylist.append({
                    'product': f,
                    'freq_match': freq,
                    'p1': (p1['x'], p1['y']),
                    'p2': (p2['x'], p2['y']),
                    'ratio_p1': p1['ratio'],
                    'ratio_p2': p2['ratio'],
                    'matched_point': matched_point
                })
                break

# Write results to file
output_file = 'harmony_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HARMONY ANALYSIS RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Grid size: N = {N}\n")
    f.write(f"Harmony filter: harmony > -7 (log2(1/(x*y)) > -7)\n")
    f.write(f"Coprime filter: gcd(x,y) = 1\n")
    f.write(f"Ratio filter: 0.5 <= y/x <= 2\n\n")
    
    f.write("=" * 80 + "\n")
    f.write(f"FILTERED POINTS: {len(points)} points\n")
    f.write("=" * 80 + "\n\n")
    
    for i, p in enumerate(points, 1):
        f.write(f"{i:3d}. ({p['x']:2d}, {p['y']:2d}) - harmony: {p['harmony']:8.4f}, 12*log2(ratio): {format_ratio(p['ratio']):>7s}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"FREQUENCY PRODUCT ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Unique frequencies: {len(set([round(freq, 6) for freq in freqs]))} distinct ratios\n")
    f.write(f"Threshold for matching: {threshold}\n")
    f.write(f"Total matches found: {len(mylist)}\n\n")
    
    f.write("MATCHING PRODUCTS (sorted by product value):\n")
    f.write("Format: product_value | p1=(x1,y1) ratio=r1 | p2=(x2,y2) ratio=r2 | matched_point=(x,y) ratio=r\n")
    f.write("-" * 80 + "\n\n")
    
    # Sort by product value
    sorted_mylist = sorted(mylist, key=lambda x: x['product'])
    
    for i, match in enumerate(sorted_mylist, 1):
        matched_pt = match['matched_point']
        matched_pt_str = f"({matched_pt['x']}, {matched_pt['y']}) 12*log2(r)={format_ratio(matched_pt['ratio']):>7s}" if matched_pt else "N/A"
        
        f.write(f"{i:4d}. product={match['product']:10.6f} | ")
        f.write(f"p1=({int(match['p1'][0])}, {int(match['p1'][1])}) 12*log2(r)={format_ratio(match['ratio_p1']):>7s} | ")
        f.write(f"p2=({int(match['p2'][0])}, {int(match['p2'][1])}) 12*log2(r)={format_ratio(match['ratio_p2']):>7s} | ")
        f.write(f"matched={matched_pt_str}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Filtered points (harmony > -7): {len(points)}\n")
    f.write(f"Unique frequency ratios: {len(set([round(freq, 6) for freq in freqs]))}\n")
    f.write(f"Frequency products that match existing ratios: {len(mylist)}\n")
    f.write(f"Matching ratio: {len(mylist) / (len(points) ** 2) * 100:.2f}% of all pairs\n")

print(f"Results written to: {output_file}")
print(f"\nSummary:")
print(f"  Filtered points: {len(points)}")
print(f"  Unique frequencies: {len(set([round(freq, 6) for freq in freqs]))}")
print(f"  Matching products: {len(mylist)}")
print(f"  Matching ratio: {len(mylist) / (len(points) ** 2) * 100:.2f}% of all pairs")
