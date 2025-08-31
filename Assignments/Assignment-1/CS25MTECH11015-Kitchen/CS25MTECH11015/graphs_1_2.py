#!/usr/bin/env python
import os
import re
import matplotlib.pyplot as plt

# Paths
RESULTS_FILE = "results/results_1_2_2.txt"
PLOTS_DIRECTORY = "plots"

trials, grid_sizes, block_sizes, times = [], [], [], []
shapeA, shapeB, shapeC = None, None, None

with open(RESULTS_FILE, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("Shape(A)"):
        shapeA = line.split("=")[1].strip()
    if line.startswith("Shape(B)"):
        shapeB = line.split("=")[1].strip()
    if line.startswith("Shape(C)"):
        shapeC = line.split("=")[1].strip()

    if line.startswith("2DGrid"):
        g_match = re.search(r"\((\d+),\s*(\d+)\)", line)
        if g_match:
            grid = (int(g_match.group(1)), int(g_match.group(2)))

    if line.startswith("2DBlock"):
        b_match = re.search(r"\((\d+),\s*(\d+)\)", line)
        if b_match:
            block = (int(b_match.group(1)), int(b_match.group(2)))

    if "Kernel Execution Time:" in line:
        m = re.search(r"(\d+)\s+microseconds", line)
        if m:
            time_us = int(m.group(1))
            trials.append(len(trials)+1)
            grid_sizes.append(grid)
            block_sizes.append(block)
            times.append(time_us)

# Build X-axis labels
labels = [f"G{gx}x{gy}_B{bx}x{by}" for (gx,gy), (bx,by) in zip(grid_sizes, block_sizes)]

# Plot
plt.figure(figsize=(14, 6))
bars = plt.bar(labels, times, color='skyblue', edgecolor=None)

# Highlight fastest runtime
min_index = times.index(min(times))
bars[min_index].set_color("lightgreen")

plt.xlabel("2DGrid(X1×Y1) & 2DBlock(X2×Y2)")
plt.ylabel("Kernel Execution Time (µs)")
plt.title("Problem 1_2_0: Runtime vs 2D Grid & Block Sizes")
plt.xticks(rotation=45)
plt.tight_layout()

# Ensure plots directory exists
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

# Build filename
if shapeA and shapeB and shapeC:
    shapeA_str = shapeA.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    shapeB_str = shapeB.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    shapeC_str = shapeC.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    plot_file = os.path.join(PLOTS_DIRECTORY, f"problem1_2_0_A{shapeA_str}_B{shapeB_str}_C{shapeC_str}.png")
else:
    plot_file = os.path.join(PLOTS_DIRECTORY, "problem1_2_0.png")

plt.savefig(plot_file, dpi=300)
plt.show()
plt.close()

print(f"Plot saved to {plot_file}")
print(f"Fastest runtime: {labels[min_index]} -> {times[min_index]} µs")
