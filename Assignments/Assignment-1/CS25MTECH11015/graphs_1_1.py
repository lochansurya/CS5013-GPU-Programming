#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

# Paths
RESULTS_FILE = "results/results_1_1.txt"
PLOTS_DIRECTORY = "plots"

Ms, Ns, times = [], [], []
shapeA, shapeB = None, None

with open(RESULTS_FILE, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("Shape(A)"):
        shapeA = line.split("=")[1].strip()
    if line.startswith("Shape(B)"):
        shapeB = line.split("=")[1].strip()

    if line.startswith("#Thread Blocks(M):"):
        M = int(line.split(":")[1].strip())
        N = int(lines[i+1].split(":")[1].strip())  # #Threads Per Block(N)
        t_line = lines[i+2]
        time_us = int(t_line.split(":")[1].strip().split()[0])
        Ms.append(M)
        Ns.append(N)
        times.append(time_us)

# Build labels for X-axis
labels = [f"({m},{n})" for m, n in zip(Ms, Ns)]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(labels, times, color='skyblue')
plt.xlabel("(GridBlocks M, ThreadsPerBlock N)")
plt.ylabel("Kernel Execution Time (µs)")
plt.title("Problem 1_1: Runtime vs (M, N) pairs")
plt.xticks(rotation=45)
plt.tight_layout()

# Ensure plots directory exists
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

# Build filename including matrix shapes
if shapeA and shapeB:
    shapeA_str = shapeA.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    shapeB_str = shapeB.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    plot_file = os.path.join(PLOTS_DIRECTORY, f"problem1_1_A{shapeA_str}_B{shapeB_str}.png")
else:
    plot_file = os.path.join(PLOTS_DIRECTORY, "problem1_1.png")

plt.savefig(plot_file)
plt.show()

print(f"Plot saved to {plot_file}")
