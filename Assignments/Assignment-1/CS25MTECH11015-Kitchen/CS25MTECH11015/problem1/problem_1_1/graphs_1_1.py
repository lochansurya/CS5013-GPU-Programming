#!/usr/bin/env python
import matplotlib.pyplot as plt

# Path to your concatenated results file
RESULTS_FILE = "results_1_1.txt"

Ms = []
Ns = []
times = []

with open(RESULTS_FILE, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("#Thread Blocks(M):"):
        M = int(line.split(":")[1].strip())
        N = int(lines[i+1].split(":")[1].strip())  # #Threads Per Block(N)
        t_line = lines[i+2]
        # Extract numeric part
        time_us = int(t_line.split(":")[1].strip().split()[0])
        Ms.append(M)
        Ns.append(N)
        times.append(time_us)

# Create X-axis labels as (M,N)
labels = [f"({m},{n})" for m,n in zip(Ms,Ns)]

# Plot
plt.figure(figsize=(12,6))
plt.bar(labels, times, color='green')
plt.xlabel("(GridBlocks M, ThreadsPerBlock N)")
plt.ylabel("Kernel Execution Time (µs)")
plt.title("Problem 1_1: Runtime vs (M, N) pairs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

