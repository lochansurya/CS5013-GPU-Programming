#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

# Paths
RESULTS_FILE = "results/results_2_2.txt"
PLOTS_DIRECTORY = "plots"

# Data containers
matrix_file, shapeA, shapeB = None, None, None
trial_numbers, tile_widths, times = [], [], []
global_reads, global_writes = [], []
shared_reads, shared_writes = [], []

# Parse file
with open(RESULTS_FILE, "r") as f:
    lines = f.readlines()

capture_trial = False
for line in lines:
    line = line.strip()
    if line.startswith("Matrix:"):
        matrix_file = line.split(":")[1].strip()
    elif line.startswith("Shape(A)"):
        shapeA = line.split("=")[1].strip()
    elif line.startswith("Shape(B)"):
        shapeB = line.split("=")[1].strip()
    elif line.startswith("Trial No."):
        trial_no = int(line.split()[2])
        trial_numbers.append(trial_no)
        capture_trial = True
    elif capture_trial and line.startswith("TileWidth:"):
        tw = int(line.split(":")[1].strip())
        tile_widths.append(tw)
    elif capture_trial and line.startswith("Kernel Execution Time:"):
        t = int(line.split(":")[1].strip().split()[0])
        times.append(t)
        capture_trial = False  # done with summary
    elif line.startswith("Global Memory Reads"):
        global_reads.append(int(line.split(":")[1].strip()))
    elif line.startswith("Global Memory Writes"):
        global_writes.append(int(line.split(":")[1].strip()))
    elif line.startswith("Shared Memory Reads"):
        shared_reads.append(int(line.split(":")[1].strip()))
    elif line.startswith("Shared Memory Writes"):
        shared_writes.append(int(line.split(":")[1].strip()))

# Sanity check
n = len(trial_numbers)
if not (n == len(tile_widths) == len(times) == len(global_reads) == len(global_writes) == len(shared_reads) == len(shared_writes)):
    raise RuntimeError("Data length mismatch across collected metrics.")

# Make plots directory
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

# === Plot 1: Execution Time vs Tile Width ===
plt.figure(figsize=(10, 6))
plt.plot(tile_widths, times, marker="o", linestyle="-", color="b", label="Execution Time (µs)")
for trial, tw, t in zip(trial_numbers, tile_widths, times):
    plt.text(tw, t + 1, f"T{trial}", ha="center", fontsize=9)

plt.xlabel("Tile Width")
plt.ylabel("Kernel Execution Time (µs)")
plt.title("Problem 2_0: Runtime vs Tile Width")
plt.legend()
plt.grid(True)

if matrix_file and shapeA and shapeB:
    matrix_base = os.path.splitext(os.path.basename(matrix_file))[0]
    shapeA_str = shapeA.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    shapeB_str = shapeB.replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
    plot_file_time = os.path.join(PLOTS_DIRECTORY, f"problem2_0_time_{matrix_base}_A{shapeA_str}_B{shapeB_str}.png")
else:
    plot_file_time = os.path.join(PLOTS_DIRECTORY, "problem2_0_time.png")

plt.savefig(plot_file_time)
plt.close()

# === Plot 2: Memory Reads/Writes vs Tile Width ===
plt.figure(figsize=(10, 6))
plt.plot(tile_widths, global_reads, marker="o", label="Global Reads")
plt.plot(tile_widths, global_writes, marker="s", label="Global Writes")
plt.plot(tile_widths, shared_reads, marker="^", label="Shared Reads")
plt.plot(tile_widths, shared_writes, marker="d", label="Shared Writes")

plt.xlabel("Tile Width")
plt.ylabel("Memory Access Count")
plt.title("Problem 2_0: Memory Access vs Tile Width")
plt.legend()
plt.grid(True)

if matrix_file:
    plot_file_mem = os.path.join(PLOTS_DIRECTORY, f"problem2_0_memory_{matrix_base}_A{shapeA_str}_B{shapeB_str}.png")
else:
    plot_file_mem = os.path.join(PLOTS_DIRECTORY, "problem2_0_memory.png")

plt.savefig(plot_file_mem)
plt.show()
plt.close()

# Print parsed data
print("Parsed Data:")
for t, tw, tm, gr, gw, sr, sw in zip(trial_numbers, tile_widths, times, global_reads, global_writes, shared_reads, shared_writes):
    print(f"Trial {t}: TileWidth={tw}, Time={tm} µs, "
          f"GlobalR={gr}, GlobalW={gw}, SharedR={sr}, SharedW={sw}")

print(f"\nPlots saved to:\n  {plot_file_time}\n  {plot_file_mem}")
