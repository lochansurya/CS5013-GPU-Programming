import os
import re
import matplotlib.pyplot as plt

# Paths
RESULTS_FILE = "results/results_3_1.txt"
PLOTS_DIRECTORY = "plots"
PLOT_FILE = os.path.join(PLOTS_DIRECTORY, "problem3_1_transpose.png")
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)

# Regex patterns
trial_pattern = re.compile(r"Trial No\. (\d+)")
pair_pattern = re.compile(r"grid_x, block_x: \((\d+),(\d+)\)")
time_pattern = re.compile(r"Kernel Execution Time: (\d+) microseconds")

pairs = []
exec_times = []

with open(RESULTS_FILE, "r") as f:
    grid_x = block_x = None
    for line in f:
        pair_match = pair_pattern.search(line)
        time_match = time_pattern.search(line)

        if pair_match:
            grid_x = int(pair_match.group(1))
            block_x = int(pair_match.group(2))
        if time_match and grid_x is not None and block_x is not None:
            time = int(time_match.group(1))
            pairs.append(f"({grid_x},{block_x})")
            exec_times.append(time)
            grid_x = block_x = None  # reset for next trial

# Plot bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(pairs, exec_times, color="skyblue", edgecolor=None)

plt.xlabel("Grid x, Block x")
plt.ylabel("Execution Time (µs)")
plt.title("Matrix Transpose Execution Time vs Grid/Block Dimensions")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Label bars
for bar, time in zip(bars, exec_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{time}", ha="center", va="bottom", fontsize=9)

# Highlight fastest runtime
min_index = exec_times.index(min(exec_times))
bars[min_index].set_color("lightgreen")

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.show()
plt.close()

print(f"Bar chart saved to {PLOT_FILE}")
print(f"Fastest execution: {pairs[min_index]} -> {exec_times[min_index]} µs")
