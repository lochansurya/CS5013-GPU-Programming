import os
import re
import matplotlib.pyplot as plt

# Paths
RESULTS_FILE = "results/results_4_1.txt"
PLOTS_DIRECTORY = "plots"
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)
PLOT_FILE = os.path.join(PLOTS_DIRECTORY, "problem4_1_tilewidth.png")

# Parse results
tile_widths = []
exec_times = []

with open(RESULTS_FILE, "r") as f:
    content = f.read()

pattern = re.compile(
    r"Trial No\.\s*\d+.*?TileWidth:\s*(\d+).*?Kernel Execution Time:\s*(\d+)",
    re.S
)

matches = pattern.findall(content)
for tile, time in matches:
    tile_widths.append(int(tile))
    exec_times.append(int(time))

# Find optimal tile width
min_time = min(exec_times)
optimal_index = exec_times.index(min_time)
optimal_tile = tile_widths[optimal_index]
print(f"Optimal Tile Width: {optimal_tile} (Execution Time = {min_time} µs)")

# Plot as bar chart with explicit X-axis labels
plt.figure(figsize=(8,6))
bars = plt.bar([str(t) for t in tile_widths], exec_times, color="lightblue", edgecolor=None)

# Highlight fastest execution
bars[optimal_index].set_color("lightgreen")

plt.xlabel("Tile Width")
plt.ylabel("Execution Time (µs)")
plt.title("Execution Time vs Tile Width (Problem 4.1)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Annotate bars with execution times
for bar, time in zip(bars, exec_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{time}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.show()
plt.close()

print(f"Bar chart saved to {PLOT_FILE}")
