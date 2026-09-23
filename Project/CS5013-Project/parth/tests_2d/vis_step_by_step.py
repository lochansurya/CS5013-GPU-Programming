import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import imageio

"""
Visualization script for step-by-step graph construction or search.
Generates a GIF of the process.
"""

def read_snapshot(filename, dim):
    """Reads a snapshot binary file."""
    with open(filename, 'rb') as f:
        num_nodes = struct.unpack('i', f.read(4))[0]
        
        vecs = np.frombuffer(f.read(num_nodes * dim * 4), dtype=np.float32).reshape(num_nodes, dim)
        adj = np.frombuffer(f.read(num_nodes * 32 * 4), dtype=np.int32).reshape(num_nodes, 32)
        mask = np.frombuffer(f.read(num_nodes * 4), dtype=np.int32)
        
        # Read Search Info
        try:
            has_search_bytes = f.read(4)
            if len(has_search_bytes) == 4:
                has_search = struct.unpack('i', has_search_bytes)[0]
                if has_search:
                    query_vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                    
                    found_count = struct.unpack('i', f.read(4))[0]
                    found_ids = np.frombuffer(f.read(found_count * 4), dtype=np.int32)
                    
                    gt_count = struct.unpack('i', f.read(4))[0]
                    gt_ids = np.frombuffer(f.read(gt_count * 4), dtype=np.int32)
                    
                    return vecs, adj, mask, {'query': query_vec, 'found': found_ids, 'gt': gt_ids}
        except Exception as e:
            pass 
            
    return vecs, adj, mask, None

def plot_graph(vecs, adj, mask, output_path, step_idx, op_info=None):
    """Plots the graph state and search details."""
    plt.figure(figsize=(8, 8))
    plt.title(f"Step {step_idx}")
    
    # Plot edges
    for i in range(len(vecs)):
        if mask[i]: continue
        for nbr in adj[i]:
            if nbr != -1 and not mask[nbr]:
                plt.plot([vecs[i, 0], vecs[nbr, 0]], [vecs[i, 1], vecs[nbr, 1]], 'k-', alpha=0.1, linewidth=0.5)
                
    # Plot nodes
    valid_indices = np.where(mask == 0)[0]
    plt.scatter(vecs[valid_indices, 0], vecs[valid_indices, 1], c='blue', s=20, label='Nodes')
    
    if op_info:
        # Plot Query Vector
        q = op_info['query']
        plt.scatter(q[0], q[1], c='red', marker='*', s=100, label='Query', zorder=10)
        
        # Highlight Found Nodes
        found_ids = op_info['found']
        valid_found = [fid for fid in found_ids if fid != -1 and not mask[fid]]
        if valid_found:
            plt.scatter(vecs[valid_found, 0], vecs[valid_found, 1], facecolors='none', edgecolors='lime', s=80, linewidths=2, label='Found', zorder=5)
            
        # Highlight GT Nodes
        gt_ids = op_info['gt']
        valid_gt = [gid for gid in gt_ids if gid != -1 and not mask[gid]]
        if valid_gt:
             plt.scatter(vecs[valid_gt, 0], vecs[valid_gt, 1], marker='x', c='orange', s=60, label='Ground Truth', zorder=6)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    output_dir = config['paths']['vis_output_dir']
    dim = config['dim']
    
    snapshots = sorted(glob.glob(os.path.join(output_dir, "snapshot_*.bin")))
    
    images = []
    print(f"Found {len(snapshots)} snapshots.")
    
    for i, snap in enumerate(snapshots):
        vecs, adj, mask, op_info = read_snapshot(snap, dim)
        out_img = snap.replace('.bin', '.png')
        plot_graph(vecs, adj, mask, out_img, i+1, op_info)
        images.append(imageio.imread(out_img))
        print(f"Processed {snap}")
        
    # Create GIF
    gif_path = os.path.join(output_dir, "step_by_step.gif")
    imageio.mimsave(gif_path, images, duration=1000, loop=0)
    print(f"Saved GIF to {gif_path}")
    
    # Cleanup PNGs
    for snap in snapshots:
        png_path = snap.replace('.bin', '.png')
        if os.path.exists(png_path):
            os.remove(png_path)
    print("Cleaned up intermediate PNG files.")

if __name__ == "__main__":
    main()
