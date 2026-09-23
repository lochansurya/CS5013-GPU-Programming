import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob

import argparse
import json
import glob

"""
Visualization script for the 2D toy dataset.
Generates animations of graph updates and static plots for queries.
"""

def read_snapshot(filename):
    with open(filename, 'rb') as f:
        num_nodes = struct.unpack('i', f.read(4))[0]
        
        vec_data = f.read(num_nodes * 2 * 4)
        vectors = np.frombuffer(vec_data, dtype=np.float32).reshape(num_nodes, 2)
        
        adj_data = f.read(num_nodes * 32 * 4)
        adj = np.frombuffer(adj_data, dtype=np.int32).reshape(num_nodes, 32)
        
        mask_data = f.read(num_nodes * 4)
        mask = np.frombuffer(mask_data, dtype=np.int32)
        
    return vectors, adj, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    vis_dir = config['paths']['vis_output_dir']
    files = sorted(glob.glob(os.path.join(vis_dir, "snapshot_*.bin")))
    
    if not files:
        print(f"No snapshots found in {vis_dir}")
        return

    # 1. Generate GIF
    print("Generating GIF...")
    fig, ax = plt.subplots(figsize=(12, 12)) # Bigger size
    
    def update(frame):
        ax.clear()
        fname = files[frame]
        vectors, adj, mask = read_snapshot(fname)
        
        active_indices = np.where(mask == 0)[0]
        deleted_indices = np.where(mask == 1)[0]
        
        # Plot Edges
        for i in active_indices:
            for nbr in adj[i]:
                if nbr != -1 and mask[nbr] == 0:
                    ax.plot([vectors[i,0], vectors[nbr,0]], 
                            [vectors[i,1], vectors[nbr,1]], 
                            c='gray', alpha=0.3, linewidth=0.5)
        
        # Plot Nodes
        ax.scatter(vectors[active_indices, 0], vectors[active_indices, 1], c='blue', s=50, label='Active')
        if len(deleted_indices) > 0:
            ax.scatter(vectors[deleted_indices, 0], vectors[deleted_indices, 1], c='red', marker='x', s=50, label='Deleted')
            
        ax.set_title(f"Step {frame+1}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ani = animation.FuncAnimation(fig, update, frames=len(files), interval=500)
    out_path = os.path.join(vis_dir, 'toy_vis.gif')
    ani.save(out_path, writer='pillow')
    print(f"Saved {out_path}")
    plt.close()

    # 2. Generate Static Query Plots
    query_file = os.path.join(vis_dir, "query_data.bin")
    if os.path.exists(query_file):
        print("Generating Query Plots...")
        queries = read_query_data(query_file)
        
        for q in queries:
            batch_id = q['batch_id']
            # Find snapshot for this batch. Snapshots are 1-indexed in filename usually?
            # In engine.cu: save_snapshot(batch_idx, ...). batch_idx starts at 1.
            snap_path = os.path.join(vis_dir, f"snapshot_{batch_id:03d}.bin")
            
            if not os.path.exists(snap_path):
                print(f"Snapshot {snap_path} not found for query {q['id']}")
                continue
                
            vectors, adj, mask = read_snapshot(snap_path)
            
            plt.figure(figsize=(10, 10))
            
            # Plot Graph
            active_indices = np.where(mask == 0)[0]
            for i in active_indices:
                for nbr in adj[i]:
                    if nbr != -1 and mask[nbr] == 0:
                        plt.plot([vectors[i,0], vectors[nbr,0]], 
                                 [vectors[i,1], vectors[nbr,1]], 
                                 c='lightgray', alpha=0.5, linewidth=0.5)
            
            plt.scatter(vectors[active_indices, 0], vectors[active_indices, 1], c='gray', s=30, alpha=0.5, label='Nodes')
            
            # Plot Query Info
            plt.scatter(q['vec'][0], q['vec'][1], c='red', marker='x', s=150, linewidth=3, label='Query')
            
            # Found
            for fid in q['found_ids']:
                if fid != -1 and fid < len(vectors):
                    plt.scatter(vectors[fid, 0], vectors[fid, 1], c='blue', marker='o', s=80, label='Found')
            
            # GT
            for gid in q['gt_ids']:
                if gid != -1 and gid < len(vectors):
                    plt.scatter(vectors[gid, 0], vectors[gid, 1], c='green', marker='*', s=150, label='Ground Truth')
            
            plt.title(f"Query {q['id']} (Batch {batch_id}) - Recall: {q['recall']:.2f}")
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            q_out = os.path.join(vis_dir, f"query_{q['id']}.png")
            plt.savefig(q_out)
            plt.close()
            print(f"Saved {q_out}")

def read_query_data(filename):
    queries = []
    with open(filename, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        for _ in range(count):
            q = {}
            q['id'] = struct.unpack('i', f.read(4))[0]
            q['batch_id'] = struct.unpack('i', f.read(4))[0]
            q['recall'] = struct.unpack('f', f.read(4))[0]
            q['vec'] = np.frombuffer(f.read(2*4), dtype=np.float32) # 2D
            
            fc = struct.unpack('i', f.read(4))[0]
            q['found_ids'] = np.frombuffer(f.read(fc*4), dtype=np.int32)
            
            gc = struct.unpack('i', f.read(4))[0]
            q['gt_ids'] = np.frombuffer(f.read(gc*4), dtype=np.int32)
            
            # Skip Found/GT vecs as they are in snapshot
            for _ in range(fc): f.read(2*4) # Found vecs
            for _ in range(gc): f.read(2*4) # GT vecs
            
            queries.append(q)
    return queries

if __name__ == "__main__":
    main()
