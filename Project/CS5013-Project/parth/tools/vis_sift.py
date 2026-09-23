import argparse
import json
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def read_sift_vis(filename):
    queries = []
    with open(filename, 'rb') as f:
        count = struct.unpack('i', f.read(4))[0]
        for _ in range(count):
            q = {}
            q['id'] = struct.unpack('i', f.read(4))[0]
            q['batch_id'] = struct.unpack('i', f.read(4))[0] # Read batch_id
            q['recall'] = struct.unpack('f', f.read(4))[0]
            q['vec'] = np.frombuffer(f.read(128*4), dtype=np.float32)
            
            fc = struct.unpack('i', f.read(4))[0]
            q['found_ids'] = np.frombuffer(f.read(fc*4), dtype=np.int32)
            
            gc = struct.unpack('i', f.read(4))[0]
            q['gt_ids'] = np.frombuffer(f.read(gc*4), dtype=np.int32)
            
            q['found_vecs'] = []
            for _ in range(fc):
                q['found_vecs'].append(np.frombuffer(f.read(128*4), dtype=np.float32))
            
            q['gt_vecs'] = []
            for _ in range(gc):
                q['gt_vecs'].append(np.frombuffer(f.read(128*4), dtype=np.float32))
            
            queries.append(q)
    return queries

def load_initial_graph_vecs(filename, dim):
    # Just read vectors to find neighbors for plotting context
    with open(filename, 'rb') as f:
        num_nodes = struct.unpack('i', f.read(4))[0]
        d = struct.unpack('i', f.read(4))[0]
        md = struct.unpack('i', f.read(4))[0]
        vecs = np.frombuffer(f.read(num_nodes * dim * 4), dtype=np.float32).reshape(num_nodes, dim)
    return vecs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    vis_dir = config['paths']['vis_output_dir']
    data_file = os.path.join(vis_dir, "query_data.bin") # Renamed file
    
    if not os.path.exists(data_file):
        print(f"No data found at {data_file}")
        return

    queries = read_sift_vis(data_file)
    
    # Load full graph to get context neighbors
    print("Loading full graph for context...")
    full_vecs = load_initial_graph_vecs(config['paths']['initial_graph'], 128)
    
    for i, q in enumerate(queries):
        # 1. Get Context: 
        # Find ~20 nearest neighbors of the QUERY
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='brute').fit(full_vecs)
        _, context_ids_q = nbrs.kneighbors([q['vec']])
        
        # Find ~20 nearest neighbors of the Top-1 FOUND node (to see where search went)
        # We use the first found node for context
        _, context_ids_f = nbrs.kneighbors([q['found_vecs'][0]])
        
        # Combine unique context IDs
        context_ids = np.unique(np.concatenate((context_ids_q[0], context_ids_f[0])))
        context_vecs = full_vecs[context_ids]
        
        # 2. PCA Projection
        # Combine all relevant vectors to fit PCA
        all_vecs = [q['vec']] + list(context_vecs)
        X = np.array(all_vecs)
        
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        q_2d = X_2d[0]
        context_2d = X_2d[1:]
        
        plt.figure(figsize=(10, 10))
        
        # 3. Plot Context Nodes & Edges
        # Draw edges between context nodes if they are close (simulating graph edges)
        # We compute local KNN on 2D projection for visualization edges
        ctx_nbrs = NearestNeighbors(n_neighbors=3).fit(context_2d)
        _, ctx_adj = ctx_nbrs.kneighbors(context_2d)
        
        for idx, neighbors in enumerate(ctx_adj):
            for n in neighbors:
                plt.plot([context_2d[idx,0], context_2d[n,0]], 
                         [context_2d[idx,1], context_2d[n,1]], 
                         c='gray', alpha=0.2, linewidth=0.5)

        plt.scatter(context_2d[:, 0], context_2d[:, 1], c='lightgray', s=30, label='Context')
        
        # 4. Highlight Found & GT
        # Project Found
        found_vecs = np.array(q['found_vecs'])
        if len(found_vecs) > 0:
            found_2d = pca.transform(found_vecs)
            plt.scatter(found_2d[:, 0], found_2d[:, 1], c='blue', marker='o', s=80, label='Found')
        
        # Project GT
        gt_vecs = np.array(q['gt_vecs'])
        if len(gt_vecs) > 0:
            gt_2d = pca.transform(gt_vecs)
            plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c='green', marker='*', s=150, label='Ground Truth')

        # Plot Query
        plt.scatter(q_2d[0], q_2d[1], c='red', marker='x', s=100, label='Query')
        
        plt.title(f"Query {q['id']} - Recall@10: {q['recall']:.2f}")
        plt.legend()
        
        out_path = os.path.join(vis_dir, f"query_{q['id']}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
