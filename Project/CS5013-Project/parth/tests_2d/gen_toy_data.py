import numpy as np
import struct
import argparse
from sklearn.neighbors import NearestNeighbors
import subprocess
import tempfile
import os
import json

def build_graph(data, k=6, output_file="temp_graph.bin"):
    """Builds the Vamana graph using the CUDA executable."""
    print(f"Building Vamana graph for {len(data)} nodes (R={k}) using CUDA...")
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(struct.pack('ii', len(data), data.shape[1]))
        tmp.write(data.tobytes())
        input_file = tmp.name
        
    cmd = [
        "./build_graph_toy",
        input_file,
        output_file,
        str(k),
        str(min(len(data), 2*k)),
        "1.2"
    ]
    
    subprocess.run(cmd, check=True)
    os.unlink(input_file)
    
    with open(output_file, 'rb') as f:
        f.read(12)
        f.read(len(data) * data.shape[1] * 4)
        adj = np.frombuffer(f.read(len(data) * 32 * 4), dtype=np.int32).reshape(len(data), 32)
        
    return adj

def write_initial_graph(filename, vectors, adj, dim, max_degree):
    """Writes the initial graph to a binary file."""
    print(f"Writing {filename}...")
    with open(filename, 'wb') as f:
        f.write(struct.pack('iii', len(vectors), dim, max_degree))
        f.write(vectors.tobytes())
        if adj.shape[1] < max_degree:
            padding = np.full((adj.shape[0], max_degree - adj.shape[1]), -1, dtype=np.int32)
            adj = np.hstack((adj, padding))
        f.write(adj.tobytes())

def write_workload(filename, operations, dim):
    """Writes the workload operations to a binary file."""
    print(f"Writing {filename}...")
    with open(filename, 'wb') as f:
        for op_type, op_id, op_vec in operations:
            f.write(struct.pack('ii', op_type, op_id))
            f.write(op_vec.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config json")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dim = config['dim']
    size = config['graph'].get('size', 50) # Default to 50 if not present
    k = config['graph']['k']
    max_degree = config['graph']['max_degree']
    
    # Generate random 2D data
    initial_data = np.random.rand(size, dim).astype(np.float32)
    
    # Build Graph
    adj = build_graph(initial_data, k=k)
    write_initial_graph(config['paths']['initial_graph'], initial_data, adj, dim, max_degree)
    
    # Generate Workload (Mixed)
    operations = []
    dummy_vec = np.zeros(dim, dtype=np.float32)
    
    # Hardcoded mix for visualization demo - KEEPING FIRST FEW for consistent demo start
    # 1. Search
    operations.append((2, -1, np.random.rand(dim).astype(np.float32)))
    # 2. Insert
    operations.append((1, -1, np.random.rand(dim).astype(np.float32)))
    # 3. Delete
    operations.append((0, 5, dummy_vec)) # Delete node 5
    # 4. Search
    operations.append((2, -1, np.random.rand(dim).astype(np.float32)))
    
    wl_config = config['workload']
    num_ops = wl_config['num_ops']
    
    # Calculate probabilities
    total_frac = wl_config['frac_search'] + wl_config['frac_insert'] + wl_config['frac_delete']
    p_search = wl_config['frac_search'] / total_frac
    p_insert = wl_config['frac_insert'] / total_frac
    
    # Fill rest with random based on config
    for _ in range(num_ops - 4):
        r = np.random.rand()
        if r < p_search:
            operations.append((2, -1, np.random.rand(dim).astype(np.float32)))
        elif r < p_search + p_insert:
            operations.append((1, -1, np.random.rand(dim).astype(np.float32)))
        else:
            did = np.random.randint(0, size)
            operations.append((0, did, dummy_vec))
            
    write_workload(config['paths']['workload'], operations, dim)
    print("Done.")

if __name__ == "__main__":
    main()
