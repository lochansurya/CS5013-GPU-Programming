import numpy as np
import os
import struct
import argparse
from sklearn.neighbors import NearestNeighbors
import subprocess
import tempfile
import json

def read_fvecs(filename):
    """Reads vectors from an fvecs file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        while f.tell() < file_size:
            header = f.read(4)
            if not header: break
            dim = struct.unpack('i', header)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            yield np.array(vec, dtype=np.float32)

def get_sift10k():
    """Downloads and loads the SIFT-10K dataset."""
    if not os.path.exists('siftsmall'):
        print("Downloading SIFT-10K...")
        if os.system("wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz") != 0:
            os.system("curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz")
        os.system("tar -zxvf siftsmall.tar.gz")
    
    base = list(read_fvecs('siftsmall/siftsmall_base.fvecs'))
    query = list(read_fvecs('siftsmall/siftsmall_query.fvecs'))
    return np.array(base), np.array(query)

def build_graph(data, k=32, output_file="temp_graph.bin"):
    """Builds the Vamana graph using the CUDA executable."""
    print(f"Building Vamana graph for {len(data)} nodes (R={k}) using CUDA...")
    
    # Write temp input file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(struct.pack('ii', len(data), data.shape[1]))
        tmp.write(data.tobytes())
        input_file = tmp.name
        
    # Run build_graph_sift
    cmd = [
        "./build_graph_sift",
        input_file,
        output_file,
        str(k),
        str(min(len(data), 2*k)), # L
        "1.2" # alpha
    ]
    
    subprocess.run(cmd, check=True)
    os.unlink(input_file)
    
    # Read back adjacency to return it
    with open(output_file, 'rb') as f:
        # Skip header [N, DIM, MAX_DEGREE]
        f.read(12)
        # Skip vectors
        f.read(len(data) * data.shape[1] * 4)
        # Read Adj
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

    base, query = get_sift10k()
    
    # Use full base for initial graph or subset if specified
    initial_data = base
    if 'size' in config['graph']:
        size = config['graph']['size']
        if size < len(base):
            print(f"Using subset of {size} nodes (from {len(base)})")
            initial_data = base[:size]
            
    dim = config['dim']
    k = config['graph']['k']
    max_degree = config['graph']['max_degree']
    
    # Build Graph
    adj = build_graph(initial_data, k=k)
    write_initial_graph(config['paths']['initial_graph'], initial_data, adj, dim, max_degree)
    
    # Generate Workload
    wl_config = config['workload']
    num_ops = wl_config['num_ops']
    total_frac = wl_config['frac_search'] + wl_config['frac_insert'] + wl_config['frac_delete']
    p_search = wl_config['frac_search'] / total_frac
    p_insert = wl_config['frac_insert'] / total_frac
    
    operations = []
    dummy_vec = np.zeros(dim, dtype=np.float32)
    
    print(f"Generating {num_ops} mixed operations...")
    
    for _ in range(num_ops):
        r = np.random.rand()
        if r < p_search: # Search
            q_idx = np.random.randint(0, len(query))
            operations.append((2, -1, query[q_idx]))
        elif r < p_search + p_insert: # Insert
            base_idx = np.random.randint(0, len(base))
            new_vec = base[base_idx] + np.random.normal(0, 0.1, dim).astype(np.float32)
            operations.append((1, -1, new_vec))
        else: # Delete
            did = np.random.randint(0, len(base))
            operations.append((0, did, dummy_vec))
            
    write_workload(config['paths']['workload'], operations, dim)
    print("Done.")

if __name__ == "__main__":
    main()
