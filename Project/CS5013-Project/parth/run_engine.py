import json
import argparse
import subprocess
import os

"""
Script to run the Vamana engine with a specified configuration.
"""

def main():
    """Runs the Vamana engine with the specified configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config json")
    parser.add_argument('--exe', type=str, required=True, help="Path to compiled engine executable")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract args
    initial_graph = config['paths']['initial_graph']
    workload = config['paths']['workload']
    batch_size = str(config['execution']['batch_size'])
    vis_output_dir = config['paths']['vis_output_dir']
    
    # Ensure output dir exists
    os.makedirs(vis_output_dir, exist_ok=True)

    # Run Engine
    cmd = [
        args.exe,
        initial_graph,
        workload,
        batch_size,
        vis_output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
