import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class VamanaIndex:
    """
    Python implementation of the Vamana graph indexing algorithm.
    
    Attributes:
        data (np.ndarray): The dataset vectors.
        R (int): Maximum degree of the graph.
        L (int): Beam width for search during construction.
        alpha (float): Pruning parameter.
    """
    def __init__(self, data, R=32, L=75, alpha=1.2):
        self.data = data
        self.N = len(data)
        self.dim = data.shape[1]
        self.R = R
        self.L = L
        self.alpha = alpha
        self.adj = [set() for _ in range(self.N)]
        self.medoid = self._calculate_medoid()

    def _calculate_medoid(self):
        # Approximate medoid by centroid
        centroid = np.mean(self.data, axis=0)
        dists = np.linalg.norm(self.data - centroid, axis=1)
        return np.argmin(dists)

    def _dist(self, idx1, idx2):
        return np.linalg.norm(self.data[idx1] - self.data[idx2])

    def _dist_vec(self, vec, idx):
        return np.linalg.norm(vec - self.data[idx])

    def build(self):
        """
        Builds the Vamana graph index.
        
        Initializes a random graph and performs 2 passes of Vamana optimization.
        """
        print(f"Initializing random graph (R={self.R})...")
        # 1. Random Initialization
        for i in range(self.N):
            candidates = list(range(self.N))
            candidates.remove(i)
            neighbors = random.sample(candidates, min(self.R, len(candidates)))
            self.adj[i] = set(neighbors)
        
        # Make undirected (optional but good for start)
        # Actually Vamana starts random regular, let's just stick to random out-neighbors

        print("Starting Vamana optimization (2 passes)...")
        for pass_idx in range(2):
            print(f"Pass {pass_idx + 1}/2...")
            # Random permutation
            perm = list(range(self.N))
            random.shuffle(perm)
            
            for i, node in enumerate(perm):
                if i % 1000 == 0:
                    print(f"  Processed {i}/{self.N} nodes", end='\r')
                
                # Greedy Search
                candidates = self.greedy_search(self.data[node], 1, self.L, start_node=self.medoid)
                
                # Robust Prune
                self.robust_prune(node, set(candidates), self.alpha, self.R)
                
                # Back-edges
                for nbr in list(self.adj[node]):
                    if len(self.adj[nbr]) < self.R:
                        self.adj[nbr].add(node)
                    else:
                        # Prune nbr's connections
                        candidates_nbr = set(self.adj[nbr])
                        candidates_nbr.add(node)
                        self.robust_prune(nbr, candidates_nbr, self.alpha, self.R)
            print()

    def greedy_search(self, query_vec, k, beam_width, start_node=None):
        if start_node is None:
            start_node = self.medoid
            
        visited = set()
        candidates = [(self._dist_vec(query_vec, start_node), start_node)]
        results = []
        
        # Simple greedy search (not beam search for construction, but Vamana uses beam)
        # For construction, we just need to find L closest nodes
        
        # Using a set for fast lookup
        candidate_set = {start_node}
        
        # Max heap for L closest? No, we want to expand closest.
        # Let's use a sorted list for simplicity in Python
        
        # Working set W
        W = [(self._dist_vec(query_vec, start_node), start_node)]
        visited.add(start_node)
        
        while True:
            # Find closest unexpanded
            W.sort(key=lambda x: x[0])
            
            found_new = False
            # Iterate through W to find first unexpanded
            # Optimization: keep track of expanded index? 
            # For Python, let's just iterate. L is small (75-100).
            
            current_best = W[0][1]
            
            for dist, node in W:
                if node in visited and getattr(self, '_expanded_' + str(node), False): 
                    # Hacky way to mark expanded in this local scope? No.
                    # Just keep a separate expanded set
                    continue
                
                # Expand 'node'
                # Actually Vamana greedy search:
                # 1. Start with p
                # 2. Iterate
                pass
            
            # Re-implementing standard greedy search
            # L is capacity of candidate list
            
            # Let's use a simpler logic:
            # Maintain top-L candidates.
            # Pick closest unvisited from candidates.
            
            best_l = W[:beam_width] # Actually L is usually search list size
            
            # Find closest unexpanded in W
            unexpanded = [n for d, n in W if n not in visited] # Wait, visited means expanded?
            # Usually visited means "computed distance". Expanded means "visited neighbors".
            
            # Let's use: visited = set of nodes where we computed distance
            # expanded = set of nodes where we visited neighbors
            
            expanded = set()
            
            # Reset
            W = [(self._dist_vec(query_vec, start_node), start_node)]
            visited = {start_node}
            
            while True:
                W.sort(key=lambda x: x[0])
                # Truncate to L
                if len(W) > beam_width:
                    W = W[:beam_width]
                
                # Find closest node in W that hasn't been expanded
                target_node = -1
                for dist, node in W:
                    if node not in expanded:
                        target_node = node
                        break
                
                if target_node == -1:
                    break # All in W are expanded
                
                expanded.add(target_node)
                
                for nbr in self.adj[target_node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        d = self._dist_vec(query_vec, nbr)
                        W.append((d, nbr))
            
            return [n for d, n in W]

    # Optimized Greedy Search for Construction
    def greedy_search(self, query_vec, k, search_list_size, start_node=None):
        """
        Performs greedy search on the graph.
        
        Args:
            query_vec (np.ndarray): Query vector.
            k (int): Number of results to return (unused in construction logic).
            search_list_size (int): Beam width / candidate list size.
            start_node (int, optional): Starting node ID. Defaults to medoid.
            
        Returns:
            list: List of nearest neighbor IDs.
        """
        if start_node is None: start_node = self.medoid
        
        results = set()
        results.add(start_node)
        
        candidates = [(self._dist_vec(query_vec, start_node), start_node)]
        visited = {start_node}
        
        # To emulate Vamana search:
        # Maintain a list of L best candidates found so far.
        # Expand the closest unexpanded candidate from this list.
        
        expanded = set()
        
        while True:
            candidates.sort(key=lambda x: x[0])
            
            # Find closest unexpanded within the top L
            # Note: Vamana paper says "greedy search returns list V". 
            # Usually this is the list of visited nodes or the top L.
            # We return top L.
            
            current_k = candidates[:search_list_size]
            
            next_node = -1
            for dist, node in current_k:
                if node not in expanded:
                    next_node = node
                    break
            
            if next_node == -1:
                break
                
            expanded.add(next_node)
            
            for nbr in self.adj[next_node]:
                if nbr not in visited:
                    visited.add(nbr)
                    d = self._dist_vec(query_vec, nbr)
                    candidates.append((d, nbr))
        
        candidates.sort(key=lambda x: x[0])
        return [n for d, n in candidates[:search_list_size]]

    def robust_prune(self, node, candidates, alpha, R):
        """
        Prunes the candidate list to maintain graph connectivity and diversity.
        
        Args:
            node (int): The node being updated.
            candidates (set): Set of candidate neighbor IDs.
            alpha (float): Pruning threshold.
            R (int): Maximum degree.
        """
        # candidates is a set of node indices
        # Add current neighbors to candidates
        candidates.update(self.adj[node])
        if node in candidates: candidates.remove(node)
        
        # Sort by distance to node
        cand_list = []
        for c in candidates:
            cand_list.append((self._dist(node, c), c))
        cand_list.sort(key=lambda x: x[0])
        
        new_adj = []
        while cand_list:
            # Pick closest
            d_p_star, p_star = cand_list.pop(0)
            new_adj.append(p_star)
            
            if len(new_adj) == R: break
            
            # Prune
            # Remove points p' from cand_list if alpha * dist(p*, p') < dist(node, p')
            to_remove = []
            for i, (d_p_prime, p_prime) in enumerate(cand_list):
                dist_p_star_prime = self._dist(p_star, p_prime)
                if alpha * dist_p_star_prime < d_p_prime:
                    to_remove.append(i)
            
            # Remove in reverse order
            for i in sorted(to_remove, reverse=True):
                cand_list.pop(i)
                
        self.adj[node] = set(new_adj)

    def get_adj_matrix(self):
        # Convert to numpy array with padding
        max_d = self.R
        adj_arr = np.full((self.N, max_d), -1, dtype=np.int32)
        for i in range(self.N):
            nbrs = list(self.adj[i])
            # If more than R (shouldn't happen after prune, but safety), truncate
            nbrs = nbrs[:max_d]
            adj_arr[i, :len(nbrs)] = nbrs
        return adj_arr
