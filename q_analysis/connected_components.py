import numpy as np
from typing import List, Set, Tuple, Union
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components

def dynamic_method(incidence_matrix: np.ndarray, q: int, return_labels: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Find connected components in a simplicial complex using DFS without computing
    the full connectivity matrix.
    
    Parameters:
    -----------
    incidence_matrix : np.ndarray
        Simplicial incidence matrix where rows represent simplices and columns represent vertices.
    q : int
        Threshold for connectivity:
        - Only consider simplices of order >= q
        - Simplices are connected if they share faces of order >= q
    return_labels : bool, default=False
        If True, returns component labels for each simplex in addition to the
        number of components.
        
    Returns:
    --------
    int or tuple
        If return_labels=False, returns the number of q-connected components.
        If return_labels=True, returns a tuple (num_of_components, labels) where
        labels is an array indicating the component of each simplex (-1 for simplices
        that don't meet the q-connectivity threshold).
    """
    n = q + 1
    # Step 1: Filter simplices with at least n vertices
    simplex_vertices_count = np.sum(incidence_matrix, axis=1)
    valid_indices = np.where(simplex_vertices_count >= n)[0]
    
    # Initialize labels array with -1 (invalid/unassigned component)
    labels = np.full(incidence_matrix.shape[0], -1)
    
    # Early return if no valid simplices
    if len(valid_indices) == 0:
        if return_labels:
            return 0, labels
        return 0
    
    # Step 2: Extract the filtered incidence matrix (only keep rows we need)
    filtered_incidence = incidence_matrix[valid_indices]
    
    # Step 3: Find connected components using DFS
    num_valid_simplices = len(valid_indices)
    visited = [False] * num_valid_simplices
    components = []
    component_count = 0
    
    def count_shared_vertices(idx1, idx2):
        """Count shared vertices between two simplices without matrix multiplication."""
        # Use dot product between the two simplex vertex vectors
        return np.sum(filtered_incidence[idx1] & filtered_incidence[idx2])
    
    def get_neighbors(idx):
        """Find all unvisited neighbors that share at least n vertices without precomputing."""
        neighbors = []
        for j in range(num_valid_simplices):
            if not visited[j] and idx != j:
                # Compute shared vertices directly
                if count_shared_vertices(idx, j) >= n:
                    neighbors.append(j)
        return neighbors
    
    def dfs(start_idx, component_idx):
        """Iterative DFS to find a connected component."""
        stack = [start_idx]
        component = set()
        visited[start_idx] = True
        orig_idx = valid_indices[start_idx]
        component.add(orig_idx)
        labels[orig_idx] = component_idx
        
        while stack:
            current = stack.pop()
            neighbors = get_neighbors(current)
            
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    orig_idx = valid_indices[neighbor]
                    component.add(orig_idx)
                    labels[orig_idx] = component_idx
                    stack.append(neighbor)
        
        return component
    
    # Perform DFS for each unvisited simplex
    for idx in range(num_valid_simplices):
        if not visited[idx]:
            component = dfs(idx, component_count)
            components.append(component)
            component_count += 1
    
    if return_labels:
        return component_count, labels
    else:
        return component_count


def connectivity_matrix_method(connectivity_matrix, q, return_labels=False):
    """
    Find q-connected components using the connectivity matrix approach.
    
    This method identifies connected components in a simplicial complex where
    simplices are connected if they share at least q+1 vertices. It uses the
    scipy.sparse.csgraph.connected_components function on a thresholded adjacency matrix.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        The connectivity matrix where each entry (i,j) represents the number of
        shared vertices between simplices i and j.
    q : int
        The q-connectivity level. Simplices are considered connected if they
        share at least q+1 vertices.
    return_labels : bool, default=False
        If True, returns component labels for each simplex in addition to the
        number of components.
        
    Returns:
    --------
    int or tuple
        If return_labels=False, returns the number of q-connected components.
        If return_labels=True, returns a tuple (num_of_components, labels) where
        labels is an array indicating the component of each simplex (-1 for simplices
        that don't meet the q-connectivity threshold).
    """
    # Create adjacency matrix where simplices are connected if they share at least q+1 vertices
    adj_full = (connectivity_matrix >= q + 1).astype(np.int8)
    adj_diag = adj_full.diagonal()
    adj = adj_full - diags(adj_diag)
    
    if return_labels:
        num_of_components, labels = connected_components(
            adj,
            directed=False,
            return_labels=return_labels
        )
        # Adjust component count by excluding simplices that don't meet the threshold
        num_of_components -= np.sum(adj_diag == 0)
        # Mark simplices that don't meet the threshold with label -1
        labels[adj_diag == 0] = -1
        return num_of_components, labels
    else:
        num_of_components = connected_components(
            adj,
            directed=False,
            return_labels=return_labels
        ) - np.sum(adj_diag == 0)
        return num_of_components
