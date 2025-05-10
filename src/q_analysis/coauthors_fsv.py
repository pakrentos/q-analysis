import numpy as np
from pathlib import Path
from typing import List, Tuple
from q_analysis._q_analysis_ext import py_find_hierarchical_q_components

def read_coauthors_dataset(data_dir: str) -> List[List[int]]:
    """Read coauthors dataset and return list of simplices.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        List of simplices, where each simplex is a list of vertex indices
    """
    data_dir = Path(data_dir)
    
    # Read number of vertices per simplex
    with open(data_dir / "coauth-DBLP-nverts.txt") as f:
        nverts = [int(line.strip()) for line in f]
    
    # Read vertices for each simplex
    with open(data_dir / "coauth-DBLP-simplices.txt") as f:
        vertices = [int(line.strip()) for line in f]
    
    # Read timestamps (not used in this analysis)
    with open(data_dir / "coauth-DBLP-times.txt") as f:
        times = [int(line.strip()) for line in f]
    
    # Convert to list of simplices
    simplices = []
    idx = 0
    for n in nverts:
        simplex = vertices[idx:idx + n]
        simplices.append(simplex)
        idx += n
    
    return simplices

def compute_fsv(simplices: List[List[int]]) -> List[int]:
    """Compute FSV (Filtration Simplicial Vector) for given simplices.
    
    Args:
        simplices: List of simplices, where each simplex is a list of vertex indices
        
    Returns:
        List where i-th element is number of components at q-level i
    """
    # Find hierarchical q-components
    components = py_find_hierarchical_q_components(simplices)
    
    # Count number of components at each q-level
    max_q = max(max(comp) for comp in components)
    fsv = [0] * (max_q + 1)
    
    for comp in components:
        for q in comp:
            fsv[q] += 1
    
    return fsv

def main():
    # Path to dataset directory
    data_dir = "coauth-DBLP"  # Adjust this path as needed
    
    # Read dataset
    simplices = read_coauthors_dataset(data_dir)
    print(f"Read {len(simplices)} simplices")
    
    # Compute FSV
    fsv = compute_fsv(simplices)
    print("\nFSV (number of components at each q-level):")
    for q, count in enumerate(fsv):
        print(f"q={q}: {count} components")

if __name__ == "__main__":
    main() 