"""
Q-analysis Package
Copyright (C) 2024 Nikita Smirnov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.sparse import csr_array
import networkx as nx
from typing import Iterator
from tqdm import tqdm
from q_analysis.simplicial_complex import IncidenceSimplicialComplex, VECTORS

def efficient_laplacian(A):
    Q, R = np.linalg.qr(A)
    return np.round(Q @ (R @ R.T) @ Q.T)

def efficient_kirchoff(A):
    _, R = np.linalg.qr(A)
    return np.round(R.T @ R)

def efficient_laplacian_kirchoff(A):
    Q, R = np.linalg.qr(A)
    return np.round(Q @ (R @ R.T) @ Q.T), np.round(R.T @ R)

def adj_matrices_to_q_analysis_vectors(adj_matrices: np.ndarray, max_order=None, fill_value=np.nan) -> np.ndarray:
    """
    Compute q-analysis structure vectors for each provided adjacency matrix
    """
    structure_vectors_list = [
        IncidenceSimplicialComplex
            .from_adjacency_matrix(adj)
            .q_analysis_vectors()
        for adj in tqdm(adj_matrices)
    ]
    
    return pad_structure_vectors(structure_vectors_list, max_order, fill_value)

def pad_structure_vectors(structure_vectors_list, max_order=None, fill_value=np.nan) -> np.ndarray:
    """
    Pad structure vectors to a consistent shape and apply max_order limit if specified
    
    Parameters:
    -----------
    structure_vectors_list : list of numpy arrays
        List of structure vectors to pad
    max_inferred_order : int
        Maximum order inferred from the structure vectors
    max_order : int, optional
        Maximum order to include in the result
    fill_value : float, optional
        Value to use for padding
        
    Returns:
    --------
    numpy.ndarray
        Padded structure vectors array
    """
    def calc_pad_shape(max_q, array):
        n_q, _ = array.shape
        return ((0, max_q - n_q), (0, 0))
    
    max_inferred_order = max([i.shape[0] for i in structure_vectors_list])
    
    if max_order is not None and max_order > max_inferred_order:
        max_inferred_order = max_order
    
    structure_vectors_array = np.array([
        np.pad(
            arr, 
            calc_pad_shape(max_inferred_order, arr),
            constant_values=fill_value
        )
        for arr in structure_vectors_list
    ])
    structure_vectors_array = structure_vectors_array[:, :max_order]
    return structure_vectors_array

def apply_order_mask(structure_vectors_dataset, fill_value=np.nan):
    """
    Use the 7th metric as mask for each sample to mark padded values
    (i.e. metric values for non-existent orders) as np.nan
    """
    mask = structure_vectors_dataset[..., -1].astype(bool)
    structure_vectors_dataset[~mask] = fill_value
    return structure_vectors_dataset[..., :-1]

def align_structure_vectors_datasets(*datasets):
    """
    Each dataset has different maximal clique order. This method is used
    for padding those datasets, so that their shape would be equal
    """
    # Find the maximum order across all datasets
    max_order = max(dataset.shape[1] for dataset in datasets)
    
    # Pad each dataset to the maximum order
    padded_datasets = []
    for dataset in datasets:
        _, current_order, _ = dataset.shape
        padded_dataset = np.pad(
            dataset, 
            ((0, 0), (0, max_order - current_order), (0, 0)), 
            constant_values=0
        )
        padded_datasets.append(padded_dataset)
    
    return padded_datasets

def construct_structure_vectors_datasets(*adj_datasets, max_order=None, fill_value=np.nan):
    """
    Based on provided graph datasets (represented by their corresponding
    adjacency matrices), compute q-analysis structure vectors and return them in a unified format
    Resulting arrays have shape (n_samples, n_orders, n_metrics)
    
    Parameters:
    -----------
    *adj_datasets : variable number of numpy arrays
        Each array contains adjacency matrices for a dataset
    max_order : int, optional
        The maximum order of the structure vectors. If not provided, the maximum order from the provided datasets will be used.
    fill_value : float, optional
        The value to fill the padded values with. If not provided, np.nan will be used.
    
    Returns:
    --------
    list of numpy arrays
        Q-analysis metrics for each dataset, with consistent shapes
    """
    # Compute q-metrics for each dataset
    structure_vectors_list = [adj_matrices_to_q_analysis_vectors(adj_dataset, max_order, fill_value) for adj_dataset in adj_datasets]
    
    # Align all datasets to have the same shape
    aligned_structure_vectors = align_structure_vectors_datasets(*structure_vectors_list)
    
    # Apply mask to each dataset
    masked_structure_vectors = [apply_order_mask(structure_vectors, fill_value) for structure_vectors in aligned_structure_vectors]

    if len(masked_structure_vectors) == 1:
        return masked_structure_vectors[0]
    else:
        return masked_structure_vectors

def calculate_consensus_adjacency_matrix(adjacency_matrices: np.ndarray, edge_inclusion_threshold=0.95, axis=0):
    consensus_adjacency_matrix = adjacency_matrices.mean(axis=axis) > edge_inclusion_threshold
    return consensus_adjacency_matrix.astype(int)


