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
from q_analysis.transformers import GradedParametersTransformer

def calculate_consensus_adjacency_matrix(adjacency_matrices: np.ndarray, edge_inclusion_threshold=0.95, axis=0):
    consensus_adjacency_matrix = adjacency_matrices.mean(axis=axis) > edge_inclusion_threshold
    return consensus_adjacency_matrix.astype(int)

def consensus_statistic(
    adjacency_matrices_a: np.ndarray, 
    adjacency_matrices_b: np.ndarray, 
    axis=None,
    edge_inclusion_threshold=0.95, 
    max_order=None,
):
    """
    Calculates the difference in Q-analysis graded parameters between consensus simplicial complexes
    derived from two sets of adjacency matrices.
    """
    consensus_axis = 0
    if axis is not None:
        consensus_axis = axis
    
    adj_shape = adjacency_matrices_a.shape[-2:]
    consensus_adj_a = calculate_consensus_adjacency_matrix(adjacency_matrices_a, edge_inclusion_threshold, axis=consensus_axis).reshape(-1, *adj_shape)
    consensus_adj_b = calculate_consensus_adjacency_matrix(adjacency_matrices_b, edge_inclusion_threshold, axis=consensus_axis).reshape(-1, *adj_shape)
    
    transformer = GradedParametersTransformer(max_order=max_order)
    q_metrics = transformer.fit_transform(np.concatenate([consensus_adj_a, consensus_adj_b], axis=0))
    q_metrics_a = q_metrics[:consensus_adj_a.shape[0]]
    q_metrics_b = q_metrics[consensus_adj_a.shape[0]:]

    difference = q_metrics_a - q_metrics_b
    difference[np.isnan(difference)] = 0
    return difference
