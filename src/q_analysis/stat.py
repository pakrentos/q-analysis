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
from scipy import stats
from .simplicial_complex import IncidenceSimplicialComplex
from .utils import pad_structure_vectors, calculate_consensus_adjacency_matrix, adj_matrices_to_q_analysis_vectors

def permutation_test_simplicial_complexes(
    complex1, 
    complex2, 
    n_permutations=1000, 
    random_state=None, 
    alternative='two-sided'
):
    """
    Perform a permutation test to compare two simplicial complexes.
    
    Parameters:
    -----------
    complex1 : IncidenceSimplicialComplex
        First simplicial complex to compare
    complex2 : IncidenceSimplicialComplex
        Second simplicial complex to compare
    n_permutations : int, optional
        Number of permutations to use for the test (default: 1000)
    random_state : int or numpy.random.RandomState, optional
        Random seed or random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'structure_vectors1': Structure vectors from the first complex
        - 'structure_vectors2': Structure vectors from the second complex
        - 'difference': Difference between structure vectors
        - 'pvalues': p-values for each metric at each order
    """
    def statistic(incidence_a, incidence_b):
        sv_a = IncidenceSimplicialComplex(incidence_a).q_analysis_vectors()
        sv_b = IncidenceSimplicialComplex(incidence_b).q_analysis_vectors()
        if tail == 'two-sided':
            return np.abs(sv_a - sv_b)
        elif tail == 'less':
            return sv_a - sv_b
        elif tail == 'greater':
            return sv_b - sv_a
    
    if tail not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid tail argument. Must be one of: 'two-sided', 'less', 'greater'")
    
    # Validate input
    if not isinstance(complex1, IncidenceSimplicialComplex) or not isinstance(complex2, IncidenceSimplicialComplex):
        raise TypeError("Both inputs must be IncidenceSimplicialComplex objects")
    
    # Compute structure vectors for both complexes (with order mask)
    sv1, sv2 = complex1.q_analysis_vectors(), complex2.q_analysis_vectors()
    max_order = max(sv1.shape[0], sv2.shape[0])

    combined_data = pad_structure_vectors([sv1, sv2])
    sv1, sv2 = combined_data
    
    # Calculate the observed difference between structure vectors
    observed_diff = statistic(complex1.incidence, complex2.incidence)
    
    # Initialize p-values array
    n_metrics = sv1.shape[-1]
    pvalues = np.zeros((max_order, n_metrics))
    
    # Perform permutation test for each order and metric
    rng = np.random.RandomState(random_state)
    
    # pvalues = stats.permutation_test(
    #     (complex1.incidence, complex2.incidence), 
     
    return {
        'structure_vectors1': sv1_metrics,
        'structure_vectors2': sv2_metrics,
        'difference': observed_diff,
        'pvalues': pvalues,
        'max_order': max_order
    }


def consensus_statistic(
    adjacency_matrices_a, 
    adjacency_matrices_b, 
    axis=None,
    edge_inclusion_threshold=0.95, 
    max_order=None,
):
    consensus_adjacency_matrix_a = calculate_consensus_adjacency_matrix(adjacency_matrices_a, edge_inclusion_threshold, axis=-3)
    consensus_adjacency_matrix_b = calculate_consensus_adjacency_matrix(adjacency_matrices_b, edge_inclusion_threshold, axis=-3)
    
    if consensus_adjacency_matrix_a.ndim == 2:
        aligned_consensus_q_metrics_a = adj_matrices_to_q_analysis_vectors(
            consensus_adjacency_matrix_a[None, ...], max_order=max_order
        )[0]
        aligned_consensus_q_metrics_b = adj_matrices_to_q_analysis_vectors(
            consensus_adjacency_matrix_b[None, ...], max_order=max_order
        )[0]
    else:
        aligned_consensus_q_metrics_a = adj_matrices_to_q_analysis_vectors(
            consensus_adjacency_matrix_a, max_order=max_order
        )
        aligned_consensus_q_metrics_b = adj_matrices_to_q_analysis_vectors(consensus_adjacency_matrix_b, max_order=max_order)
    
    difference = aligned_consensus_q_metrics_a - aligned_consensus_q_metrics_b
    difference[np.isnan(difference)] = 0
    return difference

def difference_statistic(
    incidence_a, 
    incidence_b, 
    axis=None,
    max_order=None,
    number_of_nodes=1,
    simplicial_set=set(),
):  
    print(axis)
    print(incidence_a.shape, incidence_b.shape)
    
    if incidence_a.ndim == 2:
        print(f'{hash(tuple(incidence_a.flatten().tolist()))=}')
        print(f'{hash(tuple(incidence_b.flatten().tolist()))=}')
        aligned_consensus_q_metrics_a, aligned_consensus_q_metrics_b = pad_structure_vectors(
            [
                IncidenceSimplicialComplex(incidence_a.T).q_analysis_vectors(),
                IncidenceSimplicialComplex(incidence_b.T).q_analysis_vectors()
            ],
            max_order=max_order
        )
        print(aligned_consensus_q_metrics_a)
        print(aligned_consensus_q_metrics_b)
    else:
        len_a = len(incidence_a)
        for incidence in incidence_a:
            for simplex in incidence.T:
                assert tuple(simplex.astype(int).tolist()) in simplicial_set
        for incidence in incidence_b:
            for simplex in incidence.T:
                assert tuple(simplex.astype(int).tolist()) in simplicial_set
        aligned_consensus_q_metrics = pad_structure_vectors(
            [
                IncidenceSimplicialComplex(incidence.T).q_analysis_vectors()
                for incidence in incidence_a
            ] + [
                IncidenceSimplicialComplex(incidence.T).q_analysis_vectors()
                for incidence in incidence_b
            ],
            max_order=max_order
        )
        aligned_consensus_q_metrics_a = aligned_consensus_q_metrics[:len_a]
        aligned_consensus_q_metrics_b = aligned_consensus_q_metrics[len_a:]
    
    difference = aligned_consensus_q_metrics_a - aligned_consensus_q_metrics_b
    difference[np.isnan(difference)] = 0
    return difference