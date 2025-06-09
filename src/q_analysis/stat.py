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
from q_analysis.simplicial_complex import IncidenceSimplicialComplexDev, VECTORS, GradedParameters
from .utils import pad_structure_vectors, calculate_consensus_adjacency_matrix, adj_matrices_to_q_analysis_vectors

# TODO: need to refactor this file

def _convert_graded_parameters_to_numpy(graded_params: GradedParameters, complex_order: int, fill_value=np.nan) -> np.ndarray:
    """
    Converts a GradedParameters object to a 2D NumPy array.

    Parameters:
    -----------
    graded_params : GradedParameters
        The GradedParameters object to convert.
    complex_order : int
        The maximum q-level (order) of the complex. Array will have complex_order + 1 rows.
    fill_value : float, optional
        Value to use for missing data or if a parameter's values are shorter than expected.

    Returns:
    --------
    np.ndarray
        A 2D NumPy array of shape (complex_order + 1, len(VECTORS)).
    """
    num_rows = complex_order + 1 if complex_order != -1 else 0
    num_metrics = len(VECTORS)
    
    output_array = np.full((num_rows, num_metrics), fill_value, dtype=float)

    if complex_order == -1: # Empty complex, returns array of shape (0, num_metrics)
        return output_array

    for i, vec_name in enumerate(VECTORS):
        param_set = graded_params.get_parameter(vec_name)
        if param_set:
            if len(param_set.values) == num_rows:
                output_array[:, i] = param_set.values.astype(float)
            elif len(param_set.values) < num_rows and len(param_set.values) > 0:
                # Parameter values are shorter than expected, pad with fill_value at the end
                output_array[:len(param_set.values), i] = param_set.values.astype(float)
                # The rest of the column remains fill_value due to np.full initialization
            elif len(param_set.values) > num_rows: # Should not happen with correct GradedParameterSet construction
                output_array[:, i] = param_set.values[:num_rows].astype(float) # Truncate
            # If param_set.values is empty but num_rows > 0, it remains fill_value
        # If param_set is None (vector not found), it also remains fill_value
            
    return output_array

def permutation_test_simplicial_complexes(
    complex1: IncidenceSimplicialComplexDev, 
    complex2: IncidenceSimplicialComplexDev, 
    n_permutations=1000, 
    random_state=None, 
    alternative='two-sided'
):
    """
    Perform a permutation test to compare two simplicial complexes.
    NOTE: The permutation logic itself (how to shuffle/permute data between complexes)
    is not fully implemented here; this function primarily calculates the observed
    difference between the structure vectors of two *given* complexes and sets up the structure.
    The scipy.stats.permutation_test call is illustrative and would need proper data input.
    
    Parameters:
    -----------
    complex1 : IncidenceSimplicialComplexDev
        First simplicial complex to compare.
    complex2 : IncidenceSimplicialComplexDev
        Second simplicial complex to compare.
    n_permutations : int, optional
        Number of permutations (placeholder, as test is not fully implemented).
    random_state : int or numpy.random.RandomState, optional
        Random seed or random state (placeholder).
    alternative : str, optional
        Defines the alternative hypothesis: 'two-sided', 'less', 'greater'.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'structure_vectors1': GradedParameters from the first complex.
        - 'structure_vectors2': GradedParameters from the second complex.
        - 'difference': NumPy array of the difference between (padded) structure vectors.
        - 'pvalues': Placeholder for p-values (currently zeros).
        - 'max_order': The maximum order used for padding/comparison.
    """
    
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid alternative argument. Must be one of: 'two-sided', 'less', 'greater'")
    
    # Validate input type
    if not isinstance(complex1, IncidenceSimplicialComplexDev) or not isinstance(complex2, IncidenceSimplicialComplexDev):
        raise TypeError("Both inputs must be IncidenceSimplicialComplexDev objects")
    
    # Get GradedParameters objects
    gp1 = complex1.q_analysis_vectors(desired_vectors=VECTORS)
    gp2 = complex2.q_analysis_vectors(desired_vectors=VECTORS)

    # Convert GradedParameters to NumPy arrays using the helper
    # complex.order() gives the max q-level.
    np_sv1 = _convert_graded_parameters_to_numpy(gp1, complex1.order())
    np_sv2 = _convert_graded_parameters_to_numpy(gp2, complex2.order())
    
    # Pad the NumPy arrays to the same dimensions for comparison
    # pad_structure_vectors expects a list of 2D arrays and returns a 3D array (n_samples, n_orders, n_metrics)
    # Here, n_samples is 2.
    # max_order for pad_structure_vectors refers to q-level, it will add 1 for rows.
    effective_max_order = max(complex1.order(), complex2.order())
    padded_data_3d = pad_structure_vectors([np_sv1, np_sv2], max_order=effective_max_order) # max_order here is q-level
    
    sv1_padded_np = padded_data_3d[0] # Shape: (effective_max_order + 1, num_metrics)
    sv2_padded_np = padded_data_3d[1]
    
    # Calculate the observed difference based on the alternative hypothesis
    if alternative == 'two-sided':
        observed_diff_np = np.abs(sv1_padded_np - sv2_padded_np)
    elif alternative == 'less': # complex1 is less than complex2 implies sv1 < sv2, so diff = sv2 - sv1
        observed_diff_np = sv2_padded_np - sv1_padded_np
    elif alternative == 'greater': # complex1 is greater than complex2 implies sv1 > sv2, so diff = sv1 - sv2
        observed_diff_np = sv1_padded_np - sv2_padded_np
    
    # Initialize p-values array (actual permutation test logic is a placeholder)
    # The shape of pvalues should match observed_diff_np
    pvalues_np = np.zeros_like(observed_diff_np)
    
    # The scipy.stats.permutation_test would require:
    # 1. A callable statistic function that takes permuted data and returns the test statistic.
    # 2. The actual data to be permuted.
    # Example (conceptual, as data structure for simplices/complexes needs careful handling for permutation):
    # def stat_func_for_permutation(perm_simplices1, perm_simplices2, q_level, metric_idx):
    #     c1 = IncidenceSimplicialComplexDev(perm_simplices1)
    #     c2 = IncidenceSimplicialComplexDev(perm_simplices2)
    #     gp1_perm = c1.q_analysis_vectors(desired_vectors=VECTORS)
    #     gp2_perm = c2.q_analysis_vectors(desired_vectors=VECTORS)
    #     np1_perm = _convert_graded_parameters_to_numpy(gp1_perm, c1.order())
    #     np2_perm = _convert_graded_parameters_to_numpy(gp2_perm, c2.order())
    #     padded_perm = pad_structure_vectors([np1_perm, np2_perm], max_order=effective_max_order)
    #     return padded_perm[0, q_level, metric_idx] - padded_perm[1, q_level, metric_idx] # or abs, etc.
    # This is highly complex to implement generically here.

    return {
        'structure_vectors1': gp1, # Return GradedParameters objects
        'structure_vectors2': gp2,
        'difference': observed_diff_np, # NumPy array of differences
        'pvalues': pvalues_np, # Placeholder NumPy array
        'max_order': effective_max_order # The q-level used for comparison length
    }


def consensus_statistic(
    adjacency_matrices_a: np.ndarray, 
    adjacency_matrices_b: np.ndarray, 
    axis=None, # Retained for signature compatibility, but original use of -3 for axis in calculate_consensus not obvious
    edge_inclusion_threshold=0.95, 
    max_order=None,
):
    """
    Calculates the difference in Q-analysis vectors between consensus simplicial complexes
    derived from two sets of adjacency matrices.

    The `axis` parameter for `calculate_consensus_adjacency_matrix` in the original code
    was `axis=-3`. This implies `adjacency_matrices_a` and `_b` might have more than 3 dimensions.
    Typically, a collection of adjacency matrices is (n_samples, n_nodes, n_nodes).
    If so, `axis=0` would be for consensus over samples. Clarification on typical input shapes needed if -3 is critical.
    Assuming standard (n_samples, n_nodes, n_nodes) for now, so `axis=0` for consensus_calc.
    """
    # Determine the correct axis for mean calculation in calculate_consensus_adjacency_matrix.
    # If adj matrices are (samples, nodes, nodes), axis=0 is typical.
    consensus_axis = 0 # Default if axis is None or not otherwise determined
    if axis is not None:
        consensus_axis = axis # Use provided axis if specified, user needs to ensure it's correct.

    consensus_adj_a = calculate_consensus_adjacency_matrix(adjacency_matrices_a, edge_inclusion_threshold, axis=consensus_axis)
    consensus_adj_b = calculate_consensus_adjacency_matrix(adjacency_matrices_b, edge_inclusion_threshold, axis=consensus_axis)
    
    # adj_matrices_to_q_analysis_vectors expects input of shape (n_samples, n_nodes, n_nodes)
    # or (n_nodes, n_nodes) if it's a single matrix to be wrapped.
    # If consensus_adj_a is 2D (single consensus matrix), wrap it to be (1, N, N)
    if consensus_adj_a.ndim == 2:
        consensus_adj_a_batch = consensus_adj_a[None, ...]
    else: # Assumes it's already batched if > 2D, e.g. (batch_of_consensus_adj, N, N)
        consensus_adj_a_batch = consensus_adj_a
        
    if consensus_adj_b.ndim == 2:
        consensus_adj_b_batch = consensus_adj_b[None, ...]
    else:
        consensus_adj_b_batch = consensus_adj_b

    # adj_matrices_to_q_analysis_vectors returns a 3D NumPy array (n_samples, n_orders, n_metrics)
    q_metrics_a_3d = adj_matrices_to_q_analysis_vectors(consensus_adj_a_batch, max_order=max_order)
    q_metrics_b_3d = adj_matrices_to_q_analysis_vectors(consensus_adj_b_batch, max_order=max_order)

    # If inputs were single consensus matrices, results are (1, n_orders, n_metrics).
    # We want the difference of the 2D arrays (n_orders, n_metrics).
    if q_metrics_a_3d.shape[0] == 1:
        q_metrics_a_2d = q_metrics_a_3d[0]
    else:
        # If multiple consensus matrices were processed (e.g. if input was 4D and consensus_adj became 3D)
        # then this requires clarification on how to compute a single difference.
        # Assuming for now we want the first (or only) sample if batched.
        # This might need to be an average or sum of differences if multiple consensus comparisons are made.
        # print("Warning: consensus_statistic received multiple consensus Q-metrics for group A. Using first.")
        q_metrics_a_2d = q_metrics_a_3d[0] # Or handle error/averaging

    if q_metrics_b_3d.shape[0] == 1:
        q_metrics_b_2d = q_metrics_b_3d[0]
    else:
        # print("Warning: consensus_statistic received multiple consensus Q-metrics for group B. Using first.")
        q_metrics_b_2d = q_metrics_b_3d[0]

    difference = q_metrics_a_2d - q_metrics_b_2d
    difference[np.isnan(difference)] = 0 # As per original
    return difference

def difference_statistic(
    incidence_a: np.ndarray, 
    incidence_b: np.ndarray, 
    axis=None, # Retained for signature, but not used in current refactoring logic
    max_order=None, # q-level
    # These were unused in the original snippet for this function's logic
    # number_of_nodes=1, 
    # simplicial_set=set(), 
):
    """
    Calculates the difference in Q-analysis vectors between two simplicial structures
    provided as incidence matrices (or batches of them).

    `incidence_a` and `incidence_b` are expected to be incidence matrices (simplices x vertices)
    or batches of them (n_samples, n_simplices, n_vertices).
    The original code used `incidence_a.T` with `IncidenceSimplicialComplex`, suggesting inputs
    might have been (vertices x simplices). This implementation assumes (simplices x vertices)
    which is more standard for an incidence matrix input directly representing simplices.
    If input is (vertices x simplices), it should be transposed before this function or handled within.
    Let's assume input `incidence_a` is (simplices x vertices) or (batch, simplices, vertices).
    """
    
    def process_single_or_batch_incidence(incidence_data: np.ndarray, complex_max_order: int | None) -> list[np.ndarray]:
        """Converts one or a batch of incidence matrices to list of 2D NumPy SV arrays."""
        np_sv_list = []
        if incidence_data.ndim == 2: # Single incidence matrix (simplices x vertices)
            # Convert incidence matrix to list of simplices (sets of vertex indices)
            simplices = [set(np.where(row == 1)[0]) for row in incidence_data]
            complex_inst = IncidenceSimplicialComplexDev(simplices)
            gp = complex_inst.q_analysis_vectors(desired_vectors=VECTORS)
            np_sv = _convert_graded_parameters_to_numpy(gp, complex_inst.order())
            np_sv_list.append(np_sv)
        elif incidence_data.ndim == 3: # Batch of incidence matrices (batch, simplices, vertices)
            for inc_matrix_2d in incidence_data:
                simplices = [set(np.where(row == 1)[0]) for row in inc_matrix_2d]
                complex_inst = IncidenceSimplicialComplexDev(simplices)
                gp = complex_inst.q_analysis_vectors(desired_vectors=VECTORS)
                np_sv = _convert_graded_parameters_to_numpy(gp, complex_inst.order())
                np_sv_list.append(np_sv)
        else:
            raise ValueError("Incidence data must be 2D (single) or 3D (batch).")
        return np_sv_list

    # Process incidence_a and incidence_b to get lists of 2D NumPy arrays of their structure vectors
    # The `max_order` parameter for `pad_structure_vectors` is a q-level.
    # It's better to determine the max_order from the data itself if not provided,
    # or use the provided one to ensure consistent length for padding.
    
    # We need to determine an overall max_order for padding IF max_order is not given.
    # This requires creating complexes first to find their orders.
    # For now, `process_single_or_batch_incidence` uses the complex's own order for conversion.
    # `pad_structure_vectors` will then align them based on the passed `max_order` or its own inference.

    list_np_sv_a = process_single_or_batch_incidence(incidence_a, max_order)
    list_np_sv_b = process_single_or_batch_incidence(incidence_b, max_order)

    # Pad all these NumPy arrays together to ensure they have the same number of rows (orders)
    # `pad_structure_vectors` pads a list of 2D arrays into a 3D array.
    # (n_total_samples, n_orders_after_padding, n_metrics)
    all_np_svs_padded_3d = pad_structure_vectors(list_np_sv_a + list_np_sv_b, max_order=max_order)
    
    len_a_samples = len(list_np_sv_a)
    
    # Separate back into A and B groups, now as 3D arrays (or potentially 2D if only 1 sample in a group and unbatched)
    metrics_a_padded_3d = all_np_svs_padded_3d[:len_a_samples]
    metrics_b_padded_3d = all_np_svs_padded_3d[len_a_samples:]
    
    # The original code returned a single difference array.
    # This implies an aggregation (e.g., mean) if there were multiple samples in A and B.
    # If incidence_a and incidence_b were single 2D matrices, then metrics_a/b_padded_3d are (1, orders, metrics).
    # If they were batches, then (batch_size, orders, metrics).
    # Let's assume we need to average if batched, then difference of averages.
    # Or if only one sample, just use that.

    if metrics_a_padded_3d.shape[0] > 1:
        mean_metrics_a = np.nanmean(metrics_a_padded_3d, axis=0) # Average over samples
    elif metrics_a_padded_3d.shape[0] == 1:
        mean_metrics_a = metrics_a_padded_3d[0] # Single sample, take it directly
    else: # No samples in A (e.g., empty input incidence_a)
        # Determine shape for empty result based on B or max_order
        num_orders_fallback = all_np_svs_padded_3d.shape[1] if all_np_svs_padded_3d.ndim==3 and all_np_svs_padded_3d.shape[1]>0 else (max_order + 1 if max_order is not None and max_order >=0 else 0)
        num_metrics_fallback = len(VECTORS)
        mean_metrics_a = np.full((num_orders_fallback, num_metrics_fallback), 0.0, dtype=float) # or np.nan

    if metrics_b_padded_3d.shape[0] > 1:
        mean_metrics_b = np.nanmean(metrics_b_padded_3d, axis=0)
    elif metrics_b_padded_3d.shape[0] == 1:
        mean_metrics_b = metrics_b_padded_3d[0]
    else: # No samples in B
        num_orders_fallback = all_np_svs_padded_3d.shape[1] if all_np_svs_padded_3d.ndim==3 and all_np_svs_padded_3d.shape[1]>0 else (max_order + 1 if max_order is not None and max_order >=0 else 0)
        num_metrics_fallback = len(VECTORS)
        mean_metrics_b = np.full((num_orders_fallback, num_metrics_fallback), 0.0, dtype=float)

    difference = mean_metrics_a - mean_metrics_b
    difference[np.isnan(difference)] = 0 # As per original behavior
    return difference