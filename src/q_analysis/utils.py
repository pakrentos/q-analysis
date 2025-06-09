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
from q_analysis.simplicial_complex import IncidenceSimplicialComplex, VECTORS

def adj_matrices_to_q_analysis_vectors(adj_matrices: np.ndarray, max_order=None, fill_value=np.nan) -> np.ndarray:
    """
    Compute q-analysis structure vectors for each provided adjacency matrix.
    The output is a NumPy array of shape (n_samples, n_orders, n_metrics),
    where n_metrics corresponds to len(VECTORS).
    """
    structure_vectors_list_np = [] # This will be a list of 2D numpy arrays
    for adj in adj_matrices:
        complex_instance = IncidenceSimplicialComplex.from_adjacency_matrix(adj)
        # q_analysis_vectors returns a GradedParameters object.
        # We specify desired_vectors=VECTORS to ensure consistency in output columns.
        graded_params_obj = complex_instance.graded_parameters(desired_vectors=VECTORS) 

        current_complex_order = complex_instance.order()
        num_rows_for_current_complex = current_complex_order + 1 if current_complex_order != -1 else 0
        
        # Initialize an array for the current complex's vectors
        # Shape: (num_rows_for_current_complex, len(VECTORS))
        complex_vectors_np = np.full((num_rows_for_current_complex, len(VECTORS)), fill_value, dtype=float)

        if current_complex_order != -1: # Only populate if the complex is not empty
            for i, vec_name in enumerate(VECTORS):
                param_set = graded_params_obj.get_parameter(vec_name)
                if param_set and len(param_set.values) == num_rows_for_current_complex:
                    complex_vectors_np[:, i] = param_set.values.astype(float)
                elif param_set and len(param_set.values) != num_rows_for_current_complex :
                    # This case indicates an inconsistency that should ideally be handled
                    # within GradedParameterSet or q_analysis_vectors ensuring values array matches q_max.
                    # For robustness here, we are already filling with `fill_value`.
                    # We could print a warning.
                    # print(f"Warning: Mismatch in length for vector {vec_name} in adj_matrices_to_q_analysis_vectors.")
                    # Ensure shorter vectors are padded if they occur.
                    if len(param_set.values) > 0 and len(param_set.values) < num_rows_for_current_complex:
                         complex_vectors_np[:len(param_set.values), i] = param_set.values.astype(float)
                    # If longer, it's an issue too, but stack would fail later if not handled.
                    # The pre-fill with fill_value covers cases where param_set.values is empty or shorter.

        structure_vectors_list_np.append(complex_vectors_np)
    
    # pad_structure_vectors now takes this list of 2D arrays and returns a single 3D array
    return pad_structure_vectors(structure_vectors_list_np, max_order=max_order, fill_value=fill_value)

def pad_structure_vectors(structure_vectors_list_np: list[np.ndarray], max_order=None, fill_value=np.nan) -> np.ndarray:
    """
    Pad a list of 2D structure vector arrays (each being [n_q, n_metrics])
    to a consistent 3D shape (n_samples, max_effective_order_rows, n_metrics)
    and apply max_order (q-level) limit if specified.
    """
    num_samples = len(structure_vectors_list_np)
    num_metrics = len(VECTORS) # Assuming all arrays will conform to VECTORS count

    if num_samples == 0:
        # max_order is q-level, so rows = max_order + 1. If max_order is None, default to 0 rows.
        effective_rows = 0
        if max_order is not None:
            effective_rows = max_order + 1 if max_order >=0 else 0
        return np.empty((0, effective_rows, num_metrics), dtype=float)

    # Determine max_inferred_rows (max number of q-levels + 1 across all samples)
    max_inferred_rows = 0
    if num_samples > 0:
        valid_rows = [arr.shape[0] for arr in structure_vectors_list_np if arr.ndim == 2]
        if valid_rows:
            max_inferred_rows = max(valid_rows)
        # if all arrays are empty or not 2D, max_inferred_rows remains 0.
    
    # Determine the effective number of rows for the output array
    # max_order is a q-value. Number of rows will be max_order + 1.
    if max_order is not None: # max_order is specified (q-level)
        effective_rows = max_order + 1 if max_order >=0 else 0
        # The final array will have `effective_rows`.
        # Padding inside the loop will be up to the greater of `max_inferred_rows` or `effective_rows`
        # to ensure all data is captured before potential final truncation.
        padding_target_rows = max(max_inferred_rows, effective_rows)
    else: # max_order is None, use inferred max
        effective_rows = max_inferred_rows
        padding_target_rows = max_inferred_rows

    padded_arrays_collector = []
    for arr_2d in structure_vectors_list_np:
        if arr_2d.ndim == 2:
            current_rows, current_metrics = arr_2d.shape
        elif arr_2d.ndim == 1 and arr_2d.shape[0] == 0 and num_metrics > 0 : # special case like np.array([]) from empty complex, treat as (0, num_metrics)
            current_rows, current_metrics = 0, num_metrics
            arr_2d = np.empty((0,num_metrics), dtype=float) # Reshape for consistency
        else: # Unexpected shape, fill with NaNs
            # print(f"Warning: Unexpected array shape {arr_2d.shape} in pad_structure_vectors. Filling.")
            padded_arr = np.full((padding_target_rows, num_metrics), fill_value, dtype=float)
            padded_arrays_collector.append(padded_arr)
            continue

        # Pad or truncate metrics dimension (columns)
        if current_metrics < num_metrics:
            col_padding = np.full((current_rows, num_metrics - current_metrics), fill_value, dtype=float)
            arr_metric_adjusted = np.hstack((arr_2d, col_padding))
        elif current_metrics > num_metrics:
            arr_metric_adjusted = arr_2d[:, :num_metrics]
        else:
            arr_metric_adjusted = arr_2d
        
        # Pad rows
        row_pad_length = padding_target_rows - arr_metric_adjusted.shape[0]
        if row_pad_length > 0:
            row_padding_shape = ((0, row_pad_length), (0,0))
            padded_arr = np.pad(arr_metric_adjusted, row_padding_shape, mode='constant', constant_values=fill_value)
        elif row_pad_length < 0: # Should not happen if padding_target_rows is max
            padded_arr = arr_metric_adjusted[:padding_target_rows, :]
        else:
            padded_arr = arr_metric_adjusted
            
        padded_arrays_collector.append(padded_arr)
    
    # Stack to form 3D array (n_samples, padding_target_rows, num_metrics)
    if not padded_arrays_collector: # handles num_samples = 0 again, though covered
         # effective_rows determined earlier based on max_order
        return np.empty((0, effective_rows, num_metrics), dtype=float)

    stacked_3d_array = np.array(padded_arrays_collector)

    # Final slice to effective_rows if max_order was specified and led to fewer rows than some items had.
    # Or if max_order was None, effective_rows is max_inferred_rows, so this slice is benign.
    # Or if max_order led to more rows, array was padded to it.
    return stacked_3d_array[:, :effective_rows, :]

def calculate_consensus_adjacency_matrix(adjacency_matrices: np.ndarray, edge_inclusion_threshold=0.95, axis=0):
    consensus_adjacency_matrix = adjacency_matrices.mean(axis=axis) > edge_inclusion_threshold
    return consensus_adjacency_matrix.astype(int)


