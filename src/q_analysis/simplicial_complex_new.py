import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from q_analysis import py_find_hierarchical_q_components


VECTORS = ['FSV', 'SSV', 'TSV', 'Topological Entropy', 'Simplex Count', 'Shared Faces Count']

ABBREVIATIONS_MAP = {
    'fsv': 'FSV',
    'ssv': 'SSV',
    'tsv': 'TSV',
    'entropy': 'Topological Entropy',
    'simplices': 'Simplex Count',
    'faces': 'Shared Faces Count',
}

class IncidenceSimplicialComplexDev:
    """
    Represents a simplicial complex using a list of simplices (sets of vertices).
    """
    def __init__(self, simplices):
        """
        Initialize with a list of simplices.

        Parameters:
        -----------
        simplices : list of sets/lists/tuples
            A list where each element represents a simplex as a collection of vertex indices.
            Internal representation uses sets for efficiency.
        """
        self.simplices = [set(s) for s in simplices]
        self._order = None 
        self._simplex_orders = None 
        self._connected_components_cache = None 
        self._num_vertices = None

        # Map vector names to their corresponding methods
        self._vector_method_map = {
            'FSV': self.first_structure_vector,
            'SSV': self.second_structure_vector,
            'TSV': self.third_structure_vector,
            'Topological Entropy': self.full_topological_entropy,
            'Simplex Count': self.simp_count,
            'Shared Faces Count': self.shared_faces_count,
        }

    @classmethod
    def from_adjacency_matrix(cls, adj_matrix):
        """
        Create a complex from a graph's adjacency matrix by finding maximal cliques.

        Parameters:
        -----------
        adj_matrix : numpy.ndarray or scipy.sparse matrix
            The adjacency matrix of the graph.

        Returns:
        --------
        IncidenceSimplicialComplexDev
            A new complex instance built from the cliques of the graph.
        """
        graph = nx.Graph(adj_matrix)
        # find_cliques returns iterators of nodes, convert to sets
        cliques = [set(clique) for clique in nx.clique.find_cliques(graph)]
        return cls(cliques)

    @property
    def num_simplices(self):
        """Returns the number of simplices in the complex."""
        return len(self.simplices)

    @property
    def num_vertices(self):
        """
        Calculates and caches the number of vertices (highest vertex index + 1).
        """
        if self._num_vertices is None:
            if not self.simplices:
                self._num_vertices = 0
            else:
                max_vertex = -1
                for s in self.simplices:
                    if s: # Check if simplex is not empty
                        current_max = max(s)
                        if current_max > max_vertex:
                            max_vertex = current_max
                self._num_vertices = max_vertex + 1
        return self._num_vertices
    
    @property
    def vertices(self):
        """Returns the list of vertices in the complex."""
        return sorted(
            list(
                set().union(*[set(s) for s in self.simplices])
            )
        )


    def simplex_orders(self):
        """
        Calculates and caches the order (dimension) of each simplex.
        Order = |simplex| - 1.
        """
        if self._simplex_orders is None:
            self._simplex_orders = np.array([len(s) - 1 for s in self.simplices], dtype=int)
        return self._simplex_orders

    def order(self):
        """
        Calculates and caches the maximum order (dimension) of the complex.
        This is the maximum order among all its simplices.
        """
        if self._order is None:
            if not self.simplices:
                self._order = -1 # Define order of empty complex as -1
            else:
                orders = self.simplex_orders()
                self._order = np.max(orders) if orders.size > 0 else -1
        return self._order

    def _get_all_q_connected_components(self):
        """
        Calls the (placeholder) Rust function to find connected components for all q levels
        and caches the result. 
        Returns a list where the element at index `q` is a list of sets of simplex indices,
        representing the q-connected components.
        """
        if self._connected_components_cache is None:
            # Pass list of sets directly if supported by the binding
            simplices_as_lists = [list(s) for s in self.simplices]
            self._connected_components_cache = py_find_hierarchical_q_components(simplices_as_lists)
        return self._connected_components_cache

    def q_connected_components_labeled(self, q):
        """
        Returns the list of connected components (as sets of simplex indices) for a given q,
        retrieved from the cache.

        Parameters:
        -----------
        q : int
            The dimension level for connectivity.

        Returns:
        --------
        list[set[int]]
            A list where each set contains the indices of simplices in a q-connected component.
            Returns an empty list if no components exist at level q (assuming q is a valid index
            for the cached list of components).
        """
        all_components = self._get_all_q_connected_components()
        return all_components[q]

    def q_connected_components(self, q):
        """
        Returns the *number* of connected components for a given q.

        Parameters:
        -----------
        q : int
            The dimension level for connectivity.

        Returns:
        --------
        int
            The number of q-connected components.
        """
        return len(self.q_connected_components_labeled(q))

    # --- Methods requiring re-implementation without incidence matrix ---

    def q_simplices_indices(self, q):
         """Returns indices of simplices with order exactly q."""
         orders = self.simplex_orders()
         return np.where(orders == q)[0]

    def q_upper_simplices_indices(self, q):
         """Returns indices of simplices with order >= q."""
         orders = self.simplex_orders()
         return np.where(orders >= q)[0]

    def q_occupation(self, q):
        """
        Counts how many simplices of exactly order q each vertex belongs to.

        Parameters:
        -----------
        q : int
            The simplex dimension.

        Returns:
        --------
        numpy.ndarray
            An array where the index corresponds to the vertex ID and the value
            is the count of q-simplices containing that vertex. The array length
            is determined by `self.num_vertices`.
        """
        q_indices = self.q_simplices_indices(q)
        n_verts = self.num_vertices
        if n_verts == 0 or len(q_indices) == 0:
            return np.zeros(n_verts, dtype=int)

        # Use Counter for potentially sparse vertex indices
        occupation_counts = Counter()
        for idx in q_indices:
            simplex = self.simplices[idx]
            # Update counts for vertices in this simplex
            occupation_counts.update(simplex)

        # Convert Counter to a dense numpy array of the correct size
        occ_array = np.zeros(n_verts, dtype=int)
        for vertex, count in occupation_counts.items():
            if 0 <= vertex < n_verts:
                 occ_array[vertex] = count
        return occ_array


    def q_occupation_prob(self, q):
        """
        Calculates the probability distribution of vertex occupation for simplices of order q.
        The probability for a vertex is its q-occupation count divided by the total q-occupation count
        across all vertices.

        Parameters:
        -----------
        q : int
            The simplex dimension.

        Returns:
        --------
        numpy.ndarray
            An array of probabilities, indexed by vertex ID. Returns an array of zeros if
            the total occupation count is zero.
        """
        occupation = self.q_occupation(q)
        total_occupation = np.sum(occupation)

        if total_occupation == 0:
            return np.zeros_like(occupation, dtype=float)

        return occupation / total_occupation

    def topological_dimensionality(self, as_dataframe=False, node_names=None):
        """
        Calculates the topological dimensionality for each vertex.
        This is defined as the maximum order (dimension) of a simplex to which the vertex belongs.

        Note: This definition differs from the original implementation's `incidence.sum(0)`,
        which represented the vertex degree in the incidence graph (number of simplices containing the vertex).

        Parameters:
        -----------
        as_dataframe : bool, optional
            If True, returns the result as a Pandas DataFrame. Defaults to False.
        node_names : list or array-like, optional
            Custom names for nodes (vertices) to use in the DataFrame output.
            If None, default integer indices are used. Length must match `self.num_vertices`.

        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            If `as_dataframe` is False, returns a NumPy array where the index is the vertex ID
            and the value is its topological dimensionality (max simplex order).
            If `as_dataframe` is True, returns a DataFrame with 'Node' and
            'Topological Dimensionality' columns.
        """
        vertices = self.vertices
        vertices_topological_dimensionality = {k: 0 for k in vertices}
        for simplex in self.simplices:
            for vertex in simplex:
                vertices_topological_dimensionality[vertex] += 1

        if as_dataframe:
            if node_names is None:
                node_names = vertices
            
            return pd.DataFrame({'Node': node_names, 'Topological Dimensionality': [vertices_topological_dimensionality[v] for v in vertices]})
        else:
            return np.array([vertices_topological_dimensionality[v] for v in vertices])


    def first_structure_vector(self):
        """
        Calculates the First Structure Vector (FSV), Q.
        Q_k = number of k-connected components.

        Returns:
        --------
        numpy.ndarray
            An array where the index `k` corresponds to the q-level, and the value
            is the number of q-connected components, Q_k. The length is `self.order() + 1`.
        """
        max_q = self.order()
        if max_q == -1:
             return np.array([], dtype=int)

        # Ensure connected components are computed and cached
        self._get_all_q_connected_components()

        fsv = [self.q_connected_components(q) for q in range(max_q + 1)]
        return np.array(fsv, dtype=int)

    def simp_count(self):
        """
        Counts the number of simplices for each order q (dimension).

        Returns:
        --------
        numpy.ndarray
            An array where the index `q` corresponds to the dimension, and the value
            is the number of simplices of that dimension. The length is `self.order() + 1`.
        """
        max_q = self.order()
        if max_q == -1:
            return np.array([], dtype=int)

        orders = self.simplex_orders()
        # bincount counts occurrences of each non-negative integer value in 'orders'
        # minlength ensures the output array has size at least max_q + 1
        simps_per_q_count = np.bincount(orders[orders >= 0], minlength=max_q + 1) if orders.size > 0 else np.zeros(max_q + 1, dtype=int)
        return simps_per_q_count

    def second_structure_vector(self):
        """
        Calculates the Second Structure Vector (SSV), N.
        N_k = number of simplices of dimension >= k.

        Returns:
        --------
        numpy.ndarray
            An array where the index `k` corresponds to the q-level, and the value
            is the number of simplices with dimension `q >= k`. The length is `self.order() + 1`.
        """
        simps_per_q_count = self.simp_count()
        if len(simps_per_q_count) == 0:
            return np.array([], dtype=int)

        # N_k is the sum of counts for dimensions k, k+1, ... max_q
        # This is achieved by taking the cumulative sum from right-to-left (reverse, cumsum, reverse)
        return np.cumsum(simps_per_q_count[::-1])[::-1]

    @staticmethod
    def _third_structure_vector(fsv, ssv):
        """
        Calculates the Third Structure Vector (TSV), pi.
        pi_k = 1 - Q_k / N_k. If N_k = 0 (i.e., ssv_k = 0), pi_k is defined as 0.

        Parameters:
        -----------
        fsv : numpy.ndarray
            First Structure Vector (Q).
        ssv : numpy.ndarray
            Second Structure Vector (N).

        Returns:
        --------
        numpy.ndarray
            The Third Structure Vector (pi).
        """
        # Ensure input arrays are numpy arrays
        fsv = np.asarray(fsv)
        ssv = np.asarray(ssv)

        if fsv.size == 0 or ssv.size == 0:
             return np.array([], dtype=float)

        # Initialize pi_k = 1 for the 1 - (ratio) part of the formula.
        pi = np.ones_like(fsv, dtype=float)
        
        # Calculate pi_k = 1 - Q_k / N_k for N_k > 0 (ssv_k > 0)
        valid_indices = np.where(ssv > 0)[0]
        if valid_indices.size > 0: # Ensure indexing is safe if all ssv are zero
            pi[valid_indices] = 1.0 - fsv[valid_indices].astype(float) / ssv[valid_indices].astype(float)
        
        # Where N_k = 0, pi_k is defined as 0.
        pi[ssv == 0] = 0.0
        return pi


    def third_structure_vector(self):
        """
        Calculates the Third Structure Vector (TSV), pi = 1 - Q/N.
        Uses the static method `_third_structure_vector` for the calculation logic.

        Returns:
        --------
        numpy.ndarray
            The Third Structure Vector (pi).
        """
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()

        if len(fsv) != len(ssv):
             # This case indicates an internal inconsistency
             raise ValueError("FSV and SSV lengths do not match.")

        return self._third_structure_vector(fsv, ssv)

    def shared_faces_count(self):
        """
        Counts the number of shared faces for each order q.
        A shared face of order q between two distinct simplices sigma_1, sigma_2
        is their intersection, provided |sigma_1 intersect sigma_2| = q + 1.

        Returns:
        --------
        numpy.ndarray
            An array where the index `q` is the face dimension, and the value is
            the count of shared faces of that dimension. Length is `self.order() + 1`.
        """
        max_q = self.order()
        if max_q == -1:
            return np.array([], dtype=int)

        shared_faces = np.zeros(max_q + 1, dtype=int)
        n_simplices = self.num_simplices

        # Iterate through all unique pairs of distinct simplices
        for i in range(n_simplices):
            for j in range(i + 1, n_simplices):
                intersection = self.simplices[i].intersection(self.simplices[j])
                face_order = len(intersection) - 1

                if 0 <= face_order <= max_q:
                    shared_faces[face_order] += 1
        return shared_faces

    def topological_entropy(self, q):
        """
        Calculates the normalized topological entropy for a given dimension q.
        Entropy = - sum(p_i * log10(p_i)) / log10(N_occupied), where p_i is the
        q-occupation probability for vertex i, summed over vertices with non-zero
        q-occupation, and N_occupied is the number of such vertices.

        Parameters:
        -----------
        q : int
            The dimension level.

        Returns:
        --------
        float
            The normalized topological entropy. Returns NaN if the number of
            occupied vertices is 0 or 1 (entropy is undefined or trivial).
        """
        occupation = self.q_occupation(q)
        # Filter out vertices with zero occupation count
        occupied_vertices_counts = occupation[occupation > 0]

        num_occupied = len(occupied_vertices_counts)

        if num_occupied <= 1:
            return np.nan # Consistent with original implementation

        # Calculate probabilities p_i based *only* on the occupied vertices
        total_occupied_count = np.sum(occupied_vertices_counts)
        if total_occupied_count == 0: # Should not be reached if num_occupied > 0
             return np.nan

        p_i = occupied_vertices_counts / total_occupied_count

        # Calculate Shannon entropy term: sum(-p_i * log10(p_i))
        # Use np.log10 for base-10 logarithm as in the original code
        entropy_sum = -np.sum(p_i * np.log10(p_i))

        # Normalize by log10 of the number of occupied vertices
        normalization_factor = np.log10(num_occupied)
        if normalization_factor == 0: # Should be caught by num_occupied <= 1 check
             return np.nan

        normalized_entropy = entropy_sum / normalization_factor
        return normalized_entropy

    def full_topological_entropy(self):
        """
        Calculates topological entropy for all dimensions q from 0 to max_order.

        Returns:
        --------
        numpy.ndarray
            An array containing the topological entropy for each q.
            Length is `self.order() + 1`.
        """
        max_q = self.order()
        if max_q == -1:
            return np.array([], dtype=float)

        entropies = [self.topological_entropy(q) for q in range(max_q + 1)]
        return np.array(entropies, dtype=float)

    def full_char(self):
        """
        Gathers all primary characteristic vectors and dimensions computed by this class.

        Note: The 'topological dimensionality' included here is the max simplex order per vertex,
        which differs from the 'incidence degree' used in the original `IncidenceSimplicialComplex`.

        Returns:
        --------
        tuple
            A tuple containing:
            (fsv, ssv, tsv, top_entropies, top_dims_per_vertex, simp_counts, shared_faces_count, order_vector)
            - fsv: First Structure Vector (Q_k)
            - ssv: Second Structure Vector (N_k)
            - tsv: Third Structure Vector (pi_k)
            - top_entropies: Topological Entropy per q
            - top_dims_per_vertex: Max simplex order containing each vertex
            - simp_counts: Number of simplices per order q
            - shared_faces_count: Number of shared faces per order q
            - order_vector: Placeholder vector of ones (matches original behavior)
        """
        max_q = self.order()
        if max_q == -1:
            # Return empty structures consistent with vector lengths
            empty_float = np.array([], dtype=float)
            empty_int = np.array([], dtype=int)
            return (empty_int, empty_int, empty_float, empty_float,
                    np.array([], dtype=int), 
                    empty_int, empty_int, empty_float)

        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        tsv = self.third_structure_vector()
        top_entropies = self.full_topological_entropy()
        top_dims_per_vertex = self.topological_dimensionality() # Max order per vertex
        simp_counts = self.simp_count()
        shared_faces_count = self.shared_faces_count()

        # Placeholder 'order' vector from original code (array of ones of length max_q + 1)
        order_vector = np.ones(max_q + 1, dtype=float)

        return fsv, ssv, tsv, top_entropies, top_dims_per_vertex, simp_counts, shared_faces_count, order_vector

    def q_analysis_vectors(self, as_dataframe=False, with_order_mask=False, desired_vectors: list[str] | None = None):
        """
        Computes and returns specified Q-analysis vectors as a NumPy array or DataFrame.

        Parameters:
        -----------
        as_dataframe : bool, optional
            If True, returns the result as a Pandas DataFrame in a long format
            with columns 'q', 'Vector', 'Value'. Defaults to False.
        with_order_mask : bool, optional
            If True, appends a column of ones (labeled 'Order' if using DataFrame output,
            otherwise just appended) to the result array. Defaults to False.
        desired_vectors : list[str] | None, optional
            A list of strings specifying which Q-analysis vectors to compute.
            Can use full names (e.g., 'FSV', 'Topological Entropy') or abbreviations
            (e.g., 'fsv', 'entropy'). Available abbreviations are:
            'fsv': 'FSV', 'ssv': 'SSV', 'tsv': 'TSV',
            'entropy': 'Topological Entropy', 'simplices': 'Simplex Count',
            'faces': 'Shared Faces Count'.
            If None, all available vectors defined in the global `VECTORS` constant are computed.
            The order of vectors in the output will match the order in this list 
            (for valid and known vectors/abbreviations), or the order in the `VECTORS` 
            constant if this parameter is None.
            Defaults to None.

        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            If `as_dataframe` is False: Returns a NumPy array of shape (max_q + 1, num_selected_vectors).
            Columns correspond to the selected vectors (using full names) (plus optionally the order mask).
            If `as_dataframe` is True: Returns a Pandas DataFrame.
        """
        max_q = self.order()

        # Determine the list of vector names to process and their order, resolving abbreviations.
        effective_vector_names_ordered = []
        source_selection_list = desired_vectors if desired_vectors is not None else VECTORS
        
        for v_name_or_abbr in source_selection_list:
            # Resolve abbreviation to full name if applicable, otherwise use as is (case-sensitive for full names)
            full_v_name = ABBREVIATIONS_MAP.get(v_name_or_abbr.lower(), v_name_or_abbr) # .lower() for robust abbr matching
            
            # Check if the (potentially expanded) name is a known vector and not already added
            if full_v_name in VECTORS and full_v_name not in effective_vector_names_ordered:
                effective_vector_names_ordered.append(full_v_name)
        
        # Handle empty complex case
        if max_q == -1:
            output_column_names = list(effective_vector_names_ordered)
            num_data_cols = len(output_column_names)

            if with_order_mask:
                output_column_names.append('Order')
                num_data_cols +=1
            
            empty_array = np.empty((0, num_data_cols), dtype=float)
            if as_dataframe:
                return self.q_vectors_array_to_dataframe(empty_array, vector_names=output_column_names)
            else:
                return empty_array

        # Calculate requested vectors for non-empty complex
        result_stack_list = []
        
        for v_name in effective_vector_names_ordered:
            # v_name is guaranteed to be in self._vector_method_map due to how 
            # effective_vector_names_ordered is constructed from VECTORS.
            method_to_call = self._vector_method_map[v_name]
            vec = method_to_call()
            result_stack_list.append(vec.astype(float))

        current_output_names = list(effective_vector_names_ordered)

        if not result_stack_list: # No valid vectors were selected or all computations resulted in empty
            if with_order_mask:
                order_mask_vec = np.ones(max_q + 1, dtype=float)
                result_array = order_mask_vec.reshape(-1, 1)
                current_output_names = ['Order']
            else:
                result_array = np.empty((max_q + 1, 0), dtype=float)
                current_output_names = []
        else:
            result_stack_np = np.stack(result_stack_list, axis=0)
            if with_order_mask:
                order_mask_vec = np.ones(max_q + 1, dtype=float)
                result_stack_np = np.vstack([result_stack_np, order_mask_vec])
                current_output_names.append('Order')
            result_array = result_stack_np.T

        if as_dataframe:
            return self.q_vectors_array_to_dataframe(result_array, vector_names=current_output_names)
        else:
            return result_array

    @classmethod
    def q_vectors_array_to_dataframe(cls, q_vectors_array, vector_names=VECTORS, with_index=False):
        order = q_vectors_array.shape[0]
        index = pd.MultiIndex.from_product(
            [
                np.arange(order, dtype=int),
                vector_names
            ],
            names=['q', 'Vector']
        )
        df = pd.DataFrame(
            q_vectors_array.flatten(),
            index=index,
            columns=['Value']
        )
        if not with_index:
            df.reset_index(inplace=True)
        return df


    # --- Utility method retained from original ---
    @staticmethod
    def simplecies_to_incidence(simplices):
        """
        Convert a list of simplices (sets or lists of vertex indices)
        to a dense numpy incidence matrix. Retained as a utility function.

        Parameters:
        -----------
        simplices : list[set/list/tuple]
            A list where each element represents a simplex.

        Returns:
        --------
        numpy.ndarray
            The incidence matrix where rows are simplices and columns are vertices.
            Entry (i, j) is 1 if vertex j is in simplex i, 0 otherwise.
            Returns shape (0, 0) for empty input list.
            Returns shape (M, 0) if input has M empty simplices.
        """
        num_simplices = len(simplices)
        if num_simplices == 0:
            return np.empty((0, 0), dtype=int)

        all_vertices = set()
        for s in simplices:
             all_vertices.update(v for v in s if isinstance(v, (int, np.integer)))


        if not all_vertices:
             return np.zeros((num_simplices, 0), dtype=int)

        max_vertex = max(all_vertices)
        num_vertices = max_vertex + 1

        incidence = np.zeros((num_simplices, num_vertices), dtype=int)

        for i, simplex in enumerate(simplices):
            simplex_indices = [v for v in simplex if isinstance(v, (int, np.integer)) and 0 <= v < num_vertices]
            if simplex_indices:
                 incidence[i, simplex_indices] = 1

        return incidence