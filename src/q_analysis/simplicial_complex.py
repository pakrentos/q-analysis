import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from q_analysis.q_analysis import py_find_hierarchical_q_components


VECTORS = ['FSV', 'SSV', 'TSV', 'Topological Entropy', 'Simplex Count', 'Shared Faces Count']

ABBREVIATIONS_MAP = {
    'fsv': 'FSV',
    'ssv': 'SSV',
    'tsv': 'TSV',
    'entropy': 'Topological Entropy',
    'simplices': 'Simplex Count',
    'faces': 'Shared Faces Count',
}


class GradedParameterSet:
    """
    Represents a parameter set graded by q (order/dimension).
    Examples include FSV, SSV, TSV, Simplex Count, etc.
    """
    def __init__(self, name: str, values: np.ndarray, q_max: int):
        self.name = name
        self.values = np.asarray(values)
        self.q_max = q_max # Maximum q for which this parameter is defined

    def __repr__(self):
        return f"GradedParameterSet(name='{self.name}', q_max={self.q_max}, values={self.values})"

    def __len__(self):
        return len(self.values)

    def __getitem__(self, q):
        if not 0 <= q < len(self.values):
            raise IndexError(f"q value {q} is out of bounds for {self.name} (0 to {len(self.values) -1}).")
        return self.values[q]
    
    def to_dataframe(self):
        """Returns a Pandas DataFrame representation."""
        return pd.DataFrame({'q': np.arange(len(self.values)), self.name: self.values})
    
    def to_numpy(self, max_order: int | None = None):
        if max_order is None:
            max_order = self.q_max
        if max_order < self.q_max:
            return np.pad(self.values, (0, max_order + 1 - self.q_max), mode='constant', constant_values=np.nan)
        return self.values[:max_order + 1]


class GradedParameters:
    """
    Represents a collection of GradedParameterSet objects.
    This is typically the result of computing multiple Q-analysis vectors.
    """
    def __init__(self, parameters: dict[str, GradedParameterSet]):
        self.parameters = parameters # dict mapping parameter name to GradedParameterSet instance

    def __repr__(self):
        names = list(self.parameters.keys())
        return f"GradedParameters(parameters={names})"

    def __getitem__(self, name: str) -> GradedParameterSet:
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found.")
        return self.parameters[name]
    
    def get_parameter(self, name: str, default=None) -> GradedParameterSet | None:
        """Gets a parameter by name, returning default if not found."""
        return self.parameters.get(name, default)

    @property
    def names(self) -> list[str]:
        """Returns the names of the parameters stored."""
        return list(self.parameters.keys())

    def to_dataframe(self):
        """
        Returns a Pandas DataFrame representation of all parameters.
        """
        if not self.parameters:
            return pd.DataFrame(columns=['q', 'Vector', 'Value'])

        return pd.concat([
            param_set.to_dataframe().rename(columns={name: 'Value'}).assign(Vector=name)
            for name, param_set in self.parameters.items()
        ], ignore_index=True).reset_index(drop=True)
    
    def to_numpy(self, max_order: int | None = None):
        """
        Returns a NumPy array representation of all parameters.
        """
        if max_order is None:
            max_order = max(param_set.q_max for param_set in self.parameters.values())
        return np.array([param_set.to_numpy(max_order) for param_set in self.parameters.values()])


class NodeParameterSet:
    """
    Represents a parameter set graded by node/vertex.
    Example: Topological Dimensionality.
    """
    def __init__(self, name: str, values: np.ndarray, node_ids: np.ndarray | list | None = None):
        self.name = name
        self.values = np.asarray(values)
        if node_ids is None:
            self.node_ids = np.arange(len(values))
        else:
            self.node_ids = np.asarray(node_ids)
        
        if len(self.values) != len(self.node_ids):
            raise ValueError("Length of values and node_ids must match.")

    def __repr__(self):
        return f"NodeParameterSet(name='{self.name}', num_nodes={len(self.node_ids)})"
    
    def to_dataframe(self):
        """Returns a Pandas DataFrame representation."""
        return pd.DataFrame({'Node': self.node_ids, self.name: self.values})


class QAnalysisReport:
    """
    Encapsulates the comprehensive results of a Q-analysis, including
    q-graded parameters and node-graded parameters.
    """
    def __init__(self, 
                 q_graded_parameters: GradedParameters | None = None, 
                 node_graded_parameters: dict[str, NodeParameterSet] | None = None,
                 order: int | None = None): # Max order of the complex
        self.q_graded_parameters = q_graded_parameters if q_graded_parameters is not None else GradedParameters({})
        self.node_graded_parameters = node_graded_parameters if node_graded_parameters is not None else {}
        self.order = order # Max dimension of the complex

    def __repr__(self):
        q_param_names = self.q_graded_parameters.names
        node_param_names = list(self.node_graded_parameters.keys())
        return f"QAnalysisReport(order={self.order}, q_params={q_param_names}, node_params={node_param_names})"

    def get_q_parameter(self, name: str) -> GradedParameterSet | None:
        """Retrieves a q-graded parameter set by its name."""
        return self.q_graded_parameters.get_parameter(name)

    def get_node_parameter(self, name: str) -> NodeParameterSet | None:
        """Retrieves a node-graded parameter set by its name."""
        return self.node_graded_parameters.get(name)


class IncidenceSimplicialComplex:
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
        IncidenceSimplicialComplex
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
    
    def simplicial_connectivty(self):
        """Returns the simplicial connectivity matrix."""
        connectivity_matrix = np.zeros((len(self.simplices), len(self.simplices)))
        for i in range(len(self.simplices)):
            for j in range(i, len(self.simplices)):
                connectivity_matrix[i, j] = connectivity_matrix[j, i] = len(
                    self.simplices[i].intersection(self.simplices[j])
                )
        connectivity_matrix -= 1
        return connectivity_matrix

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
                self._order = -1
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

        occupation_counts = Counter()
        for idx in q_indices:
            simplex = self.simplices[idx]
            occupation_counts.update(simplex)

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

    def topological_dimensionality(self, node_names=None) -> NodeParameterSet:
        """
        Calculates the topological dimensionality for each vertex.
        This is defined as the maximum order (dimension) of a simplex to which the vertex belongs.

        Note: This definition differs from the original implementation's `incidence.sum(0)`,
        which represented the vertex degree in the incidence graph (number of simplices containing the vertex).

        Parameters:
        -----------
        node_names : list or array-like, optional
            Custom names for nodes (vertices) to use in the DataFrame output or NodeParameterSet.
            If None, default integer indices (from self.vertices) are used. 
            Length must match the number of unique vertices in the complex.

        Returns:
        --------
        NodeParameterSet 
            A NodeParameterSet object.
        """
        unique_vertices = self.vertices # Sorted list of unique vertex IDs
        if not unique_vertices: # Handle empty complex or complex with no vertices
            node_param_set = NodeParameterSet(name='Topological Dimensionality', values=np.array([]), node_ids=np.array([]))
            return node_param_set

        vertices_topological_dimensionality_map = {vertex: 0 for vertex in unique_vertices}
        for simplex in self.simplices:
            for vertex in simplex:
                if vertex in vertices_topological_dimensionality_map: # Ensure vertex is in our list
                    vertices_topological_dimensionality_map[vertex] += 1
        
        # Extract values in the order of unique_vertices to ensure consistency
        dim_values = np.array([vertices_topological_dimensionality_map[v] for v in unique_vertices])

        # Determine node identifiers for NodeParameterSet
        current_node_ids = node_names if node_names is not None else unique_vertices
        if node_names is not None and len(node_names) != len(unique_vertices):
            raise ValueError("Length of node_names must match the number of unique vertices in the complex.")

        node_param_set = NodeParameterSet(
            name='Topological Dimensionality', 
            values=dim_values, 
            node_ids=current_node_ids
        )

        return node_param_set


    def first_structure_vector(self) -> GradedParameterSet:
        """
        Calculates the First Structure Vector (FSV), Q.
        Q_k = number of k-connected components.

        Returns:
        --------
        GradedParameterSet
            A GradedParameterSet object for FSV.
        """
        max_q = self.order()
        if max_q == -1:
             return GradedParameterSet(name='FSV', values=np.array([], dtype=int), q_max=max_q)

        # Ensure connected components are computed and cached
        self._get_all_q_connected_components()

        fsv_values = [self.q_connected_components(q) for q in range(max_q + 1)]
        return GradedParameterSet(name='FSV', values=np.array(fsv_values, dtype=int), q_max=max_q)

    def simp_count(self) -> GradedParameterSet:
        """
        Counts the number of simplices for each order q (dimension).

        Returns:
        --------
        GradedParameterSet
            A GradedParameterSet object for Simplex Count.
        """
        max_q = self.order()
        if max_q == -1:
            return GradedParameterSet(name='Simplex Count', values=np.array([], dtype=int), q_max=max_q)

        orders = self.simplex_orders()
        # bincount counts occurrences of each non-negative integer value in 'orders'
        # minlength ensures the output array has size at least max_q + 1
        simps_per_q_count_values = np.bincount(orders[orders >= 0], minlength=max_q + 1) if orders.size > 0 else np.zeros(max_q + 1, dtype=int)
        return GradedParameterSet(name='Simplex Count', values=simps_per_q_count_values, q_max=max_q)

    def second_structure_vector(self) -> GradedParameterSet:
        """
        Calculates the Second Structure Vector (SSV), N.
        N_k = number of simplices of dimension >= k.

        Returns:
        --------
        GradedParameterSet
            A GradedParameterSet object for SSV.
        """
        simp_count_param = self.simp_count()
        if not simp_count_param.values.size:
            return GradedParameterSet(name='SSV', values=np.array([], dtype=int), q_max=simp_count_param.q_max)

        # N_k is the sum of counts for dimensions k, k+1, ... max_q
        # This is achieved by taking the cumulative sum from right-to-left (reverse, cumsum, reverse)
        ssv_values = np.cumsum(simp_count_param.values[::-1])[::-1]
        return GradedParameterSet(name='SSV', values=ssv_values, q_max=simp_count_param.q_max)

    @staticmethod
    def _third_structure_vector(fsv: GradedParameterSet, ssv: GradedParameterSet) -> GradedParameterSet:
        """
        Calculates the Third Structure Vector (TSV), pi.
        pi_k = 1 - Q_k / N_k. If N_k = 0 (i.e., ssv_k = 0), pi_k is defined as 0.

        Parameters:
        -----------
        fsv : GradedParameterSet
            First Structure Vector (Q).
        ssv : GradedParameterSet
            Second Structure Vector (N).

        Returns:
        --------
        GradedParameterSet
            The Third Structure Vector (pi).
        """
        max_q = fsv.q_max # Assuming fsv and ssv will have same q_max

        if fsv.values.size == 0 or ssv.values.size == 0:
             return GradedParameterSet(name='TSV', values=np.array([], dtype=float), q_max=max_q)

        # Initialize pi_k = 1 for the 1 - (ratio) part of the formula.
        pi_values = np.ones_like(fsv.values, dtype=float)
        
        # Calculate pi_k = 1 - Q_k / N_k for N_k > 0 (ssv_k > 0)
        valid_indices = np.where(ssv.values > 0)[0]
        if valid_indices.size > 0: # Ensure indexing is safe if all ssv are zero
            pi_values[valid_indices] = 1.0 - fsv.values[valid_indices].astype(float) / ssv.values[valid_indices].astype(float)
        
        # Where N_k = 0, pi_k is defined as 0.
        pi_values[ssv.values == 0] = 0.0
        return GradedParameterSet(name='TSV', values=pi_values, q_max=max_q)


    def third_structure_vector(self) -> GradedParameterSet:
        """
        Calculates the Third Structure Vector (TSV), pi = 1 - Q/N.
        Uses the static method `_third_structure_vector` for the calculation logic.

        Returns:
        --------
        GradedParameterSet
            The Third Structure Vector (pi).
        """
        fsv_param = self.first_structure_vector()
        ssv_param = self.second_structure_vector()

        if len(fsv_param.values) != len(ssv_param.values):
             # This case indicates an internal inconsistency
             raise ValueError("FSV and SSV lengths do not match.")
        
        # q_max should be consistent from fsv_param and ssv_param
        return self._third_structure_vector(fsv_param, ssv_param)

    def shared_faces_count(self) -> GradedParameterSet:
        """
        Counts the number of shared faces for each order q.
        A shared face of order q between two distinct simplices sigma_1, sigma_2
        is their intersection, provided |sigma_1 intersect sigma_2| = q + 1.

        Returns:
        --------
        GradedParameterSet
            A GradedParameterSet object for Shared Faces Count.
        """
        max_q = self.order()
        if max_q == -1:
            return GradedParameterSet(name='Shared Faces Count', values=np.array([], dtype=int), q_max=max_q)

        shared_faces_values = np.zeros(max_q + 1, dtype=int)
        n_simplices = self.num_simplices

        # Iterate through all unique pairs of distinct simplices
        for i in range(n_simplices):
            for j in range(i + 1, n_simplices):
                intersection = self.simplices[i].intersection(self.simplices[j])
                face_order = len(intersection) - 1

                if 0 <= face_order <= max_q:
                    shared_faces_values[face_order] += 1
        return GradedParameterSet(name='Shared Faces Count', values=shared_faces_values, q_max=max_q)

    def topological_entropy(self, q: int) -> float:
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
        entropy_sum = -np.sum(p_i * np.log10(p_i))

        # Normalize by log10 of the number of occupied vertices
        normalization_factor = np.log10(num_occupied)
        if normalization_factor == 0: # Should be caught by num_occupied <= 1 check
             return np.nan

        normalized_entropy = entropy_sum / normalization_factor
        return normalized_entropy

    def full_topological_entropy(self) -> GradedParameterSet:
        """
        Calculates topological entropy for all dimensions q from 0 to max_order.

        Returns:
        --------
        GradedParameterSet
            A GradedParameterSet for Topological Entropy.
        """
        max_q = self.order()
        if max_q == -1:
            return GradedParameterSet(name='Topological Entropy', values=np.array([], dtype=float), q_max=max_q)

        entropies_values = [self.topological_entropy(q) for q in range(max_q + 1)]
        return GradedParameterSet(name='Topological Entropy', values=np.array(entropies_values, dtype=float), q_max=max_q)

    def full_char(self) -> QAnalysisReport:
        """
        Gathers all primary characteristic parameters computed by this class into a QAnalysisReport.

        Returns:
        --------
        QAnalysisReport
            A QAnalysisReport object containing:
            - q_graded_parameters: GradedParameters for FSV, SSV, TSV, Topological Entropy, 
                                   Simplex Count, Shared Faces Count.
            - node_graded_parameters: NodeParameterSet for Topological Dimensionality.
            - order: The maximum order (dimension) of the complex.
        """
        max_q = self.order()

        q_params_dict = {}
        node_params_dict = {}

        if max_q == -1:
            # For q-graded parameters
            q_params_dict['FSV'] = GradedParameterSet(name='FSV', values=np.array([], dtype=int), q_max=max_q)
            q_params_dict['SSV'] = GradedParameterSet(name='SSV', values=np.array([], dtype=int), q_max=max_q)
            q_params_dict['TSV'] = GradedParameterSet(name='TSV', values=np.array([], dtype=float), q_max=max_q)
            q_params_dict['Topological Entropy'] = GradedParameterSet(name='Topological Entropy', values=np.array([], dtype=float), q_max=max_q)
            q_params_dict['Simplex Count'] = GradedParameterSet(name='Simplex Count', values=np.array([], dtype=int), q_max=max_q)
            q_params_dict['Shared Faces Count'] = GradedParameterSet(name='Shared Faces Count', values=np.array([], dtype=int), q_max=max_q)
            
            # For node-graded parameters
            node_params_dict['Topological Dimensionality'] = NodeParameterSet(name='Topological Dimensionality', values=np.array([]), node_ids=np.array([]))
            
            q_graded_params = GradedParameters(q_params_dict)
            return QAnalysisReport(q_graded_parameters=q_graded_params, node_graded_parameters=node_params_dict, order=max_q)

        q_params_dict['FSV'] = self.first_structure_vector()
        q_params_dict['SSV'] = self.second_structure_vector()
        q_params_dict['TSV'] = self.third_structure_vector()
        q_params_dict['Topological Entropy'] = self.full_topological_entropy()
        q_params_dict['Simplex Count'] = self.simp_count()
        q_params_dict['Shared Faces Count'] = self.shared_faces_count()
        
        # Node-graded parameters
        node_params_dict['Topological Dimensionality'] = self.topological_dimensionality()
        
        q_graded_params = GradedParameters(q_params_dict)
        
        return QAnalysisReport(q_graded_parameters=q_graded_params, 
                               node_graded_parameters=node_params_dict, 
                               order=max_q)

    def graded_parameters(self, desired_vectors: list[str] | None = None) -> GradedParameters:
        """
        Computes and returns specified Q-analysis vectors and graded parameters as a GradedParameters object.
        """
        max_q = self.order()

        effective_vector_names_ordered = []
        source_selection_list = desired_vectors if desired_vectors is not None else VECTORS
        
        for v_name_or_abbr in source_selection_list:
            full_v_name = ABBREVIATIONS_MAP.get(v_name_or_abbr.lower(), v_name_or_abbr)
            if full_v_name in VECTORS and full_v_name not in effective_vector_names_ordered:
                effective_vector_names_ordered.append(full_v_name)
        
        calculated_parameters = {}

        if max_q == -1:
            for v_name in effective_vector_names_ordered:
                # Determine dtype based on vector name for empty case
                dtype = float if v_name == 'Topological Entropy' or v_name == 'TSV' else int
                calculated_parameters[v_name] = GradedParameterSet(name=v_name, values=np.array([], dtype=dtype), q_max=max_q)
            
            return GradedParameters(calculated_parameters)

        return GradedParameters({
            v_name: self._vector_method_map[v_name]()
            for v_name in effective_vector_names_ordered
            if v_name in self._vector_method_map
        })



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