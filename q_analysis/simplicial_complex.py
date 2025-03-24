import numpy as np
from .connected_components import dynamic_method as find_connected_components
import networkx as nx
from scipy.sparse import csr_array
import pandas as pd

VECTORS = ['FSV', 'SSV', 'TSV', 'Topological Entropy', 'Simplex Count', 'Shared Faces Count']

class IncidenceSimplicialComplex:
    def __init__(self, incidence):
        self.incidence = incidence
        self._order = None

    @classmethod
    def simplecies_to_incidence(cls, simplecies):
        """
        Convert a list of simplices to an incidence matrix as a regular numpy array.
        
        Parameters:
        -----------
        simplecies : list of lists
            A list where each element is a list of vertex indices representing a simplex.
            
        Returns:
        --------
        numpy.ndarray
            A binary incidence matrix where rows correspond to simplices and columns to vertices.
            Entry (i,j) is 1 if vertex j is in simplex i, and 0 otherwise.
        """
        # Get the number of simplices and find the maximum vertex index
        num_simplices = len(simplecies)
        if num_simplices == 0:
            return np.array([])
            
        max_vertex = max(max(simp) for simp in simplecies if simp)
        
        # Create a dense incidence matrix filled with zeros
        incidence = np.zeros((num_simplices, max_vertex + 1), dtype=int)
        
        # Fill the incidence matrix
        for i, simplex in enumerate(simplecies):
            incidence[i, simplex] = 1

        return incidence
    
    @classmethod
    def from_adjacency_matrix(cls, adj_matrix):
        graph = nx.Graph(adj_matrix)
        cliques = nx.clique.find_cliques(graph)
        incidence = cls.simplecies_to_incidence(list(cliques))
        return cls(incidence)

    def order(self):
        if self._order is None:
            self._order = np.max(self.simplex_orders())
        return self._order
    
    def eccentricity(self, simplex):
        if simplex.ndim == 1:
            simplex = simplex[None, :]
        simplex_vertices_not_shared = np.sum(~self.incidence.astype(bool) & simplex, axis=1)
        return simplex_vertices_not_shared/np.sum(simplex)

    def family_eccentricity(self, simplex):
        """
        Compute the eccentricity of a simplex with respect to the family of simplices.
        
        Parameters:
        -----------
        simplex : numpy.ndarray
        """
        return np.min(self.eccentricity(simplex))

    def _q_connected_components(self, q, return_labels=False):
        return find_connected_components(self.incidence, q, return_labels)

    def q_connected_components(self, q):
        return self._q_connected_components(q)

    def q_connected_components_labeled(self, q):
        return self._q_connected_components(q, return_labels=True)

    def simplex_orders(self):
        return self.incidence.sum(-1) - 1

    def q_incidence(self, q):
        simplecies_of_order_q = self.simplex_orders() == q
        return self.incidence[simplecies_of_order_q]

    def q_upper_incidence(self, q):
        simplecies_of_order_gte_q = self.simplex_orders() >= q
        return self.incidence[simplecies_of_order_gte_q]

    def q_occupation(self, q):
        q_incidence = self.q_incidence(q)
        return q_incidence.sum(0)

    def q_occupation_prob(self, q):
        occupation = self.q_occupation(q)
        return occupation/np.sum(occupation)

    def topological_dimensionality(self, as_dataframe=False, node_names=None):
        top_dim = self.incidence.sum(0)
        if as_dataframe:
            if node_names is None:
                node_names = np.arange(top_dim.shape[0])
            return pd.DataFrame(np.array([node_names, top_dim]).T, columns=['Node', 'Topological Dimensionality'])
        else:
            return top_dim

    def first_structure_vector(self):
        result = []
        # we want to include the q corresponding to the order
        for i in range(0, self.order() + 1):
            result.append(self.q_connected_components(i))
        return np.array(result)

    def second_structure_vector(self):
        simp_orders = self.simplex_orders()
        simps_per_q_count = np.bincount(simp_orders)
        return np.cumsum(simps_per_q_count[::-1])[::-1]

    @classmethod
    def _third_structure_vector(cls, fsv, ssv):
        return 1 - fsv/ssv

    def third_structure_vector(self):
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        return self._third_structure_vector(fsv, ssv)

    def simp_count(self):
        simp_orders = self.simplex_orders()
        simps_per_q_count = np.bincount(simp_orders)
        return simps_per_q_count

    def shared_faces_count(self):
        shared_faces = np.zeros(self.order() + 1)
        num_of_simplices = self.incidence.shape[0]
        for ind_a in range(num_of_simplices):
            for ind_b in range(ind_a + 1, num_of_simplices):
                face_order = (self.incidence[ind_a] & self.incidence[ind_b]).sum() - 1
                if face_order >= 0:
                    shared_faces[face_order] += 1
        return shared_faces

    def topological_entropy(self, q):
        normalization_factor = np.sum(self.q_occupation(q) > 0)
        if normalization_factor == 0:
            return np.nan
        elif normalization_factor == 1:
            return np.nan

        q_occupation_prob = self.q_occupation_prob(q)
        q_occupation_prob = q_occupation_prob[q_occupation_prob > 0]

        entropy = q_occupation_prob * np.log10(q_occupation_prob)
        entropy = np.sum(entropy)/np.log10(normalization_factor)
        return -entropy

    def full_topological_entropy(self):
        result = []
        for i in range(0, self.order() + 1):
            result.append(self.topological_entropy(i))
        return np.array(result)

    def full_char(self):
        top_dims = self.topological_dimensionality()
        top_entropies = self.full_topological_entropy()
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        tsv = self._third_structure_vector(fsv, ssv)
        simp_counts = self.simp_count()
        shared_faces_count = self.shared_faces_count()
        order = np.ones(self.order() + 1)
        return fsv, ssv, tsv, top_entropies, top_dims, simp_counts, shared_faces_count, order

    def q_analysis_vectors(self, as_dataframe=False, with_order_mask=False):

        top_entropies = self.full_topological_entropy()
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        tsv = self._third_structure_vector(fsv, ssv)
        simp_counts = self.simp_count()
        shared_faces_count = self.shared_faces_count()

        result = [fsv, ssv, tsv, top_entropies, simp_counts, shared_faces_count]
        if as_dataframe:
            return self.q_vectors_array_to_dataframe(np.array(result).T)
        elif with_order_mask:
            order = np.ones(self.order() + 1)
            result = np.array(result + [order])
            return result.T
        else:
            return np.array(result).T

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
    
    # @classmethod
    # def q_vectors_dataframe_to_array(cls, q_vectors_dataframe, vector_names=VECTORS):
    #     """
    #     Convert a DataFrame of structure vectors back to a numpy array format.
        
    #     Parameters:
    #     -----------
    #     q_vectors_dataframe : pandas.DataFrame
    #         DataFrame containing structure vectors with columns 'q', 'Vector', and 'Value'
    #     vector_names : list, optional
    #         List of vector names to include (default: VECTORS)
            
    #     Returns:
    #     --------
    #     numpy.ndarray
    #         Array of shape (n_vectors, n_q_values) containing the structure vectors
    #     """
    #     # Filter the dataframe to include only the specified vector names
    #     filtered_df = q_vectors_dataframe[q_vectors_dataframe['Vector'].isin(vector_names)]
        
    #     # Get the number of unique q values
    #     q_values = filtered_df['q'].unique()
    #     n_q_values = len(q_values)
        
    #     # Initialize the result array
    #     result = np.zeros((len(vector_names), n_q_values))
        
    #     # Fill the result array
    #     for i, vector_name in enumerate(vector_names):
    #         vector_data = filtered_df[filtered_df['Vector'] == vector_name]
    #         vector_data = vector_data.sort_values('q')
    #         result[i, :] = vector_data['Value'].values
            
    #     return result