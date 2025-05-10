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
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from .utils import get_incidence, construct_structure_vectors_datasets
from .simplicial_complex import IncidenceSimplicialComplex

class SimplicialFilter(BaseEstimator, TransformerMixin):
    """
    Transformer that filters a graph based on the dimension of simplices in its simplicial complex.
    
    This transformer takes an adjacency matrix representing a graph, finds all maximal cliques
    (which are interpreted as simplices), builds a simplicial complex, and then filters out
    all simplices with dimension less than q. Finally, it reconstructs a filtered adjacency matrix
    that preserves only the edges that are part of the remaining higher-dimensional simplices.
    
    Parameters
    ----------
    q : int, default=0
        Minimum dimension of simplices to keep. Simplices with dimension < q will be filtered out.
        
    threshold : callable or None, default=None
        Function to convert weighted adjacency matrices to binary adjacency matrices.
        If None and the input is not binary, a ValueError will be raised.
    
    Attributes
    ----------
    q : int
        The minimum dimension of simplices to keep.
        
    threshold : callable or None
        The function used to threshold weighted adjacency matrices.
    """
    def __init__(self, *, q=0, threshold=None):
        self.q = q
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_nodes, n_nodes)
            Input adjacency matrices.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform adjacency matrices by filtering based on simplex dimension.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_nodes, n_nodes)
            Input adjacency matrices.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_nodes, n_nodes)
            Filtered adjacency matrices where only edges that are part of
            simplices with dimension >= q are preserved.
            
        Raises
        ------
        ValueError
            If input is not binary and no threshold function is provided.
        """
        int_flag = (X.dtype == int) and (np.unique(X) == [0, 1]).all()
        bool_flag = X.dtype == bool
        process_fn = lambda x: x
        if not bool_flag and not int_flag:
            if self.threshold is None:
                raise ValueError("Threshold for edges is not specified")
            else:
                process_fn = self.threshold

        result = []
        for x in X:
            inc = IncidenceSimplicialComplex.from_adjacency_matrix(process_fn(x)).incidence_matrix
            simp = IncidenceSimplicialComplex(inc)
            filtered_inc = simp.q_upper_incidence(self.q)
            adj_restored = self.get_adj_mask(filtered_inc)
            temp = x.copy()
            temp[~adj_restored] = 0
            result.append(temp)
        return np.array(result)

    @classmethod
    def get_adj_mask(cls, inc):
        result = []
        for column in inc.T:
            temp = inc[column > 0]
            if temp.shape[0] == 0:
                result.append(np.zeros(temp.shape[-1], dtype=bool))
            else:
                result.append(temp.any(0))
        return np.array(result)

class StructureVectorsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts q-analysis structure vectors from graphs represented by adjacency matrices.
    
    This transformer takes adjacency matrices as input and computes various q-analysis structure vectors
    including first structure vector, second structure vector, third structure vector,
    topological entropy, simplex counts, and shared faces counts.
    
    Parameters
    ----------
    flatten : bool, default=False
        If True, flattens the output array to shape (n_samples, n_features).
        If False, returns a 3D array of shape (n_samples, n_orders, n_metrics).
    
    Attributes
    ----------
    flatten : bool
        Whether to flatten the output array.
    """
    def __init__(self, flatten=False, max_order=None, fill_value=np.nan):
        self.flatten = flatten
        self.max_order = max_order
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_nodes, n_nodes)
            Input adjacency matrices.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform adjacency matrices into q-analysis metrics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_nodes, n_nodes)
            Input adjacency matrices.
            
        Returns
        -------
        q_metrics : ndarray
            If flatten=True: Array of shape (n_samples, n_features) containing flattened q-metrics.
            If flatten=False: Array of shape (n_samples, n_orders, n_metrics) containing structured q-metrics.
        """
        structure_vectors = construct_structure_vectors_datasets(X, self.max_order, self.fill_value)
        
        if self.flatten:
            # Flatten the array to shape (n_samples, n_features)
            # First remove any NaN values by replacing them with 0
            structure_vectors = np.nan_to_num(structure_vectors, nan=0.0)
            return structure_vectors.reshape(structure_vectors.shape[0], -1)
        
        return structure_vectors

class QConnectedComponents(BaseEstimator, TransformerMixin):
    def __init__(self, q_level):
        self.q_level = q_level
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_labels = []
        for incidence_matrix in X:
            complex = IncidenceSimplicialComplex(incidence_matrix)
            _, labels = complex.q_connected_components_labeled(self.q_level)
            X_labels.append(labels)
        return np.array(X_labels)
