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
from itertools import combinations
from .simplicial_complex import SimplicialComplex
from scipy import sparse
from typing import Iterable

class GraphCliqueFilter(BaseEstimator, TransformerMixin):
    def __init__(self, *, q=0, threshold=None):
        self.q = q
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        simps = []
        for x in X:
            adj = x
            if self.threshold:
                adj = self.threshold(x)
            simps.append(SimplicialComplex.from_adjacency_matrix(adj))
        
        projector = SimplexProjection(q=self.q)
        adj_masks = projector.transform(simps)
        
        result = X.copy()
        result[~adj_masks.astype(bool)] = 0
        
        return result

class SimplexProjection(BaseEstimator, TransformerMixin):
    def __init__(self, q=0, weighted=False):
        self.q = q
        self.weighted = weighted

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        num_nodes = max(max(i) for simp in X for i in simp.simplices) + 1
        for simp in X:
            filtered_simplices = [s for s in simp.simplices if len(s) >= self.q + 1]

            adj = sparse.lil_matrix((num_nodes, num_nodes), dtype=int)
            for simplex in filtered_simplices:
                for i, j in combinations(simplex, 2):
                    if self.weighted:
                        adj[i, j] = adj[j, i] = 1 + adj[i, j]
                    else:
                        adj[i, j] = adj[j, i] = 1
            result.append(adj.tocsr())
        return np.array(result)

class GradedParametersTransformer(BaseEstimator, TransformerMixin):
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
        return self

    def transform(self, X: Iterable[np.ndarray]):
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
        graded_parameters = [
            SimplicialComplex.from_adjacency_matrix(adj)
                .graded_parameters()
            for adj in X
        ]
        
        if self.max_order is None:
            max_order = max(param_set.get_max_order() for param_set in graded_parameters)
        else:
            max_order = self.max_order
    
        graded_parameters = np.array([
            param_set.to_numpy(max_order=max_order, fill_value=self.fill_value)
            for param_set in graded_parameters
        ])
            
        if self.flatten:
            return graded_parameters.reshape(graded_parameters.shape[0], -1)
        
        return graded_parameters

class QConnectedComponents(BaseEstimator, TransformerMixin):
    def __init__(self, q_level):
        self.q_level = q_level
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_labels = []
        for incidence_matrix in X:
            complex = SimplicialComplex(incidence_matrix)
            _, labels = complex.q_connected_components_labeled(self.q_level)
            X_labels.append(labels)
        return np.array(X_labels)
