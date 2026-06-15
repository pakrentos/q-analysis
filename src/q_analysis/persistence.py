"""Persistent q-communities of a graph filtration (elder-rule intervals)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from q_analysis.q_analysis import (
    py_persistent_q_communities_edges,
    py_persistent_q_communities_matrix,
)


@dataclass(frozen=True)
class PersistentCommunities:
    """Persistence intervals of q-connected components, members packed CSR-style."""

    q: np.ndarray
    birth: np.ndarray
    death: np.ndarray
    offsets: np.ndarray
    members_flat: np.ndarray

    def __len__(self):
        return len(self.q)

    def members(self, i):
        """Vertex ids of interval i."""
        return self.members_flat[self.offsets[i]:self.offsets[i + 1]]

    def __getitem__(self, i):
        return self.q[i], self.birth[i], self.death[i], self.members(i)

    @property
    def size(self):
        return np.diff(self.offsets)

    def to_dataframe(self):
        return pd.DataFrame({
            'q': self.q,
            'birth': self.birth,
            'death': self.death,
            'size': self.size,
            'members': [self.members(i) for i in range(len(self))],
        })


def persistent_q_communities(distance_matrix, max_q=None):
    """
    Computes persistent q-communities over the ascending edge-weight filtration
    of a square distance matrix (upper triangle; non-finite entries are skipped).
    """
    matrix = np.ascontiguousarray(distance_matrix, dtype=np.float64)
    return PersistentCommunities(*py_persistent_q_communities_matrix(matrix, max_q))


def persistent_q_communities_from_edges(edges, max_q=None):
    """Same as persistent_q_communities, for an (E, 3) array of [u, v, weight] rows."""
    edge_array = np.ascontiguousarray(edges, dtype=np.float64)
    if edge_array.ndim != 2:
        raise ValueError("edges must be a 2D array with rows [u, v, weight]")
    return PersistentCommunities(*py_persistent_q_communities_edges(edge_array, max_q))
