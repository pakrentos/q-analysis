import numpy as np
from scipy.sparse import csr_array
import networkx as nx
from typing import Iterator

def efficient_laplacian(A):
    Q, R = np.linalg.qr(A)
    return np.round(Q @ (R @ R.T) @ Q.T)

def efficient_kirchoff(A):
    _, R = np.linalg.qr(A)
    return np.round(R.T @ R)

def efficient_laplacian_kirchoff(A):
    Q, R = np.linalg.qr(A)
    return np.round(Q @ (R @ R.T) @ Q.T), np.round(R.T @ R)

def simps_to_incidence(list_of_groups):
    row = []
    column = []
    data_counter = 0
    for ind, simp in enumerate(list_of_groups):
        row += [ind]*len(simp)
        data_counter += len(simp)
        column += simp

    row = np.array(row)
    column = np.array(column)

    data = np.ones(data_counter, dtype=int)

    return csr_array((data, (row, column)))

def find_cliques(adj_matrix) -> Iterator[list[int]]:
    graph = nx.Graph(adj_matrix)
    cliques = nx.clique.find_cliques(graph)
    return cliques

def get_cliques_list(adj_matrix) -> list[list[int]]:
    return list(find_cliques(adj_matrix))

def get_incidence(adj_matrix) -> np.ndarray:
    cliques = find_cliques(adj_matrix)
    return simps_to_incidence(cliques).toarray()

def get_simp_complex(adj_matrix):
    incidence = get_incidence(adj_matrix)
    from .simplicial_complex import IncidenceSimplicialComplex
    simplcial_c = IncidenceSimplicialComplex(incidence)
    return simplcial_c

def comp_chars(adj_matrix):
    simplcial_c = get_simp_complex(adj_matrix)
    return simplcial_c.full_char()
