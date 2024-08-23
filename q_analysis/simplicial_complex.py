import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import diags
from .utils import efficient_laplacian, efficient_kirchoff, efficient_laplacian_kirchoff

class IncidenceSimplicialComplex:
    def __init__(self, incidence):
        self.incidence = incidence
        self._laplacian = None
        self._order = None
        self._kirchoff = None

    def laplacian(self):
        if self._laplacian is None:
            self._laplacian = efficient_laplacian(self.incidence)
        return self._laplacian

    def kirchoff(self):
        if self._kirchoff is None:
            self._kirchoff = efficient_kirchoff(self.incidence)
        return self._kirchoff

    def order(self):
        if self._order is None:
            self._order = np.max(self.laplacian().diagonal()).astype(int) - 1
        return self._order

    def _q_connected_components(self, q, return_labels=False):
        adj_full = (self.laplacian() >= q + 1).astype(np.int8)
        adj_diag = adj_full.diagonal()
        adj = adj_full - diags(adj_diag)
        if return_labels:
            num_of_components, labels = connected_components(
                adj,
                directed=False,
                return_labels=return_labels
            )
            num_of_components -= np.sum(adj_diag == 0)
            labels[adj_diag == 0] = -1
            return num_of_components, labels
        else:
            num_of_components = connected_components(
                adj,
                directed=False,
                return_labels=return_labels
            ) - np.sum(adj_diag == 0)
            return num_of_components

    def q_connected_components(self, q):
        return self._q_connected_components(q)

    def q_connected_components_labeled(self, q):
        return self._q_connected_components(q, return_labels=True)

    def q_incidence(self, q):
        simplecies_of_order_q = self.laplacian().diagonal() == q + 1
        return self.incidence[simplecies_of_order_q]

    def q_upper_incidence(self, q):
        simplecies_of_order_q = self.incidence.sum(-1) >= q + 1
        return self.incidence[simplecies_of_order_q]

    def q_occupation(self, q):
        q_incidence = self.q_incidence(q)
        return q_incidence.sum(0)

    def q_occupation_prob(self, q):
        occupation = self.q_occupation(q)
        return occupation/np.sum(occupation)

    def topological_dim(self):
        return self.kirchoff().diagonal()

    def full_char(self):
        self._laplacian, self._kirchoff = efficient_laplacian_kirchoff(self.incidence)
        top_dims = self.topological_dim()
        top_entropies = self.full_topological_entropy()
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        tsv = self._third_structure_vector(fsv, ssv)
        simp_counts = self.simp_count()
        shared_faces_count = self.shared_faces_count()
        order = np.ones(self.order() + 1)
        self._laplacian = None
        self._kirchoff = None
        return fsv, ssv, tsv, top_entropies, top_dims, simp_counts, shared_faces_count, order

    def q_char(self):
        self._laplacian, self._kirchoff = efficient_laplacian_kirchoff(self.incidence)
        top_entropies = self.full_topological_entropy()
        fsv = self.first_structure_vector()
        ssv = self.second_structure_vector()
        tsv = self._third_structure_vector(fsv, ssv)
        simp_counts = self.simp_count()
        shared_faces_count = self.shared_faces_count()
        order = np.ones(self.order() + 1)
        self._laplacian = None
        self._kirchoff = None
        result = np.array([fsv, ssv, tsv, top_entropies, simp_counts, shared_faces_count, order])
        return result.T
