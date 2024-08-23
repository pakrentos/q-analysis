import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from .utils import get_incidence
from .simplicial_complex import IncidenceSimplicialComplex

class QReducer(BaseEstimator, TransformerMixin):
    def __init__(self, *, q=0, threshold=None):
        self.q = q
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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
            inc = get_incidence(process_fn(x))
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

class QTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q_level):
        self.q_level = q_level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = []
        for adj_matrix in X:
            complex = IncidenceSimplicialComplex(get_incidence(adj_matrix))
            q_vector = complex.first_structure_vector()[self.q_level]
            top_dim = complex.topological_dim()
            transformed.append(np.concatenate([q_vector, top_dim]))
        return np.array(transformed)

class QCluster(BaseEstimator, ClusterMixin):
    def __init__(self, q_level):
        self.q_level = q_level

    def fit(self, X, y=None):
        self.labels_ = []
        for adj_matrix in X:
            complex = IncidenceSimplicialComplex(get_incidence(adj_matrix))
            _, labels = complex.q_connected_components_labeled(self.q_level)
            self.labels_.append(labels)
        self.labels_ = np.array(self.labels_)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
