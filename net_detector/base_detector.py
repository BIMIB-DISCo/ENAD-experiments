from sklearn.base import BaseEstimator
from sklearn.covariance import EmpiricalCovariance

from .preprocessing import GroupedScaler

import numpy as np

class Mahalanobis(BaseEstimator):

    def __init__(self, with_centering=True, centering='closest'):
        self.with_centering = with_centering
        self.centering = centering

    def fit(self, X, y=None):

        self.scaler = GroupedScaler(self.with_centering)

        X = self.scaler.fit_transform(X)

        self.detector_ = EmpiricalCovariance(store_precision=True, assume_centered=False)
        self.detector_.fit(X)

        return self

    def decision_function(self, X, y=None):

        if self.centering == 'closest':

            X = X[:,:-1]

            scores = []
            for mean in self.scaler.group_means:
                score = -0.5*np.diag(np.matmul(np.matmul((X-mean), self.detector_.precision_), (X-mean).T))
                scores.append(score)
            scores = np.array(scores)
            scores = np.max(scores, axis=0)

        elif self.centering == 'group':

            X = self.scaler.transform(X)

            scores = -0.5*np.diag(np.matmul(np.matmul(X, self.detector_.precision_), X.T))

        return scores

    def get_params(self, deep=True):
        return {
            'with_centering': self.with_centering
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self