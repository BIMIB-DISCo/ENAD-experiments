from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class GroupedScaler(BaseEstimator, TransformerMixin):
    """Center the features with the class means.

    This estimator centers the features with the class means. The classes
    should be passed as the last column of X, as a workaround for having
    two distinct labelling (the classification labels and the adversarial
    labels).

    Parameters
    ----------
    with_centering : bool, default=True
        Set to False to just remove the labelling column.

    Attributes
    ----------
    group_means : list
        Means of each class
    """


    def __init__(self, with_centering=True):
        self.with_centering = with_centering

    def fit(self, X, y=None):
        """Compute the groups means and centers the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features+1)
            The last column must contain the classes as a workaround.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        groups = X[:, -1].astype(int)
        X = X[:, :-1]

        if self.with_centering:

            self.group_means = []
            for lab in np.unique(groups):

                mask_idxs = (groups == lab).nonzero()[0]

                X_mean = np.mean(X[mask_idxs], axis=0)

                self.group_means.append(X_mean)

        return self

    def transform(self, X, y=None):

        groups = X[:, -1].astype(int)
        X = X[:, :-1]

        if self.with_centering:
            old_idxs = []
            X_norm = []
            for lab in np.unique(groups):

                mask_idxs = (groups == lab).nonzero()[0]

                temp_X = X[mask_idxs] - self.group_means[lab]

                X_norm.extend(temp_X)
                old_idxs.extend(mask_idxs)

            X = np.array(X_norm)[np.argsort(old_idxs)]

        return X

    def get_params(self, deep=True):
        return {'with_centering': self.with_centering}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self