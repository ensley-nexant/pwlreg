import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


def _assemble_regression_matrix(X, breaks):
    breaks = np.sort(breaks)
    A = np.column_stack([np.clip(X - b, 0., None) for b in breaks[:-1]])
    A = np.insert(A, 0, 1, axis=1)
    return A


def _augment_breaks(breaks, break_0, break_n):
    v = np.sort(breaks)
    b_ = np.zeros(len(v) + 2)
    b_[0], b_[1:-1], b_[-1] = break_0, v.copy(), break_n
    return b_


def _fit_opt(breaks, X, y, break_0, break_n):
    b_ = _augment_breaks(breaks, break_0, break_n)
    A = _assemble_regression_matrix(X, b_)
    try:
        _, ssr, _, _ = _safe_lstsq(A, y)
    except linalg.LinAlgError:
        ssr = np.inf

    return ssr


def _safe_lstsq(A, y):
    coef_, ssr, rank_, singular_ = linalg.lstsq(A, y)
    if isinstance(ssr, list):
        ssr = ssr[0]
    elif isinstance(ssr, np.ndarray):
        if ssr.size == 0:
            e = np.dot(A, coef_) - y
            ssr = np.dot(e, e)
        else:
            ssr = ssr[0]

    coef_ = coef_.T
    return coef_, ssr, rank_, singular_


class PiecewiseLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, *, n_segments=None, breakpoints=None, random_state=None):
        self.n_segments = n_segments
        self.breakpoints = breakpoints
        self.random_state = random_state

    def _decision_function(self, X):
        X = check_array(X, accept_sparse=True, ensure_2d=False)
        check_is_fitted(self, "is_fitted_")

        A = _assemble_regression_matrix(X, self.breakpoints_)
        return np.dot(A, self.coef_)

    @staticmethod
    def _to_numpy(input_):
        return input_ if isinstance(input_, np.ndarray) else np.array(input_)

    def fit_with_breaks(self, X, y, breaks):
        breaks = self._to_numpy(breaks)
        A = _assemble_regression_matrix(X, breaks)
        self.coef_, self.ssr_, self.rank_, self.singular_ = _safe_lstsq(A, y)
        return self

    def fit(self, X, y, *, weights=None):
        random_state_ = check_random_state(self.random_state)
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)

        break_0, break_n = np.min(X), np.max(X)

        if self.breakpoints is not None:
            if len(self.breakpoints) < 2:
                warnings.warn("Not enough breakpoints. Setting to min and max of input data")
                self.breakpoints_ = np.array([break_0, break_n])
            else:
                self.breakpoints_ = self.breakpoints
            if self.n_segments is not None and self.n_segments != len(self.breakpoints_) - 2:
                warnings.warn("n_segments and breakpoints are incompatible. Ignoring n_segments")
                self.n_segments_ = len(self.breakpoints_) - 2

            self.fit_with_breaks(X, y, self.breakpoints_)

        elif self.n_segments is not None:
            self.n_segments_ = self.n_segments
            if self.n_segments_ == 1:
                self.fit_with_breaks(X, y, [break_0, break_n])
            else:
                bounds = np.zeros((self.n_segments_ - 1, 2))
                bounds[:, 0] = break_0
                bounds[:, 1] = break_n

                res = differential_evolution(_fit_opt, bounds, (X, y, break_0, break_n), popsize=50, tol=1e-3, atol=1e-4, seed=random_state_)
                self.breakpoints_ = _augment_breaks(res.x, break_0, break_n)
                self.fit_with_breaks(X, y, self.breakpoints_)
                if "nit" in res.keys():
                    self.n_iter_ = res.nit

        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._decision_function(X)
