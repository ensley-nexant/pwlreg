import itertools
import numbers
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


def _check_sample_weight(weights, X, dtype):
    n_samples = len(X)
    if dtype is not None and dtype not in (np.float32, np.float64):
        dtype = np.float64
    if weights is None:
        weights = np.ones(n_samples)
    elif isinstance(weights, numbers.Number):
        # noinspection PyTypeChecker
        weights = np.full(n_samples, weights, dtype)

    if weights.ndim != 1:
        raise ValueError("Sample weights must be 1D array or scalar")
    if weights.shape != (n_samples,):
        raise ValueError(
            "weights.shape == {}, expected {}".format(weights.shape, (n_samples,))
        )

    return weights


def _rescale_data(X, y, sample_weight):
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = sw_matrix @ X
    y = sw_matrix @ y
    return X, y, sample_weight_sqrt


# def _assemble_regression_matrix(X, breaks):
#     breaks = np.sort(breaks)
#     A = np.column_stack([np.clip(X - b, 0., None) for b in breaks[:-1]])
#     A = np.insert(A, 0, 1, axis=1)
#     return A


def _assemble_regression_matrix(X, breaks, degree=1):
    breaks = np.sort(breaks)
    if isinstance(degree, numbers.Number):
        degree = [degree] * (len(breaks) - 1)

    Acols = [np.ones_like(X)]
    for b, d in zip(breaks[:-1], degree):
        if d == 0:
            idx = np.argwhere(X > b)
            for j in range(1, len(Acols)):
                Acols[j][idx] = 0.
            Acols.append(np.where(X > b, 1., 0.))
        else:
            for k in range(1, d + 1):
                Acols.append(np.clip((X - b) ** k, 0., None))

    A = np.column_stack(Acols)
    return A


def _assemble_regression_equation(X, breaks, degree=1, continuity="c0"):
    breaks = np.sort(breaks)
    if isinstance(degree, numbers.Number):
        degree = [degree] * (len(breaks) - 1)

    Acols = []
    bins = np.digitize(X, breaks).clip(None, len(breaks) - 1)
    for i, d in enumerate(degree):
        for k in range(d + 1):
            Acols.append(np.where(bins == i+1, X ** k, 0.))

    A = np.column_stack(Acols)
    _, m = A.shape

    if continuity is None:
        return A, A, m, 0

    Crows = []
    i = 0
    for b, (d0, d1) in zip(breaks[1:-1], itertools.pairwise(degree)):
        row = np.zeros(m)
        row[i:(i+d0+1)] = [b ** k for k in range(d0 + 1)]
        row[(i+d0+1):(i+d0+d1+2)] = [-1. * b ** k for k in range(d1 + 1)]
        Crows.append(row)
        i += d0 + 1

    C = np.row_stack(Crows)
    o, _ = C.shape

    K = np.zeros((m + o, m + o))
    K[:m, :m] = np.dot(A.T, A)
    K[:m, m:] = C.T
    K[m:, :m] = C

    return K, A, m, o


def _augment_breaks(breaks, break_0, break_n):
    v = np.sort(breaks)
    b_ = np.zeros(len(v) + 2)
    b_[0], b_[1:-1], b_[-1] = break_0, v.copy(), break_n
    return b_


def _fit_opt(breaks, X, y, break_0, break_n, degree, continuity="c0"):
    b_ = _augment_breaks(breaks, break_0, break_n)
    K, A, m, o = _assemble_regression_equation(X, b_, degree, continuity)
    try:
        Z = np.zeros(m + o)
        Z[:m] = np.dot(A.T, y)
        beta = linalg.solve(K, Z)
        e = np.dot(A, beta[:m]) - y
        ssr = np.dot(e, e)
    except linalg.LinAlgError as e:
        print(e)
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


def _auto_piecewise_regression(
        X,
        y,
        n_segments,
        degree,
        continuity="c0",
        sample_weight=None,
        solver="auto",
        max_iter=None,
        tol=1e-4,
        verbose=0,
        random_state=False,
        return_n_iter=False,
):
    if solver == "auto":
        solver = "diffevo"

    _valid_solvers = ("diffevo", "l-bfgs-b")
    if solver not in _valid_solvers:
        raise ValueError(
            "Valid solvers are: {}. Got {}".format(_valid_solvers, solver)
        )

    if solver == "diffevo":
        breakpoints, n_iter = _solve_diffevo(
            X,
            y,
            n_segments,
            degree,
            continuity,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
    else:
        breakpoints, n_iter = _solve_minimize(
            X,
            y,
            n_segments,
            degree,
            continuity,
            solver,
        )

    if return_n_iter:
        return breakpoints, n_iter
    else:
        return breakpoints


def _solve_diffevo(
        X,
        y,
        n_segments,
        degree,
        continuity="c0",
        max_iter=None,
        tol=1e-4,
        random_state=None,
):
    break_0, break_n = np.min(X), np.max(X)

    bounds = np.zeros((n_segments - 1, 2))
    bounds[:, 0] = break_0
    bounds[:, 1] = break_n

    res = differential_evolution(
        _fit_opt,
        bounds,
        (X, y, break_0, break_n, degree, continuity),
        popsize=15,
        init="sobol",
        maxiter=max_iter,
        tol=tol,
        atol=1e-5,
        seed=random_state,
    )

    breaks = _augment_breaks(res.x, break_0, break_n)
    return breaks, res.nit


def _solve_minimize(
        X,
        y,
        n_segments,
        degree,
        continuity,
        solver,
):
    return 0, 0


class PiecewiseRegressorMixin(RegressorMixin):
    def _decision_function(self, X):
        X = check_array(X, accept_sparse=True, ensure_2d=False)
        check_is_fitted(self, "breakpoints_")

        # noinspection PyTypeChecker
        _, A, _, _ = _assemble_regression_equation(
            X,
            self.breakpoints_,
            self.degree,
            self.continuity,
        )

        return np.dot(A, self.coef_)

    def fit_with_breaks(self, X, y, breaks, degree, continuity):
        breaks = breaks if isinstance(breaks, np.ndarray) else np.array(breaks)
        K, A, m, o = _assemble_regression_equation(X, breaks, degree, continuity)
        Z = np.zeros(m + o)
        Z[:m] = np.dot(A.T, y)
        beta = linalg.solve(K, Z)
        self.coef_ = beta[:m]
        e = np.dot(A, self.coef_) - y
        self.ssr_ = np.dot(e, e)
        self.n_params_ = m
        return self


class PiecewiseLinearRegression(BaseEstimator, PiecewiseRegressorMixin):
    def __init__(self, *, breakpoints=None, degree=1, continuity="c0"):
        self.breakpoints = breakpoints
        self.degree = degree
        self.continuity = continuity

    def fit(self, X, y, *, weights=None):
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)
        weights = _check_sample_weight(weights, X, dtype=X.dtype)
        X, y, sample_weight_sqrt = _rescale_data(X, y, weights)

        break_0, break_n = np.min(X), np.max(X)

        if self.breakpoints is None:
            self.breakpoints_ = np.array([break_0, break_n])
        else:
            self.breakpoints_ = self.breakpoints

        self.fit_with_breaks(X, y, self.breakpoints_, self.degree, self.continuity)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._decision_function(X)


class AutoPiecewiseRegression(BaseEstimator, PiecewiseRegressorMixin):
    def __init__(self, n_segments, *, degree=1, continuity="c0", solver="auto", random_state=None):
        self.n_segments = n_segments
        self.degree = degree
        self.continuity = continuity
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y, weights=None):
        random_state_ = check_random_state(self.random_state)
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)
        weights = _check_sample_weight(weights, X, dtype=X.dtype)
        X, y, sample_weight_sqrt = _rescale_data(X, y, weights)

        if self.n_segments == 1:
            self.fit_with_breaks(X, y, np.array([np.min(X), np.max(X)]), self.degree, self.continuity)
            return self

        self.breakpoints_, self.n_iter = _auto_piecewise_regression(
            X,
            y,
            self.n_segments,
            self.degree,
            self.continuity,
            sample_weight=weights,
            solver=self.solver,
            random_state=random_state_,
            return_n_iter=True,
        )

        self.fit_with_breaks(X, y, self.breakpoints_, self.degree, self.continuity)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._decision_function(X)
