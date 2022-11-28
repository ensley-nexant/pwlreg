import itertools
import numbers

import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution
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


def _assemble_regression_matrix(X, breaks, degree):
    Acols = []
    bins = np.digitize(X, breaks).clip(None, len(breaks) - 1)
    for i, d in enumerate(degree):
        for k in range(d + 1):
            Acols.append(np.where(bins == i + 1, X ** k, 0.0))

    A = np.column_stack(Acols)
    return A


def _assemble_continuity_constraints(breaks, degree):
    m = len(degree) + sum(degree)
    Crows = []
    i = 0
    for b, (d0, d1) in zip(breaks[1:-1], itertools.pairwise(degree)):
        row = np.zeros(m)
        row[i : (i + d0 + 1)] = [b ** k for k in range(d0 + 1)]
        row[(i + d0 + 1) : (i + d0 + d1 + 2)] = [-1.0 * b ** k for k in range(d1 + 1)]
        Crows.append(row)
        i += d0 + 1

    C = np.row_stack(Crows)
    return C


def _augment_breaks(breaks, break_0, break_n):
    v = np.sort(breaks)
    b_ = np.zeros(len(v) + 2)
    b_[0], b_[1:-1], b_[-1] = break_0, v.copy(), break_n
    return b_


def _fit_opt(breaks, X, y, break_0, break_n, degree, continuity="c0", weights=None):
    b_ = _augment_breaks(breaks, break_0, break_n)
    try:
        _, ssr = _lstsq_constrained(X, y, b_, degree, continuity, weights)
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


def _lstsq_constrained(X, y, breaks, degree=1, continuity="c0", weights=None):
    breaks = np.sort(breaks)
    if isinstance(degree, numbers.Number):
        degree = [degree] * (len(breaks) - 1)

    A = _assemble_regression_matrix(X, breaks, degree)
    n, p = A.shape

    if weights is None:
        weights = np.ones(shape=n)

    Aw = A * weights.reshape(-1, 1)
    yw = y * weights

    if continuity is not None and len(breaks) > 2:
        C = _assemble_continuity_constraints(breaks, degree)
        r, _ = C.shape

        Xln, _, _, _ = _safe_lstsq(C, np.zeros(r))
        V = linalg.null_space(C)
        beta, _, _, _ = _safe_lstsq(Aw @ V, yw - Aw @ Xln)
        coef = V @ beta

        e = np.dot(A, coef) - y
        ssr = np.dot(e, e)
    else:
        coef, ssr, _, _ = _safe_lstsq(Aw, yw)

    return coef, ssr


def _auto_piecewise_regression(
    X,
    y,
    n_segments,
    degree,
    continuity="c0",
    weights=None,
    solver="auto",
    max_iter=None,
    tol=1e-4,
    random_state=False,
    return_n_iter=False,
):
    if solver == "auto":
        solver = "diffevo"

    _valid_solvers = ("diffevo", "l-bfgs-b")
    if solver not in _valid_solvers:
        raise ValueError("Valid solvers are: {}. Got {}".format(_valid_solvers, solver))

    if solver == "diffevo":
        breakpoints, n_iter = _solve_diffevo(
            X,
            y,
            n_segments,
            degree,
            continuity,
            weights,
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
            weights,
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
    weights=None,
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
        (X, y, break_0, break_n, degree, continuity, weights),
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
    weights,
    solver,
):
    return 0, 0


class _BasePiecewiseRegressor(RegressorMixin):
    def __init__(self, degree, continuity):
        self.degree = degree
        self.continuity = continuity

    def _assemble_regression_matrix(self, X, breaks):
        return _assemble_regression_matrix(X, breaks, self.degree)

    # noinspection PyTypeChecker
    def _decision_function(self, X, breaks):
        X = check_array(X, accept_sparse=True, ensure_2d=False)
        check_is_fitted(self)
        A = self._assemble_regression_matrix(X, breaks)
        return np.dot(A, self.coef_)

    def fit_with_breaks(self, X, y, breaks, weights=None):
        breaks = breaks if isinstance(breaks, np.ndarray) else np.array(breaks)

        if isinstance(self.degree, numbers.Number):
            self.degree = [self.degree] * (len(breaks) - 1)

        if len(self.degree) != len(breaks) - 1:
            msg = (
                "With {} breakpoints, the model will fit {} segment(s). "
                "However, {} degree values were supplied. "
                "The number of degree values must match the number of segments."
            )
            raise ValueError(msg.format(len(breaks), len(breaks) - 1, len(self.degree)))

        _valid_continuities = ("c0",)
        if self.continuity and self.continuity not in _valid_continuities:
            raise ValueError(
                "Continuity must be one of: {}. Got {}".format(
                    _valid_continuities, self.continuity
                )
            )

        self.coef_, self.ssr_ = _lstsq_constrained(
            X, y, breaks, self.degree, self.continuity, weights
        )
        self.n_params_ = len(self.coef_)
        self.is_fitted_ = True
        return self


class PiecewiseLinearRegression(BaseEstimator, _BasePiecewiseRegressor):
    def __init__(self, *, breakpoints=None, degree=1, continuity="c0"):
        self.breakpoints = breakpoints
        super().__init__(degree, continuity)

    def fit(self, X, y, *, weights=None):
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)
        weights = _check_sample_weight(weights, X, dtype=X.dtype)

        if self.breakpoints is None:
            self.breakpoints = np.array([np.min(X), np.max(X)])

        self.fit_with_breaks(X, y, self.breakpoints, weights)
        return self

    def predict(self, X):
        return self._decision_function(X, self.breakpoints)


class AutoPiecewiseRegression(BaseEstimator, _BasePiecewiseRegressor):
    def __init__(
        self, n_segments, *, degree=1, continuity="c0", solver="auto", random_state=None
    ):
        self.n_segments = n_segments
        self.solver = solver
        self.random_state = random_state
        super().__init__(degree, continuity)

    def fit(self, X, y, weights=None):
        random_state_ = check_random_state(self.random_state)
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)
        weights = _check_sample_weight(weights, X, dtype=X.dtype)

        if self.n_segments == 1:
            self.breakpoints_ = np.array([np.min(X), np.max(X)])
            return self.fit_with_breaks(X, y, self.breakpoints_, weights)

        self.breakpoints_, self.n_iter = _auto_piecewise_regression(
            X,
            y,
            self.n_segments,
            self.degree,
            self.continuity,
            weights,
            solver=self.solver,
            random_state=random_state_,
            return_n_iter=True,
        )

        return self.fit_with_breaks(X, y, self.breakpoints_)

    def predict(self, X):
        return self._decision_function(X, self.breakpoints_)
