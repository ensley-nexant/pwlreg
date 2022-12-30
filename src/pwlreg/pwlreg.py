"""PWLReg: A flexible implementation of piecewise least squares regression."""
import itertools
import numbers

import numpy as np
from scipy import linalg, optimize
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted


def _check_sample_weight(weights: np.ndarray, X, dtype):
    """Validate sample weights.

    Adapted from scikit-learn's implementation. Passing weights=None
    will return an array of all ones.

    Args:
        weights: Input sample weights
        X: Input data
        dtype: dtype of the validated `weights`

    Returns:
        Validated sample weights
    """
    n_samples = len(X)
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
    """Construct the regression matrix.

    Args:
        X: Input data
        breaks: Array of breakpoints. The first breakpoint must be the
            minimum value in X and the last breakpoint must be the maximum
            value in X.
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.

    Returns:
        The regression matrix

        [ X**0 * I(X < b1), ..., X**d1 * I(X < b1), X**0 * I(b1 <= X < b2), ..., X**d2 * I(b1 <= X < b2), ...]
        [ ...                                                                                             ...]

        for degrees d1, ..., di and breakpoints b1, ..., bj
    """  # noqa: E501,B950
    Acols = []
    bins = np.digitize(X, breaks).clip(None, len(breaks) - 1)
    for i, d in enumerate(degree):
        for k in range(d + 1):
            Acols.append(np.where(bins == i + 1, X**k, 0.0))

    A = np.column_stack(Acols)
    return A


def _assemble_continuity_constraints(breaks, degree):
    """Construct the continuity constraint matrix.

    This matrix enforces continuity at the breakpoints between the components
    of the piecewise function.

    Args:
        breaks: Array of breakpoints. The first breakpoint must be the
            minimum value in X and the last breakpoint must be the maximum
            value in X.
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.

    Returns:
        The continuity constraint matrix
    """
    m = len(degree) + sum(degree)  # there will be m columns
    Crows = []
    i = 0
    for b, (d0, d1) in zip(breaks[1:-1], itertools.pairwise(degree), strict=True):
        row = np.zeros(m)
        row[i : (i + d0 + 1)] = [b**k for k in range(d0 + 1)]
        row[(i + d0 + 1) : (i + d0 + d1 + 2)] = [-1.0 * b**k for k in range(d1 + 1)]
        Crows.append(row)
        i += d0 + 1

    C = np.row_stack(Crows)
    return C


def _augment_breaks(breaks, break_0, break_n):
    """Add starting and ending breakpoint to the list of all breakpoints.

    This is necessary for the optimization routine, where `breaks` is allowed
    to vary but the first and last breakpoints must be `break_0` and `break_n`,
    respectively, and cannot vary.

    Args:
        breaks: Breakpoints that do not include the starting and ending
            breakpoints
        break_0: Starting breakpoint (min of input data)
        break_n: Ending breakpoint (max of input data)

    Returns:
        An array [break_0, breaks, break_n]
    """
    v = np.sort(breaks)
    b_ = np.zeros(len(v) + 2)
    b_[0], b_[1:-1], b_[-1] = break_0, v.copy(), break_n
    return b_


def _fit_opt(breaks, X, y, break_0, break_n, degree, continuity="c0", weights=None):
    """Least squares optimization function.

    A wrapper around the least squares routine that returns the sum of squared
    errors and catches any linear algebra exceptions. This is the objective
    function that gets minimized when searching for optimal breakpoints.

    Args:
        breaks: The variable breakpoints, not including the start and end
        X: Input data, independent variable
        y: Input data, dependent variable
        break_0: Starting breakpoint (min of X)
        break_n: Ending breakpoint (max of X)
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.
        continuity: The level of continuity that will be enforced. "c0" is
            continuous, None is no continuity
        weights: Sample weights

    Returns:
        Sum of squared errors of the least square fit
    """
    b_ = _augment_breaks(breaks, break_0, break_n)
    try:
        _, ssr = _lstsq_constrained(X, y, b_, degree, continuity, weights)
    except linalg.LinAlgError as e:  # pragma: no cover
        print(e)
        ssr = np.inf

    return ssr


def _safe_lstsq(A, y):
    """Safe wrapper around lstsq solver.

    Wraps Scipy's linalg.lstsq() routine in a way that ensures the sum of
    squared errors always comes back as a scalar value. In Scipy, it can be a
    scalar, an array, or empty depending on the inputs.

    Args:
        A: Left-hand side array
        y: Right-hand side array

    Returns:
        x: Least-squares solution
        residues: Square of the 2-norm for `y - A x`
        rank: Effective rank of A
        s: Singular values of A
    """
    coef_, ssr, rank_, singular_ = linalg.lstsq(A, y)
    if isinstance(ssr, list):
        ssr = ssr[0]  # pragma: no cover
    elif isinstance(ssr, np.ndarray):
        if ssr.size == 0:
            e = np.dot(A, coef_) - y
            ssr = np.dot(e, e)
        else:
            ssr = ssr[0]  # pragma: no cover

    coef_ = coef_.T
    return coef_, ssr, rank_, singular_


def _lstsq_constrained(X, y, breaks, degree=1, continuity="c0", weights=None):
    """Solve for least squares coefficients subject to continuity constraints.

    minimize || A b - y ||^2
    subject to  C b = 0
    solve for b

    Continuity constraints are enforced with Lagrange multipliers. The linear
    system of equations is

    [ A.T W A  C.T ] [ b ]   [ A.T W y ]
    [              ] [   ] = [         ]
    [     C     0  ] [ L ]   [    0    ]

    for unknown b.

    Assembling the Gram matrix A.T A and solving this system directly is
    undesirable because A.T A is frequently ill-conditioned, leading to
    numerical instability and poor results. To avoid this, we can solve

    C x = 0,

    whose solution is parameterized by X_ln + V h where V is the null space
    of C and X_ln is the least-norm solution of x. Because we are currently
    dealing with only continuity constraints, and not point constraints, I
    think X_ln will always be 0, but the code solves for it explicitly just in
    case.

    Now the original system of equations can be reformulated as an ordinary,
    unconstrained least squares problem::

    minimize || A V h - (y - A X_ln) ||^2

    which, if X_ln is zero, simply becomes

    minimize || A V h - y ||^2

    The solution to this is h', and then the optimal coefficients are
    b' = V h'.

    See https://math.stackexchange.com/a/1985744

    Notes:
        If continuity is `None`, or if there are two breakpoints and therefore
        only one line segment to fit, then the whole constraint business is
        skipped and the problem is solved with Scipy's linalg.lstsq() function.

    Args:
        X: Input data, independent variable
        y: Input data, dependent variable
        breaks: Array of breakpoints. The first breakpoint must be the
            minimum value in X and the last breakpoint must be the maximum
            value in X.
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.
        continuity: The level of continuity that will be enforced. "c0" is
            continuous, None is no continuity
        weights: Sample weights

    Returns:
        The coefficients that solve the constrained least squares problem.
    """
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
    """Find the optimal breakpoints.

    Args:
        X: Input data, independent variable
        y: Input data, dependent variable
        n_segments: Number of line segments to fit
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.
        continuity: The level of continuity that will be enforced. "c0" is
            continuous, None is no continuity
        weights: Sample weights
        solver: Optimization method. Currently supports:
                * "diffevo", Scipy's differential evolution routine
                * "L-BFGS-B"
                * "Nelder-Mead"
                * "Powell"
        max_iter: Maximum number of iterations for the solver to perform
        tol: Numeric convergence tolerance
        random_state: Random state for reproducibility
        return_n_iter: Whether to return the number of iterations

    Returns:
        If `return_n_iter=True`, returns the optimal breakpoints as well as the
        number of iterations the solver took to converge. Otherwise, returns
        only the optimal breakpoints.
    """
    if solver == "auto":
        solver = "diffevo"

    _valid_solvers = ("diffevo", "L-BFGS-B", "Nelder-Mead", "Powell")
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
            X, y, n_segments, solver, degree, continuity, weights
        )

    if return_n_iter:
        return breakpoints, n_iter
    else:
        return breakpoints  # pragma: no cover


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
    """Find the optimal breakpoints using differential evolution.

    Args:
        X: Input data, independent variable
        y: Input data, dependent variable
        n_segments: Number of line segments to fit
        degree: Array of degrees of the regression polynomials. The number of
            degree values must be one less than the number of breakpoints.
        continuity: The level of continuity that will be enforced. "c0" is
            continuous, None is no continuity
        weights: Sample weights
        max_iter: Maximum number of iterations for the solver to perform
        tol: Numeric convergence tolerance
        random_state: Random state for reproducibility

    Returns:
        Optimal breakpoints and the number of iterations done by the solver
    """
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


def _solve_minimize(X, y, n_segments, solver, degree, continuity, weights):
    break_0, break_n = np.min(X), np.max(X)
    X_guess = np.linspace(break_0, break_n, n_segments, endpoint=False)[1:]

    bounds = np.zeros((n_segments - 1, 2))
    bounds[:, 0] = break_0
    bounds[:, 1] = break_n

    res = optimize.minimize(
        _fit_opt,
        X_guess,
        (X, y, break_0, break_n, degree, continuity, weights),
        method=solver,
        bounds=bounds,
    )

    breaks = _augment_breaks(res.x, break_0, break_n)
    return breaks, res.nit


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
    """Piecewise linear regression with known breakpoint locations.

    Args:
        breakpoints: Array of breakpoint locations. Must include the minimum
            and maximum points of your input data. If `None`, it is assumed
            to be the min and max of the input data with no points in between.
            This will result in a single line segment being fit to the data.
        degree: The polynomial degree(s) of the line segments. If it is a
            single integer, all segments will have the same degree. If it is
            specified as a list, its length must be one less than the number
            of breakpoints. A degree of 0 will fit a constant (flat) line, 1
            will fit a straight line, and 2 will fit a quadratic curve.
        continuity: The degree of continuity for the line segments at the
            breakpoints. The default of `c0` means the segments will connect,
            but not necessarily smoothly (i.e. their derivatives may not be
            equal). `None` means the line segments can be fit completely
            separately from one another.

    Attributes:
        coef_: Vector coefficients that minimize the sum of squared errors.
        ssr_: Sum of squared errors resulting from the fit.
        n_params_: Number of estimated parameters.
    """

    def __init__(self, *, breakpoints=None, degree=1, continuity="c0"):
        """Initialize the regression object."""
        self.breakpoints = breakpoints
        super().__init__(degree, continuity)

    def fit(self, X, y, *, weights=None):
        """Fit piecewise regression model.

        Args:
            X: Input data
            y: Target values
            weights: Individual weights for each sample. If given a float, every
                sample will have the same weight.

        Returns:
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, y_numeric=True)
        weights = _check_sample_weight(weights, X, dtype=X.dtype)

        if self.breakpoints is None:
            self.breakpoints = np.array([np.min(X), np.max(X)])

        self.fit_with_breaks(X, y, self.breakpoints, weights)
        return self

    def predict(self, X):
        """Predict using the fitted piecewise regression model.

        Args:
            X: Input data

        Returns:
            Predicted values
        """
        return self._decision_function(X, self.breakpoints)


class AutoPiecewiseRegression(BaseEstimator, _BasePiecewiseRegressor):
    """Piecewise linear regression with unknown breakpoint locations.

    Args:
        n_segments: The number of line segments to fit.
        degree: The polynomial degree(s) of the line segments. If it is a
            single integer, all segments will have the same degree. If it is
            specified as a list, its length must be equal to `n_segments`. A
            degree of 0 will fit a constant (flat) line, 1 will fit a straight
            line, and 2 will fit a quadratic curve.
        continuity: The degree of continuity for the line segments at the
            breakpoints. The default of `c0` means the segments will connect,
            but not necessarily smoothly (i.e. their derivatives may not be
            equal). `None` means the line segments can be fit completely
            separately from one another.
        solver: The optimization routine to use in finding the breakpoints.
        random_state: Used in stochastic solvers for reproducibility.

    Attributes:
        breakpoints_: Optimal breakpoint locations. The first and last elements
            will always be the min and max of the input data.
        coef_: Vector coefficients that minimize the sum of squared errors.
        ssr_: Sum of squared errors resulting from the fit.
        n_params_: Number of estimated parameters.
        n_iter: Number of iterations the solver needed to converge.
    """

    def __init__(
        self, n_segments, *, degree=1, continuity="c0", solver="auto", random_state=None
    ):
        """Initialize the auto regression object."""
        self.n_segments = n_segments
        self.solver = solver
        self.random_state = random_state
        super().__init__(degree, continuity)

    def fit(self, X, y, weights=None):
        """Fit piecewise regression model, finding optimal breakpoints.

        Args:
            X: Input data
            y: Target values
            weights: Individual weights for each sample. If given a float, every
                sample will have the same weight.

        Returns:
            Fitted estimator.
        """
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
        """Predict using the fitted piecewise regression model with optimal breakpoints.

        Args:
            X: Input data

        Returns:
            Predicted values
        """
        return self._decision_function(X, self.breakpoints_)
