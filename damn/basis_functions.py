import numpy as np
from .alignment import construct_timebins

'''Bases functions and other tools for building GLM design matrices.'''

def no_basis():
    # No basis function, just return a single delta function at time zero
    return np.ones([1,1]), np.zeros((1,))
    

def raised_cosine_basis(n_funcs, pre_s, post_s, binwidth_s, log_scale=False):
    """
    Construct a raised cosine basis over time.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth : float
        Width of time bins in seconds.
    log_scale : bool, default False
        If True, basis centers are spaced logarithmically (compressed near zero).

    Returns
    -------
    basis : ndarray, shape (T, n_basis_funcs)
        Raised cosine basis matrix.
    """
    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)

    if log_scale:
        if pre_s > 0 and post_s > 0:
            # warn the user that this is probably a bad idea
            raise ValueError("Log-scale basis with both pre and post times > 0 is not supported.")
            
        eps = binwidth_s
        t_pos = t + pre_s + eps

        # Log-time
        log_t = np.log(t_pos)

        centers = np.linspace(log_t.min(), log_t.max(), n_funcs)
        width = centers[1] - centers[0]

        basis = np.zeros((T, n_funcs))
        for k, c in enumerate(centers):
            x = (log_t - c) * np.pi / width
            basis[:, k] = 0.5 * (1 + np.cos(np.clip(x, -np.pi, np.pi)))
            basis[np.abs(x) >= np.pi, k] = 0.0
        if pre_s > 0:
            print('Reversing log-scale basis functions for pre-zero times.')
            basis = basis[::-1, :]

    else:
        # Linear spacing
        centers = np.linspace(t[0], t[-1], n_funcs)
        width = centers[1] - centers[0]

        basis = np.zeros((T, n_funcs))
        for k, c in enumerate(centers):
            x = (t - c) * np.pi / width
            basis[:, k] = 0.5 * (1 + np.cos(np.clip(x, -np.pi, np.pi)))
            basis[np.abs(x) >= np.pi, k] = 0.0

    return basis,t

def delta_basis(pre_s, post_s, binwidth_s):
    """
    Construct a delta (impulse) basis over time.

    Each basis function is 1 at a single time bin and 0 elsewhere.

    Parameters
    ----------
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth_s : float
        Width of time bins in seconds.

    Returns
    -------
    basis : ndarray, shape (T, T)
        Delta basis matrix (identity).
    t : ndarray, shape (T,)
        Time vector spanning [-pre_seconds, post_seconds].
    """
    # Time axis
    t = construct_timebins(pre_s, post_s, binwidth_s)[0]
    T = len(t)

    # Delta basis = identity matrix
    basis = np.eye(T)
    # flip the basis so the first index is earliest in time

    return basis, t

def bspline_basis(n_funcs, pre_s, post_s, binwidth_s, degree=3):
    """
    Construct a B-spline basis over time.

    Parameters
    ----------
    n_basis_funcs : int
        Number of basis functions.
    pre_seconds : float
        Time before zero (positive value).
    post_seconds : float
        Time after zero.
    binwidth : float
        Width of time bins in seconds.
    degree : int, default 3
        Degree of the B-splines.

    Returns
    -------
    basis : ndarray, shape (T, n_basis_funcs)
        B-spline basis matrix.
    """
    from scipy.interpolate import BSpline, make_interp_spline

    # Time axis
    t, edges,_ = construct_timebins(pre_s, post_s, binwidth_s)
    T = len(t)

    # Knot vector
    n_knots = n_funcs + degree + 1
    knots = np.linspace(t[0], t[-1], n_knots - 2 * degree)
    knots = np.concatenate(([t[0]] * degree, knots, [t[-1]] * degree))

    basis = np.zeros((T, n_funcs))
    for i in range(n_funcs):
        coeffs = np.zeros(n_funcs)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, degree)
        basis[:, i] = spline(t)

    return basis, t

basis_functions = dict(
    raised_cosine=raised_cosine_basis,
    delta=delta_basis,
    bspline=bspline_basis,
    no_basis=no_basis,
)
