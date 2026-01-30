import numpy as np
from .basis_functions import basis_functions

def generate_aligned_bases(trial_t0_index, relative_event_times, pre_s, post_s, n_bins_per_trial, binwidth_s, basis_scaling_vals=None, basis=None, silent=True, **basis_kwargs):
    # TODO: accept a list of basis functions to allow for multiple basis types

    """
    Generate event-aligned basis projections for a GLM design matrix.

    Parameters
    ----------
    relative_event_times : array-like, shape (N,)
        Event timestamps in seconds, relative to the master alignment time for each trial.
    intensity_values : callable or ndarray
        Either:
        - callable f(t) -> value
        - or array sampled at bin centers relative to event
    pre_s : float
        Time before event.
    post_s : float
        Time after event.
    n_bins_per_trial : int
        Number of bins per event.
    binwidth_s : float
        Width of time bins.
    basis : str or None
        Name of basis function ('raised_cosine', 'delta', or None).
        If None, uses delta basis.
    **basis_kwargs
        Passed to the basis function.

    Returns
    -------
    X : ndarray, shape (N, n_basis)
        Design matrix.
    """
    relative_event_times = np.asarray(relative_event_times)
    if basis_scaling_vals is None:
        basis_scaling_vals = np.ones_like(relative_event_times)
    basis_scaling_vals = np.asarray(basis_scaling_vals)

    assert len(relative_event_times) == len(basis_scaling_vals), \
        "event_times and intensity_values must have the same length"

    # Time grid relative to event
    t_rel = np.arange(-pre_s, post_s, binwidth_s)
    basis_t0_index = np.searchsorted(t_rel, 0.0)
    if not  silent:
        print(f'Basis t0 index: {basis_t0_index}')
        print(f'Trial t0 index: {trial_t0_index}')

    # Construct basis
    if basis is None:
        raise NotImplementedError()
    # handle case where basis is already constructed and passed as a numpy array
    elif isinstance(basis, np.ndarray):
        B = basis
    elif basis in basis_functions:
        assert basis in basis_functions, f"Unknown basis '{basis}'"
        B = basis_functions[basis](
            pre_s=pre_s,
            post_s=post_s,
            binwidth_s=binwidth_s,
            **basis_kwargs,
        )
    else:
        raise ValueError(f"Unknown basis specification: {basis}")

    T, n_basis = B.shape

    # Allocate design matrix
    X_full = []

    for i, (t0, val) in enumerate(zip(relative_event_times, basis_scaling_vals)):
        # TODO: handle multiple events per trial
        # TODO: handle single trial case
        X_small = np.zeros((n_bins_per_trial, n_basis))
        if np.isnan(t0) or np.isnan(val) or (val == 0):
            X_full.append(X_small)
            continue
        # get the bin of the event time on this trial
        event_bin = np.round(t0 / binwidth_s).astype(int) + trial_t0_index

        # compute basis shift
        basis_starting_ind = event_bin - basis_t0_index
        basis_last_ind = basis_starting_ind + T
        scaled_basis = B * val

        #print(f'Event {i}: t0={t0}, event_bin={event_bin}, basis_starting_ind={basis_starting_ind}, basis_last_ind={basis_last_ind}')
        # insert scaled basis into design matrix with appropriate shift
        # first chop off the parts that are out of bounds
        insert_start = max(0, basis_starting_ind)
        insert_end = min(n_bins_per_trial, basis_last_ind)
        #print(insert_start,insert_end)
        # check if basis is completely outside the bounds
        if insert_start >= insert_end:
            if not silent:
                print(f'Skipping event {i} - basis out of bounds with event_bin={event_bin}')
            X_full.append(X_small)
            continue
        basis_start = max(0, -basis_starting_ind)
        basis_end = basis_start + (insert_end - insert_start)
        X_small[insert_start:insert_end, :] = scaled_basis[basis_start:basis_end, :]
        X_full.append(X_small)
    return np.vstack(X_full), basis_t0_index