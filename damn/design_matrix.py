import numpy as np
from .basis_functions import basis_functions
from .alignment import construct_timebins

def _construct_basis(
    basis,
    basis_time,
    basis_pre_s,
    basis_post_s,
    binwidth_s,
    basis_kwargs,
):
    if basis is None:
        raise NotImplementedError("You must provide a basis or basis name.")

    # handle case where basis is already constructed and passed as a numpy array
    if isinstance(basis, np.ndarray):
        if basis_time is None:
            raise ValueError("basis_time must be provided when passing a basis array.")
        return basis, np.asarray(basis_time)

    if basis in basis_functions:
        B, basis_time = basis_functions[basis](
            pre_s=basis_pre_s,
            post_s=basis_post_s,
            binwidth_s=binwidth_s,
            **basis_kwargs,
        )
        return B, basis_time

    raise ValueError(f"Unknown basis specification: {basis}")

def generate_aligned_bases(
    master_alignment_times,
    master_pre_s,
    master_post_s,
    event_times,
    binwidth_s,
    basis_pre_s=None,
    basis_post_s=None,
    basis_scaling_vals=None,
    basis=None,
    basis_time=None,
    silent=True,
    **basis_kwargs,
):
    event_times = np.asarray(event_times)

    if basis_scaling_vals is None:
        basis_scaling_vals = np.ones_like(event_times)
    basis_scaling_vals = np.asarray(basis_scaling_vals)

    assert len(event_times) == len(basis_scaling_vals), \
        "event_times and intensity_values must have the same length"

    # ------------------------------------------------------------------
    # Construct basis
    # ------------------------------------------------------------------
    if basis is None:
        raise NotImplementedError()

    basis_matrix, basis_time = _construct_basis(
        basis,
        basis_time,
        basis_pre_s,
        basis_post_s,
        binwidth_s,
        basis_kwargs,
    )

    basis_len, n_basis = basis_matrix.shape
    basis_t0_index = np.searchsorted(basis_time, 0.0)  # where in the basis the event time falls

    # ------------------------------------------------------------------
    # compute the trial timespans based on master alignment times, and pre_s and post_s
    # TODO: handle variable trial lengths here
    # ------------------------------------------------------------------
    trial_times = [
        construct_timebins(master_pre_s, master_post_s, binwidth_s)[0] + m
        for m in master_alignment_times
    ]

    trial_timespans = np.stack([(t[0], t[-1]) for t in trial_times])
    trial_t0_indices = np.round(
        (master_alignment_times - trial_timespans[:, 0]) / binwidth_s
    ).astype(int)
    # warn the user if there is overlap in the trial timespans
    overlaps = np.where(
        trial_timespans[:-1, 1] > trial_timespans[1:, 0]
    )[0]
    if len(overlaps) > 0:
        print(
            f"Warning: Overlapping trial timespans detected between trials "
            f"Check your master alignment times "
            "and pre/post seconds."
        )

    n_bins_per_trial = np.array([len(t) for t in trial_times])
    assert np.unique(n_bins_per_trial).size == 1, (
        "All trials must have the same number of bins for now. "
        "though the code should work for variable lengths..."
    )

    # Precompute trial row offsets
    trial_row_offsets = np.concatenate(
        [[0], np.cumsum(n_bins_per_trial[:-1])]
    )

    # ------------------------------------------------------------------
    # Allocate an empty design matrix
    # ------------------------------------------------------------------
    X_full = np.zeros((np.sum(n_bins_per_trial), n_basis))

    # ------------------------------------------------------------------
    # Insert events
    # ------------------------------------------------------------------
    for i, (t0, val) in enumerate(zip(event_times, basis_scaling_vals)):
        # TODO: allow passing trial start and stop times instead of pre_s and post_s? For uneven trial lengths
        if np.isnan(t0) or np.isnan(val) or (val == 0):
            continue

        # find the start time and end time of the kernel in absolute time
        basis_start_time = t0 + basis_time[0]
        basis_end_time = t0 + basis_time[-1]

        # find which trials the kernel falls inside of, using the trial_timespans
        # (there could, in theory, be multiple)
        trials_to_insert = np.where(
            (trial_timespans[:, 0] < basis_end_time) &
            (trial_timespans[:, 1] > basis_start_time)
        )[0]

        if len(trials_to_insert) == 0:
            if not silent:
                print(f"Skipping event {i} - does not fall inside any trial")
            continue

        # warn the user if the event spans multiple trials
        elif len(trials_to_insert) > 1:
            print(
                f"Warning: Event {i} spans multiple trials {trials_to_insert}, "
                "double check your kernel size."
            )

        for trial_index in trials_to_insert:
            trial_start_ind = trial_row_offsets[trial_index]
            trial_end_ind = trial_start_ind + n_bins_per_trial[trial_index]

            trial_zero_time = master_alignment_times[trial_index]

            event_ind = (
                np.round((t0 - trial_zero_time) / binwidth_s).astype(int)
                + trial_t0_indices[trial_index]
                + trial_start_ind
            )

            basis_start_ind = event_ind - basis_t0_index
            basis_end_ind = basis_start_ind + basis_len

            insert_start = max(trial_start_ind, basis_start_ind)
            insert_end = min(trial_end_ind, basis_end_ind)

            if insert_start >= insert_end:
                if not silent:
                    print(
                        f"Skipping event {i} on trial {trial_index} - basis out of bounds"
                    )
                continue

            # clip the basis to the part that fits in the trial
            basis_start_ind = max(0, insert_start - basis_start_ind)
            basis_end_ind = basis_start_ind + (insert_end - insert_start)

            X_full[insert_start:insert_end, :] += (
                basis_matrix[basis_start_ind:basis_end_ind, :] * val
            )

    return X_full, basis_time, basis_t0_index