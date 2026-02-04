import numpy as np
from ..design_matrix import generate_aligned_bases
from .basis_function_objects import _is_basis_function_like, basis_function_classes, CustomBasisFunction
from ..alignment import generate_master_alignment_bin_times, resample_to_timebins

'''
An object oriented interface for working with basis functions and regressors. 
A Regressor object can house multiple BasisFunction objects, each representing a different set/type of basis functions
for that regressor.
'''

# =========================
# Event regressor
# =========================

class EventRegressor:
    def __init__(self, name, event_times, binwidth_s,
                 event_values=None, basis_objects=None, tags=None):
        self.name = name
        if isinstance(tags, str):
            self.tags = {tags}
        else:
            self.tags = set(tags or []) # for grouping regressors

        self.event_times = event_times
        self.binwidth_s = binwidth_s

        self.basis_functions = []

        self._X_blocks = None    # list of (T x K_i) arrays
        self.X = None            # dense cached matrix
        self._basis_col_ranges = None   # cached after build
        self._coefficients = None  # internal storage

        if event_values is None:
            self.event_values = np.ones_like(event_times)
        else:
            assert len(event_times) == len(event_values)
            self.event_values = event_values

        if basis_objects is not None:
            for basis_obj in basis_objects:
                assert _is_basis_function_like(basis_obj)
                self.basis_functions.append(basis_obj)

    def add_basis_function(self, basis_function,
                           pre_s=None, post_s=None, **kwargs):
        if self.X is not None:
            raise RuntimeError("Cannot add basis functions after build()")

        if _is_basis_function_like(basis_function):
            self.basis_functions.append(basis_function)

        elif isinstance(basis_function, str):
            if basis_function not in basis_function_classes:
                raise ValueError(f"Unknown basis function: {basis_function}")

            #assert pre_s is not None and post_s is not None
            basis_class = basis_function_classes[basis_function]
            self.basis_functions.append(
                basis_class(
                    pre_s=pre_s,
                    post_s=post_s,
                    binwidth_s=self.binwidth_s,
                    **kwargs,
                )
            )

        elif isinstance(basis_function, np.ndarray):
            self.basis_functions.append(
                CustomBasisFunction(
                    basis_function, pre_s, post_s, self.binwidth_s
                )
            )

    def build_regressor(self, master_alignment_times,
              master_pre_s, master_post_s):
        # TODO: add option to shuffle this regressor

        aligned_bases = []

        for basis in self.basis_functions:
            aligned_basis, _, _ = generate_aligned_bases(
                master_alignment_times=master_alignment_times,
                master_pre_s=master_pre_s,
                master_post_s=master_post_s,
                event_times=self.event_times,
                binwidth_s=self.binwidth_s,
                basis=basis.basis,
                basis_time=basis.basis_time,
                basis_scaling_vals=self.event_values,
            )
            aligned_bases.append(aligned_basis)

        self._X_blocks = aligned_bases
        self.X = np.hstack(self._X_blocks)
        
        # cache column ranges once
        self._basis_col_ranges = self._compute_basis_col_ranges()

        return self.X
    
    def _compute_basis_col_ranges(self):
        ranges = []
        col_start = 0
        for basis in self.basis_functions:
            n_cols = basis.basis.shape[1]
            col_end = col_start + n_cols
            ranges.append((col_start, col_end))
            col_start = col_end
        return ranges

    def _set_regressor_coefficients(self, coeffs):
        coeffs = np.asarray(coeffs)

        if coeffs.ndim != 2:
            raise ValueError(
                "Coefficients must be 2D with shape (K, N). "
                "Use (K, 1) for a single neuron."
            )

        if coeffs.shape[0] != self.n_cols:
            raise ValueError(
                f"Expected {self.n_cols} rows of coefficients, got {coeffs.shape[0]}"
            )

        self._coefficients = coeffs
    
    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coeffs):
        self._set_regressor_coefficients(coeffs)

    def reconstruct_kernel(self, coeffs=None, link_function=None, bias=None):
        # TODO: support picking only particular neurons
        """
        Reconstruct kernel from 2D coefficients.

        coeffs shape: (K, N)
        returns: kernel (T, N), time (T,)
        """
        if coeffs is None:
            if self._coefficients is None:
                raise RuntimeError("No coefficients provided or stored")
            coeffs = self._coefficients

        coeffs = np.asarray(coeffs)

        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]
            

        col_ranges = self._basis_col_ranges
        kernels, times = [], []

        for i, basis in enumerate(self.basis_functions):
            c0, c1 = col_ranges[i]
            basis_coeffs = coeffs[c0:c1, :]     # (K_i, N)
            kernel, kernel_time = basis.reconstruct_kernel(basis_coeffs)
            # kernel: (T_i, N)
            kernels.append(kernel)
            times.append(kernel_time)

        # --- common time axis ---
        min_time = min(t[0] for t in times)
        max_time = max(t[-1] for t in times)
        full_time = np.arange(
            min_time, max_time + self.binwidth_s, self.binwidth_s
        )

        T = len(full_time)
        N = coeffs.shape[1]
        full_kernel = np.zeros((T, N))

        for kernel, t in zip(kernels, times):
            shift = np.searchsorted(full_time, t[0])
            end = min(shift + kernel.shape[0], full_kernel.shape[0]) # avoid off by 1 because of bin rounding
            full_kernel[shift:end, :] += kernel[:end - shift, :]

        if link_function is not None:
            if bias is None:
                raise ValueError("Bias must be provided with link_function")
            bias = np.asarray(bias)
            if bias.ndim == 1:
                bias = bias[None, :]
            full_kernel = link_function(full_kernel + bias)

        return full_kernel, full_time
            
    @property
    def basis_blocks(self):
        if self._X_blocks is None:
            raise RuntimeError("Regressor has not been built yet")
        return self._X_blocks

    @property
    def t0s(self):
        return np.array([basis.t0 for basis in self.basis_functions])

    @property
    def n_cols(self):
        if self.X is None:
            raise RuntimeError("Regressor has not been built yet")
        return self.X.shape[1]
    
    def kernel_summary(self, norm="l2"):
        """
        Summarize norms of reconstructed kernels.

        Returns
        -------
        summary : dict
            {
                "regressor": name,
                "n_neurons": N,
                "basis": [
                    {
                        "basis": str(basis),
                        "n_cols": K_i,
                        "norm": (N,)
                    },
                    ...
                ],
                "total_norm": (N,)
            }
        """
        if self._coefficients is None:
            raise RuntimeError("No coefficients stored")

        coeffs = self._coefficients
        N = coeffs.shape[1]

        if norm == "l2":
            norm_fn = lambda x: np.linalg.norm(x, axis=0)
        elif norm == "l1":
            norm_fn = lambda x: np.sum(np.abs(x), axis=0)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        basis_summaries = []

        # --- basis-wise kernel norms ---
        for basis, (c0, c1) in zip(self.basis_functions, self._basis_col_ranges):
            basis_coeffs = coeffs[c0:c1, :]      # (K_i, N)
            kernel, _ = basis.reconstruct_kernel(basis_coeffs)  # (T_i, N)

            basis_summaries.append(
                {
                    "basis": str(basis),
                    "n_cols": c1 - c0,
                    "norm": norm_fn(kernel),
                }
            )

        # --- full regressor kernel norm ---
        full_kernel, _ = self.reconstruct_kernel()
        total_norm = norm_fn(full_kernel)

        return {
            "regressor": self.name,
            "n_neurons": N,
            "basis": basis_summaries,
            "total_norm": total_norm,
        }

    def __getitem__(self, key):
        if self.X is None:
            raise RuntimeError("Regressor has not been built yet")
        return self.X[key]

    def __str__(self):
        string = (
            f'Event Regressor: "{self.name}" of shape {self.X.shape} '
            f"with {len(self.basis_functions)} basis functions."
        )
        for bf in self.basis_functions:
            string += f"\n  - {bf}"
        return string

class ContinuousRegressor(EventRegressor):
    '''
    Doesn't need any of the fancy machinery to work with basis functions, all we need to do is resample
    values to match the bins of the master data alignment
    '''
    def __init__(self, name, sample_times, sample_values,
                 target_binwidth_s, zscore=True, basis_objects=None, tags=None,):
        super().__init__(name, event_times=None, binwidth_s=target_binwidth_s,
                 event_values=None, basis_objects=basis_objects, tags=tags)

        # arbitrary times and values
        self.sample_times = sample_times
        self.sample_values = sample_values
        # times and values aligned to the master design matrix (same as EventRegressor)
        self.event_times = None # created during .build()
        self.event_values = None # created during .build()

        self.zscore = zscore


    def build_regressor(self, master_alignment_times,
              master_pre_s, master_post_s):
        # TODO: add option to shuffle, probably in EventRegressor
        master_bin_times = generate_master_alignment_bin_times(master_alignment_times,
                                                               master_pre_s,
                                                               master_post_s,
                                                               self.binwidth_s)
            
            
            
        resampled_values = resample_to_timebins(master_bin_times, self.sample_times, self.sample_values)
        if self.zscore:
            resampled_values = (resampled_values - np.nanmean(resampled_values, axis=0)) / np.nanstd(resampled_values, axis=0)
        self.event_times = master_bin_times
        self.event_values = resampled_values

        #if len(self.basis_functions) == 0:
        #    self.basis_functions.append(NoBasis(pre_s=0, post_s=0, binwidth_s=None))

        self.X = super().build_regressor(master_alignment_times, master_pre_s, master_post_s)
        return self.X 

    def __str__(self):
        string = (
            f'Continuous Regressor: "{self.name}" of shape {self.X.shape} '
            f"with {len(self.basis_functions)} basis functions."
        )
        for bf in self.basis_functions:
            string += f"\n  - {bf}"
        return string
        
class SpatialRegressor(EventRegressor):
    # TODO: also houses coordinates for where an event happened 
    def __init__(self):
        raise NotImplementedError()

class CategoricalRegressor(EventRegressor):
    def __init__(self, name, event_times, categories,
                 binwidth_s, basis_objects=None):
        super().__init__(
            name=name,
            event_times=event_times,
            binwidth_s=binwidth_s,
            event_values=None,
            basis_objects=basis_objects,
        )
        self.categories = categories
        raise NotImplementedError()
    # TODO: should work like EventRegressor but construct 1hot encoding
    
