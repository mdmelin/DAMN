import time
import numpy as np
from abc import ABC, abstractmethod
from .basis_functions import raised_cosine_basis, delta_basis
from .alignment import construct_timebins
from .design_matrix import generate_aligned_bases

'''
An object oriented interface for working with basis functions and regressors. 
A Regressor object can house multiple BasisFunction objects, each representing a different set/type of basis functions
for that regressor.
'''


# =========================
# Basis functions
# =========================

class BasisFunction(ABC):
    '''
    An abstract base class to handle basis functions that are employed by the Regressor objects.
    Importantly, we don't store anything about the design matrix, or the coefficients for a particular basis function.
    This is because the same basis function can be provided to multiple Regressors, while allowing each Regressor to have
    its own set of coefficients for that basis function.
    '''
    def __init__(self, pre_s, post_s, binwidth_s):
        self.pre_s = pre_s
        self.post_s = post_s
        self.binwidth_s = binwidth_s
        self.basis, self.basis_time = self.construct_basis()
        self.t0 = np.searchsorted(self.basis_time, 0.0)

    @abstractmethod
    def construct_basis(self):
        pass

    def reconstruct_kernel(self, coeffs):
        if len(coeffs) != self.basis.shape[1]:
            raise ValueError(
                f"Coefficient length {len(coeffs)} does not match "
                f"number of basis functions {self.basis.shape[1]}"
            )

        kernel = self.basis @ coeffs
        return kernel, self.basis_time

    def __getitem__(self, key):
        return self.basis[key]

    def __str__(self):
        return (
            f"{self.__class__.__name__} with shape {self.basis.shape}. "
            f"T0={self.t0}, pre_s={self.pre_s}, post_s={self.post_s}, "
            f"binwidth_s={self.binwidth_s}"
        )

    def plot(self):
        pass


class DeltaBasis(BasisFunction):
    def __init__(self, pre_s, post_s, binwidth_s):
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return delta_basis(self.pre_s, self.post_s, self.binwidth_s)


class RaisedCosineBasis(BasisFunction):
    def __init__(self, n_funcs, pre_s, post_s, binwidth_s, log_scale=False):
        self.n_funcs = n_funcs
        self.log_scale = log_scale
        super().__init__(pre_s, post_s, binwidth_s)

    def construct_basis(self):
        return raised_cosine_basis(
            self.n_funcs,
            self.pre_s,
            self.post_s,
            self.binwidth_s,
            self.log_scale,
        )


class CustomBasisFunction(BasisFunction):
    def __init__(self, basis_array, pre_s, post_s, binwidth_s):
        self.basis = basis_array
        self.pre_s = pre_s
        self.post_s = post_s
        self.binwidth_s = binwidth_s
        self.basis_time, _, self.t0 = construct_timebins(
            self.pre_s, self.post_s, binwidth_s
        )


basis_function_classes = {
    "delta": DeltaBasis,
    "raised_cosine": RaisedCosineBasis,
}


def _is_basis_function_like(obj):
    return all(hasattr(obj, attr) for attr in ("construct_basis", "__getitem__"))


# =========================
# Event regressor
# =========================

# TODO: can regressor types inherit from one object?
class EventRegressor:
    def __init__(self, name, event_times, binwidth_s,
                 event_values=None, basis_objects=None):
        self.name = name
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

            assert pre_s is not None and post_s is not None
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
            f'Event Regressor: "{self.name}" '
            f"with {len(self.basis_functions)} basis functions."
        )
        for bf in self.basis_functions:
            string += f"\n  - {bf}"
        return string

class ContinuousRegressor(EventRegressor):
    def __init__(self, name, continuous_times, continuous_values,
                 binwidth_s, basis_objects=None):
        super().__init__(
            name=name,
            event_times=continuous_times,
            binwidth_s=binwidth_s,
            event_values=continuous_values,
            basis_objects=basis_objects,
        )
    # TODO: should work like EventRegresssor but also store a continuous signal 
    
# =========================
# Design matrix
# =========================

class DesignMatrix:
    def __init__(self, master_alignment_times,
                 master_pre_s, master_post_s, binwidth_s):
        self.master_alignment_times = master_alignment_times
        self.master_pre_s = master_pre_s
        self.master_post_s = master_post_s
        self.binwidth_s = binwidth_s
        self.regressors = {}

    def add_regressor(self, regressor):
            # TODO: add by name and specify the type (Event or Continuous) and parameters (including basis functions)
            self.regressors[regressor.name] = regressor

    def build_matrix(self, Y=None):
        # TODO: add option to shuffle particular regressors
        for reg in self.regressors.values():
            reg.build_regressor(
                self.master_alignment_times,
                self.master_pre_s,
                self.master_post_s,
            )
    
    def set_coefficients(self, coefficients):
        coefficients = np.asarray(coefficients)

        if coefficients.ndim != 2:
            raise ValueError(
                "DesignMatrix coefficients must be 2D with shape (K, N)"
            )

        total_cols = sum(reg.n_cols for reg in self.regressors.values())

        if coefficients.shape[0] != total_cols:
            raise ValueError(
                f"Expected {total_cols} coefficient rows, got {coefficients.shape[0]}"
            )

        for name, (c0, c1) in self._regressor_col_ranges.items():
            self.regressors[name].coefficients = coefficients[c0:c1, :]
    
    @property
    def coefficients(self):
        return np.vstack([
            reg.coefficients
            for reg in self.regressors.values()
        ])

    @property
    def regressor_coefficients(self):
        return {
            name: reg.coefficients
            for name, reg in self.regressors.items()
        }
    
    def reconstruct_kernel(self, regressor_name, **kwargs):
        return self.regressors[regressor_name].reconstruct_kernel(**kwargs)
    
    def reconstruct_Yhat(self):
        pass
        

    def __getattr__(self, name):
        if name in self.regressors:
            return self.regressors[name]
        raise AttributeError(name)

    def __str__(self):
        string = (
            f"Design Matrix with {len(self.regressors)} regressors:\n"
        )
        for reg in self.regressors.values():
            string += f"{reg}\n"
        return string

    @property
    def _regressor_col_ranges(self):
        ranges = {}
        col_start = 0
        for name, reg in self.regressors.items():
            ranges[name] = (col_start, col_start + reg.n_cols)
            col_start += reg.n_cols
        return ranges

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_idx, col_idx = key
        else:
            row_idx = slice(None)
            col_idx = key

        if isinstance(col_idx, int):
            col_idx = [col_idx]
        elif isinstance(col_idx, slice):
            total_cols = sum(reg.n_cols for reg in self.regressors.values())
            col_idx = list(range(*col_idx.indices(total_cols)))
        else:
            col_idx = list(col_idx)

        out_cols = []
        ranges = self._regressor_col_ranges

        for c in col_idx:
            for name, (start, end) in ranges.items():
                if start <= c < end:
                    reg = self.regressors[name]
                    out_cols.append(reg[row_idx, c - start])
                    break
            else:
                raise IndexError(f"Column {c} out of range")

        return np.column_stack(out_cols)
    
    @property
    def X(self):
        # return a safe copy of the full design matrix
        return np.hstack([reg.X for reg in self.regressors.values()])
    
    def regressor_summary(self, individual_basis_functions=False):
        summaries = {}
        for name, reg in self.regressors.items():
            summary = reg.kernel_summary()
            if not individual_basis_functions:
                # remove basis-wise details
                summaries[name] = summary['total_norm']
                continue
            else:
                summaries[name] = summary
        return summaries