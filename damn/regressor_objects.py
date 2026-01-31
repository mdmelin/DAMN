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
    def __init__(self, pre_s, post_s, binwidth_s):
        self.pre_s = pre_s
        self.post_s = post_s
        self.binwidth_s = binwidth_s
        self.basis, self.basis_time = self.construct_basis()
        self.t0 = np.searchsorted(self.basis_time, 0.0)

    @abstractmethod
    def construct_basis(self):
        pass

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
        if isinstance(basis_function, BasisFunction):
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

    def build(self, master_alignment_times,
              master_pre_s, master_post_s):

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

        return self.X

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

class ContinuousRegressor:
    # TODO:
    pass

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
        if isinstance(regressor, (EventRegressor, ContinuousRegressor)):
            self.regressors[regressor.name] = regressor
        elif isinstance(regressor, str):
            # TODO: add by name and specify the type (Event or Continuous) and parameters (including basis functions)
            raise NotImplementedError("Adding regressor by name is not implemented yet.")

    def build(self, Y=None):
        for reg in self.regressors.values():
            reg.build(
                self.master_alignment_times,
                self.master_pre_s,
                self.master_post_s,
            )
    
    def insert_coefficients(self, coefficients):
        # this should reach down into Regressors and create a list and an array of coefficients
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