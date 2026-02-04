import numpy as np

# =========================
# Design matrix (houses Regressor objects)
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
            # first check if overwriting an existing regressor
            if regressor.name in self.regressors:
                raise ValueError(
                    f"Regressor with name {regressor.name} already exists"
                )
            self.regressors[regressor.name] = regressor

    def build_matrix(self, Y=None,):
        # TODO: add option to shuffle particular regressors
        for reg in self.regressors.values():
            print(f'Building regressor: "{reg.name}"')
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
    
    def select(self, *, name=None, tag=None):
        """
        Select regressors by name or tag.
    
        Exactly one of {name, tag} must be provided.
        """
        if (name is None) == (tag is None):
            raise ValueError("Provide exactly one of name or tag")
    
        if name is not None:
            return self.regressors[name]
    
        return {
            k: v for k, v in self.regressors.items()
            if tag in v.tags
        }
    
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
    
    def regressor_summary(self, *, tag=None, individual_basis_functions=False):
        summaries = {}

        for name, reg in self.regressors.items():
            if tag is not None and tag not in reg.tags:
                continue

            summary = reg.kernel_summary()
            if not individual_basis_functions:
                summaries[name] = summary["total_norm"]
            else:
                summaries[name] = summary

        return summaries


    ####### Working with tagged regressor groups #######

    def columns_for_tag(self, tag):
        cols = []
        ranges = self._regressor_col_ranges

        for name, reg in self.regressors.items():
            if tag in getattr(reg, "tags", []):
                start, end = ranges[name]
                cols.extend(range(start, end))

        return np.array(cols)
    
    def X_for_tag(self, tag):
        cols = self.columns_for_tag(tag)
        return self.X[:, cols]
    
    def coefficients_for_tag(self, tag):
        betas = []
        for reg in self.regressors.values():
            if tag in reg.tags:
                betas.append(reg.coefficients)
        return np.vstack(betas)