import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .fit import fit_poisson_glm_adam, fit_poisson_glm_lbfgs, fit_poisson_glm_best_alpha, fit_poisson_glm_best_alpha_per_target

'''SKLearn compatible wrapper for functions in fit.py.'''

class PoissonGLM(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible wrapper for PyTorch Poisson GLM (multi-output).
    """

    def __init__(
        self,
        optimizer_type="lbfgs",   # "lbfgs" or "adam"
        alpha=0.0, # a float, array or "find"
        alpha_grid=None,
        max_epochs=1000,
        lr=1e-4,
        batch_size=2048,
        val_fraction=0.1,
        early_stopping="train",
        patience=10,
        tol=1e-4,
        warm_start=False,
        device=None,
        verbose=True,
        per_target_alpha=False,
        random_state=None,
        **optimizer_kwargs
    ):
        if alpha == "find":
            fit_alpha = True
            alpha_grid = np.logspace(-8, 1, 10) if alpha_grid is None else alpha_grid
        else:
            fit_alpha = False
            assert alpha_grid is None, "alpha_grid should be None if alpha is not set to 'find'"
        self.optimizer_type = optimizer_type
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.warm_start = warm_start
        self.device = device
        self.verbose = verbose
        self.fit_alpha = fit_alpha
        self.per_target_alpha = per_target_alpha
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs

    # -----------------------------
    # sklearn API
    # -----------------------------

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # -----------------------------
        # CASE 1: alpha grid search
        # -----------------------------
        if self.fit_alpha:
            if self.per_target_alpha:
                W, b, best_alpha, hist = fit_poisson_glm_best_alpha_per_target(
                    X, Y,
                    optimizer_type=self.optimizer_type,
                    alpha_grid=self.alpha_grid,
                    max_epochs=self.max_epochs,
                    val_fraction=self.val_fraction,
                    early_stopping=self.early_stopping,
                    patience=self.patience,
                    tol=self.tol,
                    device=self.device,
                    warm_start=self.warm_start,
                    **self.optimizer_kwargs
                )
            else:
                W, b, best_alpha, hist = fit_poisson_glm_best_alpha(
                    X, Y,
                    optimizer_type=self.optimizer_type,
                    alpha_grid=self.alpha_grid,
                    max_epochs=self.max_epochs,
                    val_fraction=self.val_fraction,
                    early_stopping=self.early_stopping,
                    patience=self.patience,
                    tol=self.tol,
                    device=self.device,
                    warm_start=self.warm_start,
                    **self.optimizer_kwargs
                )

            self.history_ = hist
            self.alpha_ = best_alpha

        # -----------------------------
        # CASE 2: fixed alpha
        # -----------------------------
        else:
            if self.optimizer_type.lower() == "lbfgs":
                W, b, *_ = fit_poisson_glm_lbfgs(
                    X, Y,
                    alpha=self.alpha,
                    max_epochs=self.max_epochs,
                    val_fraction=self.val_fraction,
                    early_stopping=self.early_stopping,
                    patience=self.patience,
                    tol=self.tol,
                    device=self.device,
                    W_init=None,
                    b_init=None,
                    **self.optimizer_kwargs
                )

            elif self.optimizer_type.lower() == "adam":
                W, b, *_ = fit_poisson_glm_adam(
                    X, Y,
                    alpha=self.alpha,
                    lr=self.lr,
                    batch_size=self.batch_size,
                    max_epochs=self.max_epochs,
                    val_fraction=self.val_fraction,
                    early_stopping=self.early_stopping,
                    patience=self.patience,
                    tol=self.tol,
                    device=self.device,
                    **self.optimizer_kwargs
                )
            else:
                raise ValueError("optimizer_type must be 'lbfgs' or 'adam'")

            self.alpha_ = self.alpha

        # store parameters
        self.coef_ = W
        self.intercept_ = b
        return self

    def predict(self, X):
        """
        Predict firing rates (Poisson rate λ = exp(XW + b)).
        """
        X = np.asarray(X)
        print(f'Shapes are X: {X.shape}, W: {self.coef_.shape}, b: {self.intercept_.shape}')
        eta = X @ self.coef_ + self.intercept_
        return np.exp(np.clip(eta, -8, 8))

    def score(self, X, Y):
        """
        Returns log-likelihood (higher is better, sklearn-style).
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        eta = X @ self.coef_ + self.intercept_
        eta = np.clip(eta, -8, 8)

        rate = np.exp(eta)
        log_likelihood = np.sum(Y * eta - rate)
        return log_likelihood