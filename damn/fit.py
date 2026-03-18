"""
Poisson GLM fitting with PyTorch

This module provides functions to fit multi-neuron Poisson Generalized Linear Models (GLMs)
using PyTorch. It supports both full-batch (fit_poisson_glm_lbfgs) and minibatch (fit_poisson_glm_adam) optimization, optional
internal validation splits, and early stopping. In general, LBFGS is always recommended if the data will fit in VRAM. 
Adam optimization is preferred for very large datasets or when GPU memory is limited, but it will often converge much more slowly than LBFGS would.
Adam optimizing may also require more careful tuning of learning rate and early stopping parameters for your dataset

In general, there are a couple ways to get solutions to converge: 

- If you don't care about cross-validated performance, set val_fraction=0 and early_stopping='train'. This will just monitor the training loss and stop when it plateaus.
- If you care about cross-validated performance:
    - set val_fraction to something like 0.1 to hold out a validation set, and set early_stopping='val' to monitor the validation loss for early stopping.
        - This way is quickest in practice because it will stop as soon as the validation loss plateaus, but you are not guarunteed the optimal convergent solution given the supplied alpha penalty
        - But you will want to monitor this closely and likely reduce 'patience' to stop training before val loss tails off too much.
    - Alternatively, you can set val_fraction > 0 but early_stopping='train' to monitor the training loss for early stopping, while still using the validation scores to select the best alpha.
        - fit_poisson_glm_best alpha does this for you. It's somewhat analogous to sklearn.linear_model.RidgeCV, 
          where the optimal solution should be found given alpha and the training set, and then we evaluate performance on the val set.

# TODO: hybrid optimizer that starts with Adam and finishes with LBFGS

Author: Max Melin, 2026
"""
import torch 
import numpy as np

def fit_poisson_glm_best_alpha(
    X,
    Y,
    optimizer_type="lbfgs",         # "lbfgs" or "adam"
    alpha_grid=None,                # list or array of candidate alphas
    max_epochs=100,
    val_fraction=0.1,
    early_stopping='train',
    patience=10,
    tol=1e-4,
    device=None,
    **fit_kwargs                    # extra kwargs to pass to the optimizer-specific fit function
):
    """
    Fit a Poisson GLM using either LBFGS or Adam and select the best alpha
    based on validation loss.

    Returns:
        best_W, best_b: parameters for best alpha
        best_alpha: selected alpha
        history: dict mapping alpha -> (train_loss_hist, val_loss_hist)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    assert val_fraction > 0, "val_fraction must be > 0 to select best alpha based on validation loss"

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 7)

    best_alpha = None
    best_val_loss = float("inf")
    best_W, best_b = None, None
    history = {}

    for alpha in alpha_grid:
        print(f"\n--- Trying alpha = {alpha} ---")

        if optimizer_type.lower() == "lbfgs":
            W, b, train_loss_hist, val_loss_hist, train_bps_hist, val_bps_hist = fit_poisson_glm_lbfgs(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                val_fraction=val_fraction,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                **fit_kwargs
            )
        elif optimizer_type.lower() == "adam":
            W, b, train_loss_hist, val_loss_hist, train_bps_hist, val_bps_hist = fit_poisson_glm_adam(
                X, Y,
                alpha=alpha,
                max_epochs=max_epochs,
                val_fraction=val_fraction,
                early_stopping=early_stopping,
                patience=patience,
                tol=tol,
                device=device,
                **fit_kwargs
            )
        else:
            raise ValueError("optimizer_type must be 'lbfgs' or 'adam'")

        history[alpha] = {
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist,
            "train_bps_hist": train_bps_hist,
            "val_bps_hist": val_bps_hist,
        }

        # Check best validation loss
        if val_loss_hist and val_loss_hist[-1] < best_val_loss:
            best_val_loss = val_loss_hist[-1]
            best_alpha = alpha
            best_W, best_b = W, b

    val_losses = [history[a]["val_loss_hist"][-1] for a in alpha_grid if history[a]["val_loss_hist"]]
    if len(val_losses) >= 2:
        if np.all(np.diff(val_losses) < 0):
            print("Warning: Validation loss decreases monotonically across the alpha grid. Consider adding smaller alpha values to the grid.")
        elif np.all(np.diff(val_losses) > 0):
            print("Warning: Validation loss increases monotonically across the alpha grid. Consider adding larger alpha values to the grid.")

    print(f"\nBest alpha selected: {best_alpha} with val loss {best_val_loss:.5e}")

    return best_W, best_b, best_alpha, history


# ============================================================
# -------------------- LBFGS Optimizer -----------------------
# ============================================================

def fit_poisson_glm_lbfgs(
    X,
    Y,
    alpha=0.0,
    max_epochs=100,
    lbfgs_max_iter=200,
    line_search_fn="strong_wolfe",
    history_size=100,
    val_fraction=0.0,
    early_stopping=None, # 'train' or 'val' or None
    patience=10,
    tol=1e-4,
    print_every=1,
    seed=None,
    device=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()

    X_train, Y_train, X_val, Y_val, has_val = _prepare_data(
        X, Y, val_fraction, seed
    )

    X_train = torch.as_tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.as_tensor(Y_train, dtype=torch.float32).to(device)

    if has_val:
        X_val = torch.as_tensor(X_val, dtype=torch.float32).to(device)
        Y_val = torch.as_tensor(Y_val, dtype=torch.float32).to(device)

    T_train, p = X_train.shape
    N = Y_train.shape[1]

    mean_rates = torch.mean(Y_train, dim=0)
    W, b = _initialize_params(p, N, mean_rates, device)

    optimizer = torch.optim.LBFGS(
        [W, b],
        max_iter=lbfgs_max_iter,
        line_search_fn=line_search_fn,
        history_size=history_size,
    )

    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = _poisson_loss(W, b, X_train, Y_train, alpha)
            loss.backward()
            return loss

        optimizer.step(closure)

        train_loss, train_bps = _evaluate_full_gpu(
            W, b, X_train, Y_train, alpha
        )

        train_loss_hist.append(train_loss.item())
        train_bps_hist.append(train_bps.item())

        if has_val:
            val_loss, val_bps = _evaluate_full_gpu(
                W, b, X_val, Y_val, alpha
            )
            val_loss_hist.append(val_loss.item())
            val_bps_hist.append(val_bps.item())

        if has_val:
            _print_progress(
                epoch,
                train_loss.item(),
                train_bps.item(),
                True,
                val_loss.item(),
                val_bps.item(),
                print_every,
            )
        else:
            _print_progress(
                epoch,
                train_loss.item(),
                train_bps.item(),
                False,
                None,
                None,
                print_every,
            )
        if early_stopping is not None:
            if early_stopping == 'val' and not has_val:
                raise ValueError("Early stopping on validation loss requested but no validation set provided.")
            elif early_stopping == 'val':
                monitor = val_loss.item()
            elif early_stopping == 'train':
                monitor = train_loss.item()
            else:
                raise ValueError("early_stopping must be 'train', 'val', or None.")
                
            if best_val_loss - monitor > tol:
                best_val_loss = monitor
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement greater than {tol} "
                        f"for {patience} consecutive epochs."
                    )
                    break

    Wcpu = W.detach().cpu().numpy()
    bcpu = b.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return (
        Wcpu,
        bcpu,
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
    )

# ============================================================
# -------------------- Adam Optimizer ------------------------
# ============================================================

def fit_poisson_glm_adam(
    X,
    Y,
    alpha=0.0,
    lr=1e-4,
    batch_size=2048,
    max_epochs=5000,
    val_fraction=0.0,
    early_stopping=None, # 'train' or 'val' or None
    patience=10,
    tol=1e-4,
    print_every=1,
    seed=None,
    device=None,
    eval_batch_size=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()

    X_train, Y_train, X_val, Y_val, has_val = _prepare_data(
        X, Y, val_fraction, seed
    )

    #$X_train_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    #Y_train_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    X_train_cpu = torch.as_tensor(X_train, dtype=torch.float32).pin_memory()
    Y_train_cpu = torch.as_tensor(Y_train, dtype=torch.float32).pin_memory()

    if has_val:
        X_val_cpu = torch.as_tensor(X_val, dtype=torch.float32).pin_memory()
        Y_val_cpu = torch.as_tensor(Y_val, dtype=torch.float32).pin_memory()

    T_train, p = X_train_cpu.shape
    N = Y_train_cpu.shape[1]
    mean_rates = torch.mean(Y_train_cpu, dim=0)
    W, b = _initialize_params(p, N, mean_rates, device)

    optimizer = torch.optim.Adam([W, b], lr=lr)

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        perm = torch.randperm(T_train)
        for start in range(0, T_train, batch_size):
            end = min(start + batch_size, T_train)
            idx = perm[start:end] # randomly permute to break temporal correlations

            # Stream batch to GPU
            Xb = X_train_cpu[idx].to(device, non_blocking=True)
            Yb = Y_train_cpu[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = _poisson_loss(W, b, Xb, Yb, alpha)
            loss.backward()
            optimizer.step()

            del Xb, Yb, loss

        train_loss, train_bps = _evaluate_streamed(
            W, b, X_train_cpu, Y_train_cpu, alpha, device, eval_batch_size
        )

        train_loss_hist.append(train_loss.item())
        train_bps_hist.append(train_bps.item())

        if has_val:
            val_loss, val_bps = _evaluate_streamed(
                W, b, X_val_cpu, Y_val_cpu, alpha, device, eval_batch_size
            )
            val_loss_hist.append(val_loss.item())
            val_bps_hist.append(val_bps.item())
        if has_val:
            _print_progress(
                epoch,
                train_loss.item(),
                train_bps.item(),
                True,
                val_loss.item(),
                val_bps.item(),
                print_every,
            )
        else:
            _print_progress(
                epoch,
                train_loss.item(),
                train_bps.item(),
                False,
                None,
                None,
                print_every,
            )

        if early_stopping is not None:
            if early_stopping == 'val' and not has_val:
                raise ValueError("Early stopping on validation loss requested but no validation set provided.")
            elif early_stopping == 'val':
                monitor = val_loss.item()
            elif early_stopping == 'train':
                monitor = train_loss.item()
            else:
                raise ValueError("early_stopping must be 'train', 'val', or None.")

            if best_val_loss - monitor > tol:
                best_val_loss = monitor
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement greater than {tol} "
                        f"for {patience} consecutive epochs."
                    )
                    break

    Wcpu = W.detach().cpu().numpy()
    bcpu = b.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return (
        Wcpu,
        bcpu,
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
    )




# ============================================================
# -------------------- Shared Utilities ----------------------
# ============================================================

def _check_for_bad_convergence(loss_hist, tol=1e-4, patience=5):
    # if loss tanks way down at the end, raise an error
    loss_diff = np.diff(loss_hist[-patience:])
    if np.any(loss_diff < -tol):
        raise RuntimeError(
            "Warning: Loss decreased by more than "
            f"{tol} in the last {patience} epochs. "
            "This may indicate bad convergence. "
            "Consider increasing max_epochs or adjusting optimizer parameters."
        )

def _prepare_data(X, Y, val_fraction, seed):
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    idx = np.arange(T)
    rng.shuffle(idx)

    if val_fraction > 0:
        split = int(T * (1 - val_fraction))
        train_idx, val_idx = idx[:split], idx[split:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        has_val = True
    else:
        X_train, Y_train = X, Y
        X_val, Y_val = None, None
        has_val = False

    return X_train, Y_train, X_val, Y_val, has_val


def _initialize_params(p, N, mean_rates, device):
    #b = torch.zeros(N, device=device, requires_grad=True)
    #W = 0.01 * torch.randn(p, N, device=device, requires_grad=True)
    W = torch.randn(p, N, device=device) * 0.01
    W.requires_grad_(True)
    b = torch.log(mean_rates + 1e-8).to(device).requires_grad_()
    return W, b

def _poisson_loss(W, b, X, Y, alpha):
    eta = torch.clamp(X @ W + b, max=20)
    exp_eta = torch.exp(eta)
    return torch.sum(exp_eta - Y * eta) + alpha * torch.sum(W**2)

def _poisson_deviance_loss(W, b, X, Y, alpha):
    """
    Poisson deviance loss with L2 regularization.
    Loss is normalized by number of samples, but 
    the gradient is more complex to compute.
    """
    N = X.shape[0]
    eta = torch.clamp(X @ W + b, max=20)
    mu = torch.exp(eta)
    # Poisson deviance per sample
    deviance = 2 * (Y * (torch.log((Y + 1e-8) / mu) - 1) + mu)
    return torch.sum(deviance) / N + alpha * torch.sum(W**2)


def _evaluate_streamed(W, b, X_cpu, Y_cpu, alpha, device, eval_batch_size):
    with torch.no_grad():
        log2 = torch.log(torch.tensor(2.0, device=device))
        eps = 1e-12
    
        total_loss = 0.0
        logL_model = 0.0
        logL_null = 0.0
        total_spikes = 0.0
    
        # Compute mean rate of each neuron for null model
        mean_rate = torch.mean(Y_cpu, dim=0, keepdim=True).to(device)
    
        for start in range(0, X_cpu.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, X_cpu.shape[0])
            Xb = X_cpu[start:end].to(device, non_blocking=True)
            Yb = Y_cpu[start:end].to(device, non_blocking=True)
    
            # Predicted log-rate and rate
            eta = torch.clamp(Xb @ W + b, max=12)
            mu = torch.exp(eta)
    
            # Poisson deviance / log-likelihood terms
            total_loss += torch.sum(mu - Yb * eta)
            logL_model += torch.sum(Yb * eta - mu)
            logL_null += torch.sum(Yb * torch.log(mean_rate + eps) - mean_rate)
            total_spikes += torch.sum(Yb)
    
            del Xb, Yb, eta, mu
    
        # Add L2 penalty
        total_loss += alpha * torch.sum(W**2)
    
        # Bits per spike, averaged over all spikes, independent of batch size
        bps = (logL_model - logL_null) / (total_spikes * log2)
    
    return total_loss, bps


def _evaluate_full_gpu(W, b, X, Y, alpha):
    log2 = torch.log(torch.tensor(2.0, device=X.device))
    eps = 1e-12

    with torch.no_grad():
        loss = _poisson_loss(W, b, X, Y, alpha)

        eta = torch.clamp(X @ W + b, max=20)
        exp_eta = torch.exp(eta)

        mean_rate = torch.mean(Y, dim=0, keepdim=True)
        logL_model = torch.sum(Y * eta - exp_eta)
        logL_null = torch.sum(Y * torch.log(mean_rate + eps) - mean_rate)

        bps = (logL_model - logL_null) / (torch.sum(Y) * log2)

    return loss, bps

    
def _print_progress(epoch, train_loss, train_bps,
                    has_val, val_loss, val_bps,
                    print_every):
    if epoch % print_every == 0:
        msg = (
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.5e} | "
            f"Train BPS: {train_bps:.5f}"
        )
        if has_val:
            msg += (
                f" | Val Loss: {val_loss:.5e} | "
                f"Val BPS: {val_bps:.5f}"
            )
        print(msg) 

#####


def choose_optimizer(X, Y, buffer_factor=1.2,):
    """
    Decide whether to use LBFGS (full-batch) or Adam (minibatch) based on dataset size 
    and estimated memory requirements.

    Args:
        X (np.ndarray or torch.Tensor): Design matrix, shape (T, p)
        Y (np.ndarray or torch.Tensor): Response matrix, shape (T, N)
        buffer_factor (float): Safety factor for memory estimation. Defaults to 1.2.

    Returns:
        optimizer_choice (str): "lbfgs" for full-batch or "adam" for minibatch
        batch_size (int or None): None for LBFGS, recommended minibatch size for Adam
    """

    N = Y.shape[1]

    T,p = X.shape

    # convert float precision to bytes
    xbytes = X[0,0].nbytes
    ybytes = Y[0,0].nbytes
    # get datatype of X and Y to determine bytes per element
    X_mem = T * p * xbytes
    Y_mem = T * N * ybytes
    W_mem = p * N * xbytes
    b_mem = N * xbytes
    total_mem_needed = (X_mem + Y_mem + W_mem + b_mem) * buffer_factor
    print(f'Total memory needed for LBFGS: {total_mem_needed / 1e9:.2e} GB (X: {X_mem / 1e9:.2e} GB, Y: {Y_mem / 1e9:.2e} GB, W: {W_mem / 1e9:.2e} GB, b: {b_mem / 1e9:.2e} GB)')

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem_needed < gpu_mem:
            return "lbfgs", None
        else:
            # pick minibatch size so that it would fit in GPU memory
            # We can estimate the memory for a single batch as:
            batch_W_mem = p * N * xbytes
            batch_b_mem = N * xbytes
            batch_size = int((gpu_mem / buffer_factor - batch_W_mem - batch_b_mem) / (p * xbytes + N * ybytes))
            return "adam", batch_size
    else:
        print('WARNING: NO GPU AVAILIBLE')
        return None, None
        # CPU fallback: assume ~16GB available, same logic
        cpu_mem_limit = 16 * 1024**3
        if total_mem_needed < cpu_mem_limit:
            return "lbfgs", None
        else:
            batch_size = min(max(1, int(T * 0.01)), 4096)
            return "adam", batch_size