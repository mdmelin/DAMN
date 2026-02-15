"""
Poisson GLM fitting with PyTorch

This module provides functions to fit multi-neuron Poisson Generalized Linear Models (GLMs)
using PyTorch. It supports both full-batch (LBFGS) and minibatch (Adam) optimization, optional
internal validation splits, early stopping. In general, LBFGS is always recommended if the data will fit in VRAM. 
Adam optimization is preferred for very large datasets or when GPU memory is limited, but it will often converge much more slowly than LBFGS would.

Author: Max Melin, 2025
"""
import torch 
import numpy as np

def fit_poisson_glm_torch(
    X,
    Y,
    alpha=None,
    optimizer_type="lbfgs",
    lr=1e-3, # only applies for Adam
    batch_size=2092, # only applies for Adam
    max_epochs=500,
    val_fraction=0.0, # create a validation split if >0
    early_stopping=False,
    print_every=1,
    patience=5,
    tol=1e-4,
    seed=None,
    device=None,
    eval_batch_size=None,
):

    import torch
    import numpy as np

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if alpha is None:
        alpha = 0
    if device == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Validation split
    # --------------------------------------------------
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    idx = np.arange(T)
    rng.shuffle(idx)

    if val_fraction > 0:
        split = int(T * (1 - val_fraction))
        train_idx, val_idx = idx[:split], idx[split:]
        X_train_cpu, Y_train_cpu = X[train_idx], Y[train_idx]
        X_val_cpu, Y_val_cpu = X[val_idx], Y[val_idx]
        has_val = True
    else:
        X_train_cpu, Y_train_cpu = X, Y
        X_val_cpu, Y_val_cpu = None, None
        has_val = False

    X_train_cpu = torch.as_tensor(X_train_cpu, dtype=torch.float32)
    Y_train_cpu = torch.as_tensor(Y_train_cpu, dtype=torch.float32)

    if has_val:
        X_val_cpu = torch.as_tensor(X_val_cpu, dtype=torch.float32)
        Y_val_cpu = torch.as_tensor(Y_val_cpu, dtype=torch.float32)

    T_train, p = X_train_cpu.shape
    N = Y_train_cpu.shape[1]

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    W = torch.zeros(p, N, device=device, requires_grad=True)
    b = torch.zeros(N, device=device, requires_grad=True)

    log2 = torch.log(torch.tensor(2.0, device=device))
    eps = 1e-12

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam([W, b], lr=lr)
        use_minibatch = batch_size is not None
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS([W, b], max_iter=20, line_search_fn="strong_wolfe")
        use_minibatch = False
    else:
        raise ValueError("optimizer_type must be 'adam' or 'lbfgs'")

    # --------------------------------------------------
    # Move full dataset ONCE if full-batch
    # --------------------------------------------------
    if not use_minibatch:
        X_train = X_train_cpu.to(device)
        Y_train = Y_train_cpu.to(device)
        if has_val:
            X_val = X_val_cpu.to(device)
            Y_val = Y_val_cpu.to(device)

    if eval_batch_size is None:
        eval_batch_size = batch_size if batch_size is not None else T_train

    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # --------------------------------------------------
    # Helper: streamed evaluation
    # --------------------------------------------------
    def evaluate_streamed(X_cpu, Y_cpu):
        logL_model = 0.0
        logL_null = 0.0
        total_spikes = 0.0
        total_loss = 0.0

        mean_rate = torch.mean(Y_cpu, dim=0, keepdim=True).to(device)

        for start in range(0, X_cpu.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, X_cpu.shape[0])
            Xb = X_cpu[start:end].to(device, non_blocking=True)
            Yb = Y_cpu[start:end].to(device, non_blocking=True)

            eta = torch.clamp(Xb @ W + b, max=20)

            exp_eta = torch.exp(eta)

            total_loss += torch.sum(exp_eta - Yb * eta)
            logL_model += torch.sum(Yb * eta - exp_eta)
            logL_null += torch.sum(Yb * torch.log(mean_rate + eps) - mean_rate)
            total_spikes += torch.sum(Yb)

            del Xb, Yb, eta, exp_eta

        total_loss += alpha * torch.sum(W**2)

        bps = (logL_model - logL_null) / (total_spikes * log2)

        return total_loss, bps
    
    def evaluate_full_gpu(X, Y):
        with torch.no_grad():
            eta = torch.clamp(X @ W + b, max=20)
            exp_eta = torch.exp(eta)
            loss = torch.sum(exp_eta - Y * eta) + alpha * torch.sum(W**2)

            mean_rate = torch.mean(Y, dim=0, keepdim=True)
            logL_model = torch.sum(Y * eta - exp_eta)
            logL_null = torch.sum(Y * torch.log(mean_rate + eps) - mean_rate)

            bps = (logL_model - logL_null) / (torch.sum(Y) * log2)

        return loss, bps

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(max_epochs):
        # ------------------------------
        # Adam minibatch training
        # ------------------------------
        if optimizer_type == "adam" and use_minibatch:

            perm = torch.randperm(T_train)

            for start in range(0, T_train, batch_size):
                end = min(start + batch_size, T_train)
                batch_idx = perm[start:end]

                X_batch = X_train_cpu[batch_idx].to(device, non_blocking=True)
                Y_batch = Y_train_cpu[batch_idx].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                eta = torch.clamp(X_batch @ W + b, max=20)
                exp_eta = torch.exp(eta)
                loss = torch.sum(exp_eta - Y_batch * eta) + alpha * torch.sum(W**2)

                loss.backward()
                optimizer.step()

                del X_batch, Y_batch, eta, exp_eta, loss

            # Evaluate on full training set
            with torch.no_grad():
                train_loss, train_bps = (
                    evaluate_streamed(X_train_cpu, Y_train_cpu)
                    if use_minibatch
                    else evaluate_full_gpu(X_train, Y_train)
                )

        # ------------------------------
        # Full-batch Adam
        # ------------------------------
        elif optimizer_type == "adam":

            optimizer.zero_grad(set_to_none=True)

            eta = torch.clamp(X_train @ W + b, max=20)
            exp_eta = torch.exp(eta)
            train_loss = torch.sum(exp_eta - Y_train * eta) + alpha * torch.sum(W**2)

            train_loss.backward()
            optimizer.step()

            # Evaluate on full training set
            with torch.no_grad():
                train_loss, train_bps = (
                    evaluate_streamed(X_train_cpu, Y_train_cpu)
                    if use_minibatch
                    else evaluate_full_gpu(X_train, Y_train)
                )

        # ------------------------------
        # LBFGS
        # ------------------------------
        else:

            def closure():
                optimizer.zero_grad(set_to_none=True)
                eta = torch.clamp(X_train @ W + b, max=20)
                exp_eta = torch.exp(eta)
                loss = torch.sum(exp_eta - Y_train * eta) + alpha * torch.sum(W**2)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Evaluate on full training set
            with torch.no_grad():
                train_loss, train_bps = (
                    evaluate_streamed(X_train_cpu, Y_train_cpu)
                    if use_minibatch
                    else evaluate_full_gpu(X_train, Y_train)
                )

        # ------------------------------
        # Record history
        # ------------------------------
        train_loss_hist.append(train_loss.item())
        train_bps_hist.append(train_bps.item())

        # ------------------------------
        # Validation
        # ------------------------------
        if has_val:
            with torch.no_grad():
                if use_minibatch:
                    val_loss, val_bps = evaluate_streamed(X_val_cpu, Y_val_cpu)
                else:
                    val_loss, val_bps = evaluate_full_gpu(X_val, Y_val)

            val_loss_hist.append(val_loss.item())
            val_bps_hist.append(val_bps.item())

        # ------------------------------
        # Print
        # ------------------------------
        if epoch % print_every == 0:
            msg = f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.5e} | Train BPS: {train_bps.item():.5f}"
            if has_val:
                msg += f" | Val Loss: {val_loss.item():.5e} | Val BPS: {val_bps.item():.5f}"
            print(msg)

        # ------------------------------
        # Early stopping
        # ------------------------------
        if early_stopping:
            monitor = val_loss.item() if has_val else train_loss.item()
            if best_val_loss - monitor > tol:
                best_val_loss = monitor
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break 
    Wcpu = W.detach().cpu().numpy()
    bcpu = b.detach().cpu().numpy()
    del W, b, X_train, Y_train, optimizer
    if has_val:
        del X_val, Y_val
    torch.cuda.empty_cache()
    return (
        Wcpu,
        bcpu,
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
    )


def choose_optimizer(X, Y, buffer_factor=1.2,):
    """
    Decide whether to use LBFGS (full-batch) or Adam (minibatch) based on dataset size 
    and estimated memory requirements.

    Args:
        X (np.ndarray or torch.Tensor): Design matrix, shape (T, p)
        Y (np.ndarray or torch.Tensor): Response matrix, shape (T, N)
        p (int): Number of regressors (columns of X). Defaults to 500.
        N (int, optional): Number of neurons (columns of Y). If None, inferred from Y.
        buffer_factor (float): Safety factor for memory estimation. Defaults to 1.2.
        float_precision (int): Precision of floats (16, 32, 64). Defaults to 32.

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
        raise RuntimeError("No GPU available.")
        # CPU fallback: assume ~16GB available, same logic
        cpu_mem_limit = 16 * 1024**3
        if total_mem_needed < cpu_mem_limit:
            return "lbfgs", None
        else:
            batch_size = min(max(1, int(T * 0.01)), 4096)
            return "adam", batch_size