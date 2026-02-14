"""
Poisson GLM fitting with PyTorch

This module provides functions to fit multi-neuron Poisson Generalized Linear Models (GLMs)
using PyTorch. It supports both full-batch (LBFGS) and minibatch (Adam) optimization, optional
internal validation splits, early stopping, and hybrid optimization (Adam pretraining + LBFGS
fine-tuning). In general, LBFGS is always recommended if the data will fit in VRAM. Adam optimization
is preferred for very large datasets or when GPU memory is limited, but it will often converge more slowly.
We also provide a hybrid optimizer that uses Adam for initial training and then LBFGS for final fine-tuning.
This minimized the number of LBFGS iterations, which can be very slow when memory is tight.

Author: Max Melin, 2025
"""

import torch
import numpy as np

def fit_poisson_glm_torch(
    X,
    Y,
    alpha=None,
    optimizer_type="lbfgs",      # "adam" or "lbfgs"
    lr=1e-3,
    max_epochs=100,
    device=None,
    print_every=5,
    early_stopping=False,
    patience=10,
    tol=1e-4,
    val_fraction=0.0,           # fraction of data for internal validation
    seed=None,
    batch_size=None             # minibatch size for Adam
):
    """
    Fit a multi-neuron Poisson GLM using PyTorch.

    Supports both Adam (minibatch) and LBFGS (full-batch) optimizers, optional internal 
    validation, early stopping, and returns training history.

    Args:
        X (np.ndarray or torch.Tensor): Design matrix, shape (T, p)
        Y (np.ndarray or torch.Tensor): Response matrix, shape (T, N)
        alpha (float, optional): L2 regularization weight. Defaults to 0.
        optimizer_type (str): "adam" or "lbfgs". Defaults to "lbfgs".
        lr (float): Learning rate for Adam optimizer. Ignored for LBFGS. Defaults to 1e-3.
        max_epochs (int): Maximum number of training epochs. Defaults to 100.
        device (str or torch.device): "cpu" or "cuda". Defaults to auto-detect.
        print_every (int): Print training progress every N epochs. Defaults to 5.
        early_stopping (bool): Enable early stopping. Defaults to False.
        patience (int): Number of epochs to wait for improvement before stopping. Defaults to 10.
        tol (float): Minimum loss improvement to reset patience. Defaults to 1e-4.
        val_fraction (float): Fraction of data to hold out for validation. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility.
        batch_size (int, optional): Mini-batch size for Adam. Ignored for LBFGS.

    Returns:
        W (np.ndarray): Learned weights, shape (p, N)
        b (np.ndarray): Learned biases, shape (N,)
        train_loss_hist (list[float]): Training loss history per epoch
        val_loss_hist (list[float]): Validation loss history per epoch
        train_bps_hist (list[float]): Training bits/spike history per epoch
        val_bps_hist (list[float]): Validation bits/spike history per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if alpha is None:
        alpha = 0
    if device == "cuda":
        # free GPU memory before starting
        torch.cuda.empty_cache()

    # ---- Optional internal validation split ----
    if val_fraction > 0:
        rng = np.random.default_rng(seed)
        T = X.shape[0]
        idx = np.arange(T)
        rng.shuffle(idx)
        split = int(T * (1 - val_fraction))
        train_idx = idx[:split]
        val_idx = idx[split:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
    else:
        X_train, Y_train = X, Y
        X_val, Y_val = None, None

    # ---- Move to torch ----
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
    if X_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32, device=device)

    T, p = X_train.shape
    N = Y_train.shape[1]

    W = torch.zeros(p, N, device=device, requires_grad=True)
    b = torch.zeros(N, device=device, requires_grad=True)

    log2 = torch.log(torch.tensor(2.0, device=device))
    eps = 1e-12

    # ---- Optimizer ----
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([W, b], lr=lr)
        use_minibatch = batch_size is not None
    elif optimizer_type.lower() == "lbfgs":
        if batch_size is not None:
            print("Warning: LBFGS ignores batch_size and is always full-batch")
        print("Using LBFGS optimizer (full-batch). Will ignore learning rate and batch size.")
        optimizer = torch.optim.LBFGS([W, b], max_iter=20, line_search_fn="strong_wolfe")
        use_minibatch = False
    else:
        raise ValueError("optimizer_type must be 'adam' or 'lbfgs'")

    # ---- History ----
    train_loss_hist = []
    val_loss_hist = []
    train_bps_hist = []
    val_bps_hist = []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    has_val = X_val is not None

    for epoch in range(max_epochs):

        if optimizer_type.lower() == "adam" and use_minibatch:
            # Shuffle for minibatches
            idx = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = idx[start:end]
                X_batch = X_train[batch_idx]
                Y_batch = Y_train[batch_idx]

                optimizer.zero_grad()
                eta = torch.clamp(X_batch @ W + b, max=20)
                rate = torch.exp(eta)
                nll = torch.sum(rate - Y_batch * eta)
                l2 = alpha * torch.sum(W**2)
                loss = nll + l2
                loss.backward()
                optimizer.step()

            # Full train loss after epoch for logging
            with torch.no_grad():
                eta_train = torch.clamp(X_train @ W + b, max=20)
                rate_train = torch.exp(eta_train)
                nll_train = torch.sum(rate_train - Y_train * eta_train)
                train_loss = nll_train + alpha * torch.sum(W**2)

        elif optimizer_type.lower() == "adam":
            # Full-batch Adam
            optimizer.zero_grad()
            eta_train = torch.clamp(X_train @ W + b, max=20)
            rate_train = torch.exp(eta_train)
            nll_train = torch.sum(rate_train - Y_train * eta_train)
            l2 = alpha * torch.sum(W**2)
            train_loss = nll_train + l2
            train_loss.backward()
            optimizer.step()

        elif optimizer_type.lower() == "lbfgs":
            # LBFGS closure
            def closure():
                optimizer.zero_grad()
                eta_train_local = torch.clamp(X_train @ W + b, max=20)
                rate_train_local = torch.exp(eta_train_local)
                nll_local = torch.sum(rate_train_local - Y_train * eta_train_local)
                l2_local = alpha * torch.sum(W**2)
                loss_local = nll_local + l2_local
                loss_local.backward()
                return loss_local

            optimizer.step(closure)
            # Evaluate train loss once per epoch
            with torch.no_grad():
                eta_train = torch.clamp(X_train @ W + b, max=20)
                rate_train = torch.exp(eta_train)
                nll_train = torch.sum(rate_train - Y_train * eta_train)
                train_loss = nll_train + alpha * torch.sum(W**2)

        train_loss_hist.append(train_loss.item())

        # ---- Bits/Spike ----
        with torch.no_grad():
            mean_rate_train = torch.mean(Y_train, dim=0, keepdim=True)
            logL_model_train = torch.sum(Y_train * torch.log(rate_train + eps) - rate_train)
            logL_null_train = torch.sum(Y_train * torch.log(mean_rate_train + eps) - mean_rate_train)
            train_bps = (logL_model_train - logL_null_train) / (torch.sum(Y_train) * log2)
            train_bps_hist.append(train_bps.item())

            if has_val:
                eta_val = torch.clamp(X_val @ W + b, max=20)
                rate_val = torch.exp(eta_val)
                nll_val = torch.sum(rate_val - Y_val * eta_val)
                val_loss = nll_val + alpha * torch.sum(W**2)
                val_loss_hist.append(val_loss.item())

                mean_rate_val = torch.mean(Y_val, dim=0, keepdim=True)
                logL_model_val = torch.sum(Y_val * torch.log(rate_val + eps) - rate_val)
                logL_null_val = torch.sum(Y_val * torch.log(mean_rate_val + eps) - mean_rate_val)
                val_bps = (logL_model_val - logL_null_val) / (torch.sum(Y_val) * log2)
                val_bps_hist.append(val_bps.item())

        # ---- Print ----
        if epoch % print_every == 0:
            msg = f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.2e} | Train BPS: {train_bps.item():.5f}"
            if has_val:
                msg += f" | Val Loss: {val_loss.item():.2e} | Val BPS: {val_bps.item():.5f}"
            print(msg)

        # ---- Early stopping ----
        if early_stopping and has_val:
            if best_val_loss - val_loss.item() > tol:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered (val).")
                    break
        elif early_stopping:
            if epoch > 0 and abs(train_loss_hist[-2] - train_loss_hist[-1]) < tol:
                print("Early stopping triggered (train).")
                break

    return (
        W.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
    )

def fit_poisson_glm_hybrid_optimizer(
    X,
    Y,
    alpha=None,
    lr_adam=1e-4,
    max_epochs_adam=200,
    batch_size=2048,
    max_epochs_lbfgs=100,
    device=None,
    print_every=5,
    val_fraction=0.0,
    seed=None,
    early_stopping=True,
    patience=10,
    tol=1e-4
):
    """
    Fit a multi-neuron Poisson GLM using a hybrid optimizer: Adam pretraining followed by LBFGS
    fine-tuning.

    This approach is memory-efficient and often faster for large datasets or GPUs with limited VRAM.

    Args:
        X (np.ndarray or torch.Tensor): Design matrix, shape (T, p)
        Y (np.ndarray or torch.Tensor): Response matrix, shape (T, N)
        alpha (float, optional): L2 regularization weight. Defaults to 0.
        lr_adam (float): Learning rate for Adam pretraining. Defaults to 1e-4.
        max_epochs_adam (int): Maximum epochs for Adam pretraining. Defaults to 200.
        batch_size (int): Minibatch size for Adam. Defaults to 2048.
        max_epochs_lbfgs (int): Maximum epochs for LBFGS fine-tuning. Defaults to 100.
        device (str or torch.device, optional): "cpu" or "cuda". Defaults to auto-detect.
        print_every (int): Print progress every N epochs. Defaults to 5.
        val_fraction (float): Fraction of data to hold out for validation. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility.
        early_stopping (bool): Enable early stopping. Defaults to True.
        patience (int): Number of epochs to wait for improvement before stopping. Defaults to 10.
        tol (float): Minimum loss improvement to reset patience. Defaults to 1e-4.

    Returns:
        W (np.ndarray): Learned weights, shape (p, N)
        b (np.ndarray): Learned biases, shape (N,)
        train_loss_hist (list[float]): Training loss history per epoch (Adam + LBFGS)
        val_loss_hist (list[float]): Validation loss history per epoch (Adam + LBFGS)
        train_bps_hist (list[float]): Training bits/spike history per epoch
        val_bps_hist (list[float]): Validation bits/spike history per epoch
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if alpha is None:
        alpha = 0
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Optional validation split ----
    if val_fraction > 0:
        rng = np.random.default_rng(seed)
        T = X.shape[0]
        idx = np.arange(T)
        rng.shuffle(idx)
        split = int(T * (1 - val_fraction))
        train_idx, val_idx = idx[:split], idx[split:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
    else:
        X_train, Y_train = X, Y
        X_val, Y_val = None, None

    # ---- Move to torch ----
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
    if X_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y_val, dtype=torch.float32, device=device)

    T, p = X_train.shape
    N = Y_train.shape[1]

    # ---- Initialize parameters ----
    W = torch.zeros(p, N, device=device, requires_grad=True)
    b = torch.zeros(N, device=device, requires_grad=True)

    log2 = torch.log(torch.tensor(2.0, device=device))
    eps = 1e-12

    # ---- History ----
    train_loss_hist, val_loss_hist = [], []
    train_bps_hist, val_bps_hist = [], []

    has_val = X_val is not None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    #### ---------------- Adam Pretraining ---------------- ####
    optimizer_adam = torch.optim.Adam([W, b], lr=lr_adam)

    for epoch in range(max_epochs_adam):
        # Shuffle minibatches
        idx = torch.randperm(T, device=device)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_idx = idx[start:end]
            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]

            optimizer_adam.zero_grad()
            eta = torch.clamp(X_batch @ W + b, max=20)
            rate = torch.exp(eta)
            nll = torch.sum(rate - Y_batch * eta)
            l2 = alpha * torch.sum(W**2)
            (nll + l2).backward()
            optimizer_adam.step()

        # ---- Evaluate full train loss after epoch ----
        with torch.no_grad():
            eta_train = torch.clamp(X_train @ W + b, max=20)
            rate_train = torch.exp(eta_train)
            nll_train = torch.sum(rate_train - Y_train * eta_train)
            train_loss = nll_train + alpha * torch.sum(W**2)
            train_loss_hist.append(train_loss.item())

            mean_rate_train = torch.mean(Y_train, dim=0, keepdim=True)
            logL_model_train = torch.sum(Y_train * torch.log(rate_train + eps) - rate_train)
            logL_null_train = torch.sum(Y_train * torch.log(mean_rate_train + eps) - mean_rate_train)
            train_bps = (logL_model_train - logL_null_train) / (torch.sum(Y_train) * log2)
            train_bps_hist.append(train_bps.item())

            if has_val:
                eta_val = torch.clamp(X_val @ W + b, max=20)
                rate_val = torch.exp(eta_val)
                nll_val = torch.sum(rate_val - Y_val * eta_val)
                val_loss = nll_val + alpha * torch.sum(W**2)
                val_loss_hist.append(val_loss.item())

                mean_rate_val = torch.mean(Y_val, dim=0, keepdim=True)
                logL_model_val = torch.sum(Y_val * torch.log(rate_val + eps) - rate_val)
                logL_null_val = torch.sum(Y_val * torch.log(mean_rate_val + eps) - mean_rate_val)
                val_bps = (logL_model_val - logL_null_val) / (torch.sum(Y_val) * log2)
                val_bps_hist.append(val_bps.item())

        # ---- Print ----
        if epoch % print_every == 0:
            msg = f"[Adam] Epoch {epoch:3d} | Train Loss: {train_loss.item():.2e} | Train BPS: {train_bps.item():.5f}"
            if has_val:
                msg += f" | Val Loss: {val_loss.item():.2e} | Val BPS: {val_bps.item():.5f}"
            print(msg)

        # ---- Early stopping ----
        if early_stopping:
            monitor_loss = val_loss.item() if has_val else train_loss.item()
            if best_val_loss - monitor_loss > tol:
                best_val_loss = monitor_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered during Adam pretraining.")
                    break

    #### ---------------- LBFGS Fine-tuning ---------------- ####
    optimizer_lbfgs = torch.optim.LBFGS([W, b], max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer_lbfgs.zero_grad()
        eta_train_local = torch.clamp(X_train @ W + b, max=20)
        rate_train_local = torch.exp(eta_train_local)
        nll_local = torch.sum(rate_train_local - Y_train * eta_train_local)
        l2_local = alpha * torch.sum(W**2)
        loss_local = nll_local + l2_local
        loss_local.backward()
        return loss_local

    epochs_no_improve = 0  # reset for LBFGS
    best_val_loss = float("inf")

    for epoch in range(max_epochs_lbfgs):
        optimizer_lbfgs.step(closure)

        with torch.no_grad():
            eta_train = torch.clamp(X_train @ W + b, max=20)
            rate_train = torch.exp(eta_train)
            nll_train = torch.sum(rate_train - Y_train * eta_train)
            train_loss = nll_train + alpha * torch.sum(W**2)
            train_loss_hist.append(train_loss.item())

            mean_rate_train = torch.mean(Y_train, dim=0, keepdim=True)
            logL_model_train = torch.sum(Y_train * torch.log(rate_train + eps) - rate_train)
            logL_null_train = torch.sum(Y_train * torch.log(mean_rate_train + eps) - mean_rate_train)
            train_bps = (logL_model_train - logL_null_train) / (torch.sum(Y_train) * log2)
            train_bps_hist.append(train_bps.item())

            if has_val:
                eta_val = torch.clamp(X_val @ W + b, max=20)
                rate_val = torch.exp(eta_val)
                nll_val = torch.sum(rate_val - Y_val * eta_val)
                val_loss = nll_val + alpha * torch.sum(W**2)
                val_loss_hist.append(val_loss.item())

                mean_rate_val = torch.mean(Y_val, dim=0, keepdim=True)
                logL_model_val = torch.sum(Y_val * torch.log(rate_val + eps) - rate_val)
                logL_null_val = torch.sum(Y_val * torch.log(mean_rate_val + eps) - mean_rate_val)
                val_bps = (logL_model_val - logL_null_val) / (torch.sum(Y_val) * log2)
                val_bps_hist.append(val_bps.item())

        # ---- Print ----
        if epoch % print_every == 0:
            msg = f"[LBFGS] Epoch {epoch:3d} | Train Loss: {train_loss.item():.2e} | Train BPS: {train_bps.item():.5f}"
            if has_val:
                msg += f" | Val Loss: {val_loss.item():.2e} | Val BPS: {val_bps.item():.5f}"
            print(msg)

        # ---- Early stopping ----
        if early_stopping:
            monitor_loss = val_loss.item() if has_val else train_loss.item()
            if best_val_loss - monitor_loss > tol:
                best_val_loss = monitor_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered during LBFGS fine-tuning.")
                    break

    # ---- Final summary ----
    msg = f"[Final] Train Loss: {train_loss_hist[-1]:.2e} | Train BPS: {train_bps_hist[-1]:.5f}"
    if has_val:
        msg += f" | Val Loss: {val_loss_hist[-1]:.2e} | Val BPS: {val_bps_hist[-1]:.5f}"
    print(msg)

    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), train_loss_hist, val_loss_hist, train_bps_hist, val_bps_hist

def choose_optimizer(X, Y, p=500, N=None, buffer_factor=1.2, float_precision=32):
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

    if N is None:
        N = Y.shape[1]

    T = X.shape[0]

    # convert float precision to bytes
    precision_to_bytes = {16: 2, 32: 4, 64: 8}
    bytes_per_element = precision_to_bytes[float_precision]
    # get datatype of X and Y to determine bytes per element
    X_mem = T * p * bytes_per_element
    Y_mem = T * N * bytes_per_element
    W_mem = p * N * bytes_per_element
    b_mem = N * bytes_per_element
    total_mem_needed = (X_mem + Y_mem + W_mem + b_mem) * buffer_factor

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem_needed < gpu_mem:
            return "lbfgs", None
        else:
            # pick minibatch size ~1% of dataset or 4096, whichever smaller
            batch_size = min(max(1, int(T * 0.01)), 4096)
            return "adam", batch_size
    else:
        # CPU fallback: assume ~16GB available, same logic
        cpu_mem_limit = 16 * 1024**3
        if total_mem_needed < cpu_mem_limit:
            return "lbfgs", None
        else:
            batch_size = min(max(1, int(T * 0.01)), 4096)
            return "adam", batch_size