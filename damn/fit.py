import torch
import numpy as np

def fit_poisson_glm_torch(
    X,
    Y,
    alpha=None,
    optimizer_type="lbfgs",      # "adam" or "lbfgs"
    lr=1e-2,
    max_epochs=100,
    device=None,
    print_every=5,
    early_stopping=False,
    patience=10,
    tol=1e-4,
    val_fraction=0.0,           # fraction of data for internal validation
    seed=None
):
    """
    Fit multi-neuron Poisson GLM using PyTorch.

    Features:
    - Optional internal validation split
    - Adam or LBFGS optimizer
    - Tracks training/validation loss and bits/spike
    - Early stopping on validation loss
    - Fully optimized for LBFGS (no duplicate forward passes)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if alpha is None:
        alpha = 0

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
    elif optimizer_type.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([W, b], max_iter=20, line_search_fn="strong_wolfe")
    else:
        raise ValueError("optimizer_type must be 'adam' or 'lbfgs'")

    # ---- History ----
    train_loss_hist = []
    val_loss_hist = []
    train_bps_hist = []
    val_bps_hist = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # ---- Precompute mask for validation presence ----
    has_val = X_val is not None

    for epoch in range(max_epochs):

        # ---- Forward + Step ----
        if optimizer_type.lower() == "adam":
            optimizer.zero_grad()
            eta_train = torch.clamp(X_train @ W + b, max=20)
            rate_train = torch.exp(eta_train)
            nll_train = torch.sum(rate_train - Y_train * eta_train)
            l2 = alpha * torch.sum(W**2)
            train_loss = nll_train + l2
            train_loss.backward()
            optimizer.step()

        elif optimizer_type.lower() == "lbfgs":

            # LBFGS requires a closure
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

            # Evaluate train loss once per epoch (outside closure) for logging
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
                    print("Early stopping triggered.")
                    break

    return (
        W.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        train_loss_hist,
        val_loss_hist,
        train_bps_hist,
        val_bps_hist,
    )