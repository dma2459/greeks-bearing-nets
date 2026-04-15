"""
Transformer pricing network training with experiment management.

Supports three training experiments (A: Heston only, B: GAN only, C: mixed)
and ablation studies. Includes early stopping and LR scheduling.
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.transformer import TransformerPricingNetwork, build_transformer, build_ablation_transformer


def train_transformer(
    model,
    train_dataset,
    val_dataset=None,
    val_split=0.2,
    batch_size=256,
    max_epochs=100,
    lr=1e-4,
    patience=10,
    lr_patience=5,
    lr_factor=0.5,
    device=None,
    checkpoint_dir=None,
    experiment_name="experiment",
    log_every=1,
    target_mode="time_value",
):
    """
    Train a Transformer pricing network with early stopping and LR scheduling.

    Parameters
    ----------
    model : TransformerPricingNetwork
    train_dataset : Dataset
        Must return (input, label) tuples. Input tensor's last 4 features
        along the channel dim must be the tiled contract params in the
        order [strike, time_to_expiry, rate, moneyness].
    val_dataset : Dataset or None
        If None, split from train_dataset using val_split.
    val_split : float
        Fraction for validation if val_dataset is None.
    batch_size : int
    max_epochs : int
    lr : float
    patience : int
        Early stopping patience.
    lr_patience : int
        ReduceLROnPlateau patience.
    lr_factor : float
        LR reduction factor.
    device : torch.device or None
    checkpoint_dir : str or None
    experiment_name : str
    log_every : int
    target_mode : str
        One of "raw", "log", "time_value".

        - "raw" : fit raw price directly. Overweights expensive ITM
          options and starves OTM learning (Run 1 failure mode).
        - "log" : fit log1p(price). Equalizes relative error across
          moneyness but underweights ITM, pushing ATM/ITM error higher
          (Run 2 failure mode — caused A to regress at ATM/ITM).
        - "time_value" : fit log1p(price - max(S-K, 0)). The model only
          learns the volatility premium; intrinsic value is computed
          analytically at inference. This keeps ITM numerically tractable
          because the network never has to reconstruct $50 of intrinsic
          from features. Default for Run 3.

    Returns
    -------
    model : TransformerPricingNetwork
        Best model (on CPU).
    dict
        Training history.
    """
    if target_mode not in ("raw", "log", "time_value"):
        raise ValueError(f"target_mode must be raw|log|time_value, got {target_mode!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training '{experiment_name}' on: {device}")

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", "transformer", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Validation split if needed
    if val_dataset is None:
        n_val = int(len(train_dataset) * val_split)
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        n_train = len(train_dataset)
        n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = model.to(device)

    # Re-seed the head's final bias to match the transformed target range
    # so the network isn't forced to climb out of zero.
    #   raw        → ~$15 (mean option price)
    #   log        → log1p(15) ≈ 2.77
    #   time_value → log1p(4) ≈ 1.61 (mean time value is much smaller than
    #                mean price because ITM options have tiny time value)
    _bias_by_mode = {"raw": 15.0, "log": float(np.log1p(15.0)), "time_value": float(np.log1p(4.0))}
    try:
        last_linear = [m for m in model.head if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last_linear.bias.fill_(_bias_by_mode[target_mode])
    except Exception:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience,
    )
    criterion = nn.MSELoss()

    def _transform_label(y, inputs):
        """Transform raw-price labels to the training target space.

        inputs carries contract params at channels [-4:]; we need strike
        (channel -4) and moneyness = S/K (channel -1) to compute intrinsic.
        """
        if target_mode == "raw":
            return y
        if target_mode == "log":
            return torch.log1p(torch.clamp(y, min=0.0))
        # time_value — keep dims aligned with y (batch, 1) via slice notation
        K = inputs[:, 0, -4:-3]
        moneyness = inputs[:, 0, -1:]
        intrinsic = torch.clamp((moneyness - 1.0) * K, min=0.0)
        time_value = torch.clamp(y - intrinsic, min=0.0)
        return torch.log1p(time_value)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = _transform_label(labels.to(device), inputs)

            preds = model(inputs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = _transform_label(labels.to(device), inputs)
                preds = model(inputs)
                loss = criterion(preds, labels)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        scheduler.step(avg_val_loss)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{max_epochs}  |  "
                  f"Train: {avg_train_loss:.6f}  |  Val: {avg_val_loss:.6f}  |  "
                  f"LR: {current_lr:.2e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()

    # Tag the model so predict_transformer knows which inverse transform to
    # apply at inference. target_mode is the Run 3 attribute; log_target is
    # kept for backward compat with Run 2 checkpoints.
    model.target_mode = target_mode
    model.log_target = (target_mode == "log")

    # Save final
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pt"))
    print(f"  Best val loss: {best_val_loss:.6f}  |  Saved to {checkpoint_dir}")

    return model, history


def run_experiment(experiment_id, train_dataset, device=None, **kwargs):
    """
    Run one of the three main experiments (A, B, C).

    Parameters
    ----------
    experiment_id : str
        "A" (Heston only), "B" (GAN only), or "C" (mixed).
    train_dataset : Dataset
    device : torch.device or None

    Returns
    -------
    model, history
    """
    name = f"experiment_{experiment_id.lower()}"
    print(f"\n{'=' * 60}")
    print(f"Experiment {experiment_id}: {name}")
    print(f"{'=' * 60}")

    model = build_transformer()
    return train_transformer(
        model, train_dataset,
        experiment_name=name,
        device=device,
        **kwargs,
    )


def run_ablation(ablation_id, train_dataset, seq_len=60, device=None, **kwargs):
    """
    Run a single ablation study.

    Parameters
    ----------
    ablation_id : str
        One of: A1-A7.
    train_dataset : Dataset
    seq_len : int
    device : torch.device or None

    Returns
    -------
    model, history
    """
    name = f"ablation_{ablation_id.lower()}"
    print(f"\n{'=' * 60}")
    print(f"Ablation {ablation_id}: {name}")
    print(f"{'=' * 60}")

    if ablation_id == "A7":
        # A7 is Heston vs GAN vs mixed — handled at the dataset level
        model = build_transformer()
    else:
        override_seq_len = {"A3": 20, "A4": 120}.get(ablation_id, seq_len)
        model = build_ablation_transformer(ablation_id, seq_len=override_seq_len)

    return train_transformer(
        model, train_dataset,
        experiment_name=name,
        device=device,
        **kwargs,
    )
