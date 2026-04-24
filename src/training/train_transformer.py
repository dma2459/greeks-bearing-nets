"""
Transformer pricing network training with experiment management.

Supports three training experiments (A: Heston only, B: GAN only, C: mixed)
and ablation studies. Includes early stopping and LR scheduling. Optional
weighted loss (Step 2) focuses the network on ATM and high-vol samples, which
are where the Run 3 model trailed Black-Scholes most.
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


# Channel offsets within the 8-wide augmented contract block (appended at the
# end of every input tensor — see src/data/dataset.py:augment_contract_features).
#   [-8] K              [-4] log_moneyness
#   [-7] T              [-3] sqrt_T
#   [-6] r              [-2] atm_bell
#   [-5] moneyness      [-1] abs_moneyness_dist
CONTRACT_K_IDX = -8
CONTRACT_T_IDX = -7
CONTRACT_MONEYNESS_IDX = -5


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
    loss_weighting=None,
    atm_alpha=2.0,
    vol_alpha=1.0,
):
    """
    Train a Transformer pricing network with early stopping and LR scheduling.

    Parameters
    ----------
    model : TransformerPricingNetwork
    train_dataset : Dataset
        Must return (input, label) tuples. Input tensor's last 8 channels
        are the augmented contract block [K, T, r, moneyness, log_moneyness,
        sqrt_T, atm_bell, abs_moneyness_dist].
    val_dataset : Dataset or None
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
          from features. Default for Run 3+.

    loss_weighting : str or None
        Optional per-sample weighting for the MSE loss (Step 2).

        - None      : plain MSE (default).
        - "atm"     : weight = 1 + atm_alpha * bell(moneyness - 1).
                      Emphasizes options near moneyness=1.0, where Run 3
                      trailed BS by $2.20 on MAE.
        - "vol"     : weight = 1 + vol_alpha * normalized_time_value.
                      Emphasizes high vega-normalized time value as a proxy
                      for high implied vol (doesn't require a separate
                      feature channel to work across ablations).
        - "atm_vol" : sum of the two boosts.

    atm_alpha, vol_alpha : float
        Scale of the ATM / vol weighting boosts. Defaults make deep-ATM
        samples ~3x weighted and high-vol samples up to ~2x.

    Returns
    -------
    model : TransformerPricingNetwork
        Best model (on CPU).
    dict
        Training history.
    """
    if target_mode not in ("raw", "log", "time_value"):
        raise ValueError(f"target_mode must be raw|log|time_value, got {target_mode!r}")
    if loss_weighting not in (None, "atm", "vol", "atm_vol"):
        raise ValueError(f"loss_weighting must be None|atm|vol|atm_vol, got {loss_weighting!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training '{experiment_name}' on: {device}")
    if loss_weighting is not None:
        print(f"  Loss weighting: {loss_weighting} (atm_alpha={atm_alpha}, vol_alpha={vol_alpha})")

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

    def _transform_label(y, inputs):
        """Transform raw-price labels to the training target space.

        Contract block is 8-wide; K and moneyness live at CONTRACT_K_IDX /
        CONTRACT_MONEYNESS_IDX. Slice notation keeps the (batch, 1) shape.
        """
        if target_mode == "raw":
            return y
        if target_mode == "log":
            return torch.log1p(torch.clamp(y, min=0.0))
        K = inputs[:, 0, CONTRACT_K_IDX:CONTRACT_K_IDX + 1]
        moneyness = inputs[:, 0, CONTRACT_MONEYNESS_IDX:CONTRACT_MONEYNESS_IDX + 1]
        intrinsic = torch.clamp((moneyness - 1.0) * K, min=0.0)
        time_value = torch.clamp(y - intrinsic, min=0.0)
        return torch.log1p(time_value)

    def _compute_weights(y_raw, inputs):
        """Per-sample loss weights. Returns None when weighting is disabled."""
        if loss_weighting is None:
            return None
        moneyness = inputs[:, 0, CONTRACT_MONEYNESS_IDX]
        weight = torch.ones_like(moneyness)
        if loss_weighting in ("atm", "atm_vol"):
            atm_bell = torch.exp(-((moneyness - 1.0) ** 2) / 0.02)
            weight = weight + atm_alpha * atm_bell
        if loss_weighting in ("vol", "atm_vol"):
            K = inputs[:, 0, CONTRACT_K_IDX]
            T = inputs[:, 0, CONTRACT_T_IDX]
            price = y_raw.squeeze(-1)
            intrinsic = torch.clamp((moneyness - 1.0) * K, min=0.0)
            time_value = torch.clamp(price - intrinsic, min=0.0)
            denom = torch.clamp(K * torch.sqrt(torch.clamp(T, min=1e-6)), min=1e-3)
            vol_proxy = torch.tanh((time_value / denom) * 10.0)
            weight = weight + vol_alpha * vol_proxy
        return weight

    def _loss(preds, labels, weights):
        if weights is None:
            return nn.functional.mse_loss(preds, labels)
        w = weights.view(-1, 1)
        return (w * (preds - labels) ** 2).sum() / w.sum()

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
            labels_raw = labels.to(device)
            labels_t = _transform_label(labels_raw, inputs)
            weights = _compute_weights(labels_raw, inputs)

            preds = model(inputs)
            loss = _loss(preds, labels_t, weights)

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
                labels_raw = labels.to(device)
                labels_t = _transform_label(labels_raw, inputs)
                weights = _compute_weights(labels_raw, inputs)
                preds = model(inputs)
                loss = _loss(preds, labels_t, weights)
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
    # apply at inference. target_mode is the Run 3+ attribute; log_target is
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


def run_ablation(ablation_id, train_dataset, seq_len=30, device=None, **kwargs):
    """
    Run a single ablation study.

    Parameters
    ----------
    ablation_id : str
        One of A1-A6 (feature/context ablations) or B1-B6 (Step 3 architecture
        sweep). B5 is the LR-schedule ablation and picks up slower decay
        defaults here rather than a model-level change.
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
        override_seq_len = {"A3": 20, "A4": 60}.get(ablation_id, seq_len)
        model = build_ablation_transformer(ablation_id, seq_len=override_seq_len)

    # B5 is the LR-schedule ablation: double ReduceLROnPlateau patience and
    # halve the decay factor so the model spends longer at each LR before
    # stepping down. Caller can still override either via kwargs.
    if ablation_id == "B5":
        kwargs.setdefault("lr_patience", 10)
        kwargs.setdefault("lr_factor", 0.25)

    return train_transformer(
        model, train_dataset,
        experiment_name=name,
        device=device,
        **kwargs,
    )
