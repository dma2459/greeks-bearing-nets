"""
Three-phase TimeGAN training loop.

Phase 1: Autoencoder pretraining (Embedder + Recovery)
Phase 2: Supervisor pretraining
Phase 3: Joint adversarial training
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.timegan import TimeGAN
from src.data.dataset import GANDataset


def _vol_clustering_loss(x_real, x_fake, lr_idx=0, lags=(1, 2, 3)):
    """
    Penalize mismatch in |log_return| autocorrelation between real and fake.

    Real markets exhibit strongly positive autocorrelation in absolute returns
    (volatility clustering — calm days follow calm days, jumpy days follow
    jumpy days). The base TimeGAN's per-timestep moment-matching does not
    enforce this temporal structure, so generated paths often show |return|
    series that are too IID. This term explicitly closes the gap at small lags.

    Implementation: per-batch lag-k autocorrelation of |x[:, :, lr_idx]|,
    averaged across batch elements; squared-error vs the same statistic on
    real data; mean across the requested lags.
    """
    abs_real = x_real[:, :, lr_idx].abs()
    abs_fake = x_fake[:, :, lr_idx].abs()

    real_c = abs_real - abs_real.mean(dim=1, keepdim=True)
    fake_c = abs_fake - abs_fake.mean(dim=1, keepdim=True)

    var_real = real_c.var(dim=1, unbiased=False).mean() + 1e-8
    var_fake = fake_c.var(dim=1, unbiased=False).mean() + 1e-8

    losses = []
    for lag in lags:
        cov_real = (real_c[:, :-lag] * real_c[:, lag:]).mean()
        cov_fake = (fake_c[:, :-lag] * fake_c[:, lag:]).mean()
        ac_real = cov_real / var_real
        ac_fake = cov_fake / var_fake
        losses.append((ac_real - ac_fake) ** 2)
    return sum(losses) / len(losses)


def train_timegan(
    train_sequences,
    input_dim=20,
    hidden_dim=64,
    batch_size=128,
    phase1_epochs=200,
    phase2_epochs=200,
    phase3_epochs=300,
    lr=1e-3,
    lr_d=None,
    lr_g=None,
    g_steps_per_d=2,
    device=None,
    checkpoint_dir=None,
    log_every=10,
):
    """
    Full three-phase TimeGAN training.

    Parameters
    ----------
    train_sequences : np.ndarray, shape (N, seq_len, input_dim)
        Real training sequences (scaled).
    input_dim : int
        Number of features per timestep.
    hidden_dim : int
        GRU hidden dimension.
    batch_size : int
    phase1_epochs, phase2_epochs, phase3_epochs : int
        Phase 3 is capped at 300 because the Run 2 curves showed D climbing
        and G falling after ~300 epochs — the generator was starting to
        exploit a weakening discriminator rather than converging.
    lr : float
        Learning rate for all optimizers.
    device : torch.device or None
    checkpoint_dir : str or None
    log_every : int
        Print loss every N epochs.

    Returns
    -------
    TimeGAN
        Trained model (on CPU).
    dict
        Training history with loss curves.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    if checkpoint_dir is None:
        # v2 default: bumped hidden_dim, deeper Discriminator, vol-clustering
        # loss in Phase 3. Old v1 checkpoints in checkpoints/timegan are not
        # state-dict compatible (sub-network shapes changed).
        checkpoint_dir = os.path.join("checkpoints", "timegan_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model
    model = TimeGAN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Data
    dataset = GANDataset(train_sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Loss functions
    mse = nn.MSELoss()
    # Use logit-based BCE — pairs with the Discriminator's new linear output.
    bce = nn.BCEWithLogitsLoss()

    # GANs train more stably with Adam(β1=0.5) and, usually, a lower D LR.
    if lr_g is None:
        lr_g = lr
    if lr_d is None:
        lr_d = lr * 0.5

    history = {"phase1_loss": [], "phase2_loss": [],
               "phase3_d_loss": [], "phase3_g_loss": [],
               "phase3_vc_loss": []}

    # ─── Phase 1: Autoencoder pretraining ───
    print("\n" + "=" * 60)
    print("Phase 1: Autoencoder Pretraining")
    print("=" * 60)

    opt_ae = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=lr, betas=(0.9, 0.999),
    )

    for epoch in range(1, phase1_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            h = model.embedder(batch)
            x_hat = model.recovery(h)
            loss = mse(x_hat, batch)

            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history["phase1_loss"].append(avg_loss)
        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{phase1_epochs}  |  AE Loss: {avg_loss:.6f}")

    # Save phase 1 checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase1.pt"))

    # ─── Phase 2: Supervisor pretraining ───
    print("\n" + "=" * 60)
    print("Phase 2: Supervisor Pretraining")
    print("=" * 60)

    opt_sup = torch.optim.Adam(model.supervisor.parameters(), lr=lr)

    for epoch in range(1, phase2_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                h = model.embedder(batch)

            h_hat = model.supervisor(h)
            # Predict next latent state from current
            loss = mse(h[:, 1:, :], h_hat[:, :-1, :])

            opt_sup.zero_grad()
            loss.backward()
            opt_sup.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history["phase2_loss"].append(avg_loss)
        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{phase2_epochs}  |  Sup Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "phase2.pt"))

    # ─── Phase 3: Joint adversarial training ───
    print("\n" + "=" * 60)
    print("Phase 3: Joint Adversarial Training")
    print("=" * 60)

    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d,
                              betas=(0.5, 0.999))
    # Generator step: only G + Supervisor (not E, R — those are updated
    # separately by a reconstruction step to keep the AE anchored).
    opt_g = torch.optim.Adam(
        list(model.generator.parameters())
        + list(model.supervisor.parameters()),
        lr=lr_g, betas=(0.5, 0.999),
    )
    # Embedder/Recovery stay anchored on reconstruction throughout Phase 3.
    opt_er = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=lr_g, betas=(0.5, 0.999),
    )

    seq_len = train_sequences.shape[1]

    for epoch in range(1, phase3_epochs + 1):
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        vc_epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            bs = batch.size(0)

            # --- Discriminator update ---
            h_real = model.embedder(batch)
            z = torch.randn(bs, seq_len, input_dim, device=device)
            h_fake = model.generator(z)

            logit_real = model.discriminator(h_real.detach())
            logit_fake = model.discriminator(h_fake.detach())

            d_loss = (bce(logit_real, torch.ones_like(logit_real))
                      + bce(logit_fake, torch.zeros_like(logit_fake)))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            d_epoch_loss += d_loss.item()

            # --- Generator updates (run more frequently than D) ---
            for _ in range(g_steps_per_d):
                z = torch.randn(bs, seq_len, input_dim, device=device)
                h_fake = model.generator(z)
                h_sup = model.supervisor(h_fake)
                logit_fake = model.discriminator(h_fake)

                # Adversarial loss: fool discriminator
                g_loss_adv = bce(logit_fake, torch.ones_like(logit_fake))
                # Supervisor loss: temporal consistency
                g_loss_sup = mse(h_fake[:, 1:, :], h_sup[:, :-1, :])

                # Moment matching on recovered (data-space) sequences. Only
                # meaningful now that Recovery can span the real data range.
                x_fake = model.recovery(h_fake)
                g_loss_moments = (
                    mse(x_fake.mean(dim=0), batch.mean(dim=0))
                    + mse(x_fake.var(dim=0), batch.var(dim=0))
                )

                # Vol-clustering loss: match |log_return| autocorrelation at
                # short lags. Targets the financial property (absolute-return
                # persistence) most relevant to option pricing — without it
                # GAN paths look approximately IID and produce labels with
                # too-narrow vol distributions.
                g_loss_vc = _vol_clustering_loss(batch, x_fake, lr_idx=0)

                g_loss = (g_loss_adv
                          + 10.0 * g_loss_sup
                          + g_loss_moments
                          + 5.0 * g_loss_vc)

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_epoch_loss += g_loss.item()
                vc_epoch_loss += g_loss_vc.item()

            # --- Embedder/Recovery reconstruction anchor ---
            h_real = model.embedder(batch)
            x_hat = model.recovery(h_real)
            ae_loss = mse(x_hat, batch)
            opt_er.zero_grad()
            ae_loss.backward()
            opt_er.step()

        avg_d = d_epoch_loss / len(loader)
        avg_g = g_epoch_loss / (len(loader) * g_steps_per_d)
        avg_vc = vc_epoch_loss / (len(loader) * g_steps_per_d)
        history["phase3_d_loss"].append(avg_d)
        history["phase3_g_loss"].append(avg_g)
        history["phase3_vc_loss"].append(avg_vc)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{phase3_epochs}  |  D Loss: {avg_d:.4f}  |  "
                  f"G Loss: {avg_g:.4f}  |  VC Loss: {avg_vc:.4f}")

        # Periodic checkpoint
        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"phase3_epoch{epoch}.pt"))

    # Final save
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timegan_final.pt"))
    print(f"\nTraining complete. Model saved to {checkpoint_dir}/timegan_final.pt")

    model = model.cpu()
    return model, history


def generate_synthetic_sequences(model, n_samples=50000, seq_len=60, batch_size=1000,
                                 device=None, save_path=None):
    """
    Generate synthetic sequences from a trained TimeGAN.

    Parameters
    ----------
    model : TimeGAN
    n_samples : int
    seq_len : int
    batch_size : int
    device : torch.device or None
    save_path : str or None

    Returns
    -------
    np.ndarray, shape (n_samples, seq_len, input_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    model = model.to(device).eval()
    all_seqs = []

    with torch.no_grad():
        for start in tqdm(range(0, n_samples, batch_size), desc="Generating"):
            n = min(batch_size, n_samples - start)
            seqs = model.generate(n, seq_len=seq_len, device=device)
            all_seqs.append(seqs)

    result = np.concatenate(all_seqs, axis=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, result)
        print(f"Saved {result.shape[0]} synthetic sequences to {save_path}")

    return result
