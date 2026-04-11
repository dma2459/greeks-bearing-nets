"""
TimeGAN: Time-series Generative Adversarial Network.

Five sub-networks: Embedder, Recovery, Supervisor, Generator, Discriminator.
Learns the statistical distribution of real SPY daily market sequences.
"""

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """Encode real market sequences into latent space."""

    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        h : Tensor, shape (batch, seq_len, hidden_dim)
        """
        h, _ = self.gru1(x)
        h, _ = self.gru2(h)
        h = self.sigmoid(self.fc(h))
        return h


class Recovery(nn.Module):
    """Decode latent representations back to feature space."""

    def __init__(self, hidden_dim=64, output_dim=20):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        """
        Parameters
        ----------
        h : Tensor, shape (batch, seq_len, hidden_dim)

        Returns
        -------
        x_hat : Tensor, shape (batch, seq_len, output_dim)
        """
        o, _ = self.gru(h)
        x_hat = self.sigmoid(self.fc(o))
        return x_hat


class Supervisor(nn.Module):
    """Enforce temporal consistency by predicting next latent state from current."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        """
        Parameters
        ----------
        h : Tensor, shape (batch, seq_len, hidden_dim)

        Returns
        -------
        h_hat : Tensor, shape (batch, seq_len, hidden_dim)
        """
        o, _ = self.gru(h)
        h_hat = self.sigmoid(self.fc(o))
        return h_hat


class Generator(nn.Module):
    """Produce fake latent sequences from random noise."""

    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """
        Parameters
        ----------
        z : Tensor, shape (batch, seq_len, input_dim)
            Random noise ~ N(0, 1).

        Returns
        -------
        h_fake : Tensor, shape (batch, seq_len, hidden_dim)
        """
        h, _ = self.gru1(z)
        h, _ = self.gru2(h)
        h_fake = self.sigmoid(self.fc(h))
        return h_fake


class Discriminator(nn.Module):
    """Classify sequences as real or fake."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        """
        Parameters
        ----------
        h : Tensor, shape (batch, seq_len, hidden_dim)

        Returns
        -------
        y : Tensor, shape (batch, 1)
            P(real).
        """
        o, _ = self.gru1(h)
        _, h_last = self.gru2(o)  # h_last: (1, batch, hidden_dim)
        h_last = h_last.squeeze(0)  # (batch, hidden_dim)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(h_last))))
        return y


class TimeGAN(nn.Module):
    """
    Container module holding all five TimeGAN sub-networks.

    Provides convenience methods for generation and saving/loading.
    """

    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embedder = Embedder(input_dim, hidden_dim)
        self.recovery = Recovery(hidden_dim, input_dim)
        self.supervisor = Supervisor(hidden_dim)
        self.generator = Generator(input_dim, hidden_dim)
        self.discriminator = Discriminator(hidden_dim)

    def forward(self, x):
        """Autoencoder forward pass (embed then recover)."""
        h = self.embedder(x)
        x_hat = self.recovery(h)
        return x_hat

    @torch.no_grad()
    def generate(self, n_samples, seq_len=60, device=None):
        """
        Generate synthetic sequences using the frozen Generator + Recovery.

        Parameters
        ----------
        n_samples : int
        seq_len : int
        device : torch.device or None

        Returns
        -------
        np.ndarray, shape (n_samples, seq_len, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        z = torch.randn(n_samples, seq_len, self.input_dim, device=device)
        h_fake = self.generator(z)
        x_fake = self.recovery(h_fake)
        return x_fake.cpu().numpy()
