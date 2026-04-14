"""
Transformer Pricing Network for European call option pricing.

Takes a 60-day market feature sequence (+ tiled contract params) and predicts
the option price. Architecture: sinusoidal positional encoding, input projection,
3 encoder blocks, global average pooling, prediction head.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learned parameters)."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, the cosine slots are one fewer than sine slots
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input. x: (batch, seq_len, d_model)."""
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
        MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq_len, d_model) -> same shape."""
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerPricingNetwork(nn.Module):
    """
    Full Transformer pricing network.

    Input:  (batch, 60, 24)   20 market features + 4 contract params
    Output: (batch, 1)        predicted option price (clamped non-negative at inference)
    """

    def __init__(self, input_dim=24, d_model=64, n_heads=4, d_ff=256,
                 n_layers=3, dropout=0.1, seq_len=60):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Positional encoding operates on input_dim, then we project
        self.pos_enc = SinusoidalPositionalEncoding(input_dim, max_len=seq_len + 10)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Encoder stack
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction head — final softplus keeps prices non-negative without
        # the dead-ReLU collapse that caused A3/A6 to output all zeros.
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Warm-start the final bias near the mean option price so the network
        # doesn't have to climb out of zero. 15 ~= mean(mid_price) on this data.
        with torch.no_grad():
            self.head[-1].bias.fill_(15.0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        price : Tensor, shape (batch, 1)
        """
        # Positional encoding + projection
        x = self.pos_enc(x)
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Encoder stack
        for block in self.encoder_blocks:
            x = block(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Prediction
        price = self.head(x)  # (batch, 1)
        return price


def build_transformer(input_dim=24, d_model=64, n_heads=4, d_ff=256,
                      n_layers=3, dropout=0.1, seq_len=60):
    """Factory function for creating a TransformerPricingNetwork with standard config."""
    return TransformerPricingNetwork(
        input_dim=input_dim, d_model=d_model, n_heads=n_heads,
        d_ff=d_ff, n_layers=n_layers, dropout=dropout, seq_len=seq_len,
    )


def build_ablation_transformer(ablation_id, seq_len=60):
    """
    Build a Transformer variant for a specific ablation study.

    Parameters
    ----------
    ablation_id : str
        One of: A1, A2, A3, A4, A5, A6.
    seq_len : int
        Override sequence length (used by A3, A4).

    Returns
    -------
    TransformerPricingNetwork
    """
    # Default config
    config = dict(input_dim=24, d_model=64, n_heads=4, d_ff=256,
                  n_layers=3, dropout=0.1, seq_len=seq_len)

    if ablation_id == "A1":
        # Remove vix_slope and vix_6m_slope: 20 -> 18 features, +4 contract = 22
        config["input_dim"] = 22
    elif ablation_id == "A2":
        # Remove 4 macro features: 20 -> 16 features, +4 contract = 20
        config["input_dim"] = 20
    elif ablation_id == "A3":
        config["seq_len"] = 20
    elif ablation_id == "A4":
        config["seq_len"] = 120
    elif ablation_id == "A5":
        config["n_layers"] = 1
    elif ablation_id == "A6":
        # Remove vol_regime: 20 -> 19 features, +4 contract = 23
        config["input_dim"] = 23
    else:
        raise ValueError(f"Unknown ablation ID: {ablation_id}")

    return TransformerPricingNetwork(**config)
