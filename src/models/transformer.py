"""
Transformer Pricing Network for European call option pricing.

Takes a market feature sequence (+ tiled 8-feature contract block) and predicts
the option price. Architecture: sinusoidal positional encoding on the input
dim, input projection to d_model, N encoder blocks, global average pooling
over time, and an MLP head that emits a scalar (clamped non-negative at
inference via the target-space inverse transform, not a final ReLU).
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
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


def _build_head(d_model, dropout, head_size="standard"):
    """
    Build the prediction head.

    head_size:
        "standard" — Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Linear(1)
        "deep"     — Linear(128) → ReLU → Dropout → Linear(64) → ReLU →
                     Dropout → Linear(32) → ReLU → Linear(1)
                     Used by B-series ablation B6.
    """
    if head_size == "deep":
        return nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    return nn.Sequential(
        nn.Linear(d_model, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


class TransformerPricingNetwork(nn.Module):
    """
    Full Transformer pricing network.

    Input:  (batch, seq_len, input_dim)
        Run 4 default: 16 market features + 8 augmented contract features = 24
    Output: (batch, 1) predicted option price in target space; the inverse
        transform (log1p / time-value intrinsic add-back) is applied in
        predict_transformer at inference.
    """

    def __init__(self, input_dim=24, d_model=64, n_heads=4, d_ff=256,
                 n_layers=3, dropout=0.1, seq_len=30, head_size="standard"):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.pos_enc = SinusoidalPositionalEncoding(input_dim, max_len=seq_len + 10)
        self.input_proj = nn.Linear(input_dim, d_model)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.head = _build_head(d_model, dropout, head_size=head_size)

        # Warm-start the final bias near the mean option price so the network
        # doesn't have to climb out of zero. train_transformer overrides this
        # based on target_mode ("raw" / "log" / "time_value").
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
        x = self.pos_enc(x)
        x = self.input_proj(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = x.mean(dim=1)  # (batch, d_model)

        price = self.head(x)
        return price


def build_transformer(input_dim=24, d_model=64, n_heads=4, d_ff=256,
                      n_layers=3, dropout=0.1, seq_len=30, head_size="standard"):
    """Factory function for creating a TransformerPricingNetwork.

    Run 4 defaults: 16 market features (macros dropped, per ablation A2) +
    8 augmented contract features = 24 input_dim; seq_len=30 (per ablation A3).
    Run 3 baseline was input_dim=20 (pre-augmentation).
    """
    return TransformerPricingNetwork(
        input_dim=input_dim, d_model=d_model, n_heads=n_heads,
        d_ff=d_ff, n_layers=n_layers, dropout=dropout, seq_len=seq_len,
        head_size=head_size,
    )


# Augmented contract block adds 4 channels to every model input_dim.
_CONTRACT_PAD = 4


def build_ablation_transformer(ablation_id, seq_len=30):
    """
    Build a Transformer variant for a specific ablation study.

    A-series (feature/context/depth deltas from Run 3 baseline):
        A1 — drop VIX slope features:     14 market + 8 contract = 22
        A2 — keep the macro features:     20 market + 8 contract = 28
        A3 — shorter context window:      seq_len = 20
        A4 — longer context window:       seq_len = 60
        A5 — single encoder block
        A6 — drop vol_regime:             15 market + 8 contract = 23

    B-series (Step 3 architecture sweep from Run 4 baseline):
        B1 — deeper encoder (n_layers=5)
        B2 — wider model   (d_model=128)
        B3 — more heads    (n_heads=8)
        B4 — more dropout  (dropout=0.3)
        B5 — reserved for LR-schedule ablation (handled in training loop;
             the model config equals baseline)
        B6 — deeper prediction head (head_size="deep")

    Parameters
    ----------
    ablation_id : str
        One of A1-A6 or B1-B6.
    seq_len : int
        Default context length. Overridden for A3/A4.

    Returns
    -------
    TransformerPricingNetwork
    """
    config = dict(input_dim=24, d_model=64, n_heads=4, d_ff=256,
                  n_layers=3, dropout=0.1, seq_len=seq_len,
                  head_size="standard")

    if ablation_id == "A1":
        config["input_dim"] = 14 + _CONTRACT_PAD + 4  # 22
    elif ablation_id == "A2":
        config["input_dim"] = 20 + _CONTRACT_PAD + 4  # 28
    elif ablation_id == "A3":
        config["seq_len"] = 20
    elif ablation_id == "A4":
        config["seq_len"] = 60
    elif ablation_id == "A5":
        config["n_layers"] = 1
    elif ablation_id == "A6":
        config["input_dim"] = 15 + _CONTRACT_PAD + 4  # 23
    elif ablation_id == "B1":
        config["n_layers"] = 5
    elif ablation_id == "B2":
        config["d_model"] = 128
    elif ablation_id == "B3":
        config["n_heads"] = 8
    elif ablation_id == "B4":
        config["dropout"] = 0.3
    elif ablation_id == "B5":
        pass  # baseline config; train_transformer receives the LR-schedule changes
    elif ablation_id == "B6":
        config["head_size"] = "deep"
    else:
        raise ValueError(f"Unknown ablation ID: {ablation_id}")

    return TransformerPricingNetwork(**config)
