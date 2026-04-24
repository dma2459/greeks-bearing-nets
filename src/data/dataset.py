"""PyTorch Dataset classes for TimeGAN and Transformer training.

Contract-feature augmentation
-----------------------------
The Transformer's contract block expanded from 4 → 8 features in Run 4:
the base [K, T, r, moneyness] is now augmented with four ATM-focused
signals (see augment_contract_features). The helper is applied inside
each pricing Dataset so callers pass the same 4-column contract data
they always have. Every model checkpoint trained before Run 4 has
input_dim=20 and is incompatible with the new 24-wide inputs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# Width of the augmented contract block that gets appended to the market sequence.
CONTRACT_FEATURE_WIDTH = 8


def augment_contract_features(strikes, ttes, rates, moneyness):
    """
    Expand raw contract params to 8 features tailored for ATM learning.

    Output columns, in order:
        0 K               strike (raw $)
        1 T               time to expiry (years)
        2 r               risk-free rate
        3 moneyness       S / K
        4 log_moneyness   log(S/K)  — BS-native, symmetric about ATM
        5 sqrt_T          sqrt(T)   — BS-native time scale
        6 atm_bell        exp(-(moneyness-1)^2 / 0.01) — Gaussian peak at ATM
        7 abs_dist        |moneyness - 1|               — linear ATM distance

    Parameters
    ----------
    strikes, ttes, rates, moneyness : array-like, shape (N,)

    Returns
    -------
    np.ndarray, shape (N, 8), dtype float32
    """
    strikes = np.asarray(strikes, dtype=np.float32)
    ttes = np.asarray(ttes, dtype=np.float32)
    rates = np.asarray(rates, dtype=np.float32)
    moneyness = np.asarray(moneyness, dtype=np.float32)

    log_moneyness = np.log(np.maximum(moneyness, 1e-6)).astype(np.float32)
    sqrt_T = np.sqrt(np.maximum(ttes, 0.0)).astype(np.float32)
    atm_bell = np.exp(-((moneyness - 1.0) ** 2) / 0.01).astype(np.float32)
    abs_dist = np.abs(moneyness - 1.0).astype(np.float32)

    return np.stack([
        strikes, ttes, rates, moneyness,
        log_moneyness, sqrt_T, atm_bell, abs_dist,
    ], axis=-1)


class GANDataset(Dataset):
    """
    Dataset for TimeGAN training.

    Each item is a (seq_len, n_features) real market sequence.
    """

    def __init__(self, sequences):
        """
        Parameters
        ----------
        sequences : np.ndarray, shape (N, seq_len, n_features)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class TransformerPricingDataset(Dataset):
    """
    Dataset for the Transformer pricing network.

    Each item: (input_tensor, label)
        input_tensor: shape (seq_len, n_features + CONTRACT_FEATURE_WIDTH)
                     market sequence + tiled 8-wide augmented contract block
        label:        scalar option price
    """

    def __init__(self, sequences, strikes, times_to_expiry, rates, moneyness, labels):
        """
        Parameters
        ----------
        sequences : np.ndarray or list of np.ndarray, shape (N, seq_len, n_market_feats)
            Market feature sequences.
        strikes, times_to_expiry, rates, moneyness : array-like, shape (N,)
        labels : array-like, shape (N,)
            Target option prices (mid_price or simulated prices).
        """
        self.N = len(labels)
        seq_len = sequences[0].shape[0] if isinstance(sequences, list) else sequences.shape[1]

        contracts = augment_contract_features(strikes, times_to_expiry, rates, moneyness)

        inputs = []
        for i in range(self.N):
            seq = sequences[i] if isinstance(sequences, list) else sequences[i]
            contract_tiled = np.tile(contracts[i], (seq_len, 1))  # (seq_len, 8)
            full = np.concatenate([seq, contract_tiled], axis=-1)
            inputs.append(full)

        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class SimulatedPricingDataset(Dataset):
    """
    Dataset for Transformer training from simulated (Heston/GAN) data.

    Unlike TransformerPricingDataset, here we pair randomly sampled real market
    sequences with simulated contract parameters and labels.
    """

    def __init__(self, market_sequences, contracts, labels):
        """
        Parameters
        ----------
        market_sequences : np.ndarray, shape (M, seq_len, n_market_feats)
            Pool of real or synthetic market sequences to sample from.
        contracts : np.ndarray, shape (N, 4)
            Columns: [strike, time_to_expiry, rate, moneyness]. Augmented to
            8 features internally before being fed to the model.
        labels : np.ndarray, shape (N,)
            Simulated option prices.
        """
        self.N = len(labels)
        self.market_sequences = market_sequences
        contracts = np.asarray(contracts, dtype=np.float32)
        self.contracts = augment_contract_features(
            contracts[:, 0], contracts[:, 1], contracts[:, 2], contracts[:, 3],
        )
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self.n_seqs = len(market_sequences)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        seq_idx = idx % self.n_seqs
        seq = self.market_sequences[seq_idx]

        contract = self.contracts[idx]
        seq_len = seq.shape[0]
        contract_tiled = np.tile(contract, (seq_len, 1))
        full_input = np.concatenate([seq, contract_tiled], axis=-1)

        return (
            torch.tensor(full_input, dtype=torch.float32),
            self.labels[idx],
        )


class OptionsDataset(Dataset):
    """
    Dataset wrapping preprocessed options DataFrame for evaluation or fine-tuning.

    Uses seq_idx to look up sequences from a shared array (memory-efficient).
    """

    def __init__(self, opts_df, sequences):
        """
        Parameters
        ----------
        opts_df : pd.DataFrame
            Must have columns: seq_idx, strike, time_to_expiry,
            rate_input, moneyness_input, mid_price.
        sequences : np.ndarray, shape (n_unique_dates, seq_len, n_market_feats)
            Shared sequence pool. Each option's seq_idx indexes into this.
        """
        strikes = opts_df["strike"].values.astype(np.float32)
        tte = opts_df["time_to_expiry"].values.astype(np.float32)
        rates = opts_df["rate_input"].values.astype(np.float32)
        moneyness = opts_df["moneyness_input"].values.astype(np.float32)

        self.contracts = augment_contract_features(strikes, tte, rates, moneyness)
        # Keep per-column copies for any caller that still reads them directly.
        self.strikes = strikes
        self.tte = tte
        self.rates = rates
        self.moneyness = moneyness

        self.labels = opts_df["mid_price"].values.astype(np.float32)
        self.seq_idx = opts_df["seq_idx"].values.astype(np.int64)
        self.sequences = sequences

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[self.seq_idx[idx]]
        seq_len = seq.shape[0]

        contract_tiled = np.tile(self.contracts[idx], (seq_len, 1))
        full_input = np.concatenate([seq, contract_tiled], axis=-1)
        label = np.array([self.labels[idx]], dtype=np.float32)

        return torch.tensor(full_input), torch.tensor(label)
