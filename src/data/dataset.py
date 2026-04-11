"""PyTorch Dataset classes for TimeGAN and Transformer training."""

import numpy as np
import torch
from torch.utils.data import Dataset


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
        input_tensor: shape (seq_len, n_features + 4)  market seq + tiled contract params
        label:        scalar option price
    """

    def __init__(self, sequences, strikes, times_to_expiry, rates, moneyness, labels):
        """
        Parameters
        ----------
        sequences : np.ndarray or list of np.ndarray, shape (N, 60, 20)
            Market feature sequences.
        strikes : array-like, shape (N,)
        times_to_expiry : array-like, shape (N,)
        rates : array-like, shape (N,)
        moneyness : array-like, shape (N,)
        labels : array-like, shape (N,)
            Target option prices (mid_price or simulated prices).
        """
        self.N = len(labels)
        seq_len = sequences[0].shape[0] if isinstance(sequences, list) else sequences.shape[1]

        # Build full inputs: concat market seq with tiled contract features
        inputs = []
        for i in range(self.N):
            seq = sequences[i] if isinstance(sequences, list) else sequences[i]
            contract = np.array([strikes[i], times_to_expiry[i], rates[i], moneyness[i]])
            contract_tiled = np.tile(contract, (seq_len, 1))  # (seq_len, 4)
            full = np.concatenate([seq, contract_tiled], axis=-1)  # (seq_len, 24)
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
        market_sequences : np.ndarray, shape (M, 60, 20)
            Pool of real or synthetic market sequences to sample from.
        contracts : np.ndarray, shape (N, 4)
            Columns: [strike, time_to_expiry, rate, moneyness]
        labels : np.ndarray, shape (N,)
            Simulated option prices.
        """
        self.N = len(labels)
        self.market_sequences = market_sequences
        self.contracts = contracts.astype(np.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self.n_seqs = len(market_sequences)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Each call picks a random market sequence for context diversity
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
        sequences : np.ndarray, shape (n_unique_dates, seq_len, 20)
            Shared sequence pool. Each option's seq_idx indexes into this.
        """
        self.strikes = opts_df["strike"].values.astype(np.float32)
        self.tte = opts_df["time_to_expiry"].values.astype(np.float32)
        self.rates = opts_df["rate_input"].values.astype(np.float32)
        self.moneyness = opts_df["moneyness_input"].values.astype(np.float32)
        self.labels = opts_df["mid_price"].values.astype(np.float32)
        self.seq_idx = opts_df["seq_idx"].values.astype(np.int64)
        self.sequences = sequences  # shared, not copied

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[self.seq_idx[idx]]  # (seq_len, 20)
        seq_len = seq.shape[0]

        contract = np.array([
            self.strikes[idx],
            self.tte[idx],
            self.rates[idx],
            self.moneyness[idx],
        ], dtype=np.float32)
        contract_tiled = np.tile(contract, (seq_len, 1))

        full_input = np.concatenate([seq, contract_tiled], axis=-1)
        label = np.array([self.labels[idx]], dtype=np.float32)

        return torch.tensor(full_input), torch.tensor(label)
