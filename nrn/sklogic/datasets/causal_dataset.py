import torch
import numpy as np
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """
    Dataset for causal models with clearly separated covariates (X), treatment (t), and outcome (y).
    Suitable for use with DragonNet/DragonNRN architectures.
    """

    def __init__(self, X_covariates: np.ndarray, t: np.ndarray, y: np.ndarray):
        """
        Args:
            X_covariates (np.ndarray): Covariate matrix (N, D)
            t (np.ndarray): Binary treatment assignment vector (N,)
            y (np.ndarray): Outcome vector (N,)
        """
        assert len(X_covariates) == len(t) == len(y), "All inputs must have the same number of samples"

        self.X = X_covariates.astype(np.float32)
        self.t = t.astype(np.float32).reshape(-1, 1)
        self.y = y.astype(np.float32).reshape(-1, 1)
        self.sample_idx = np.arange(len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'treatment': torch.from_numpy(self.t[idx]),
            'outcome': torch.from_numpy(self.y[idx]),
            'sample_idx': idx
        }