import random
import numpy as np
import torch

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize(X: torch.Tensor) -> torch.Tensor:
    X_mean = X.mean(dim=1)
    X_std = X.std(dim=1)
    X = (X - X_mean[:, None]) / X_std[:, None]
    return X
