import torch
import numpy as np

class Codebook:
    def __init__(self, path, device="cpu"):
        centroids = np.load(path)  # shape: (K,)
        self.centroids = torch.tensor(centroids, dtype=torch.float32, device=device)

    def encode(self, y):
        # y: (batch, d)
        # returns indices: (batch, d)

        # expand for distance computation
        y_exp = y.unsqueeze(-1)  # (batch, d, 1)
        c_exp = self.centroids.view(1, 1, -1)  # (1,1,K)

        dist = torch.abs(y_exp - c_exp)  # L1 is fine for NN
        idx = torch.argmin(dist, dim=-1)

        return idx

    def decode(self, idx):
        # idx: (batch, d)
        return self.centroids[idx]