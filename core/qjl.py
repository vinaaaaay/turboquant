import torch
import numpy as np

class QJL:
    def __init__(self, d, device="cpu", seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.d = d
        self.S = torch.randn(d, d, device=device)

    def quantize(self, r):
        # r: (batch, d)
        return torch.sign(r @ self.S.T)

    def dequantize(self, z, norm):
        # z: (batch, d)
        # norm: (batch,)

        scale = np.sqrt(np.pi / 2) / self.d
        recon = z @ self.S  # (batch, d)

        return scale * norm.unsqueeze(1) * recon