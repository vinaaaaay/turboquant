import torch
from .rotation import RandomRotation
from .codebook import Codebook

class TurboQuantMSE:
    def __init__(self, d, codebook_path, device="cpu"):
        self.device = device
        self.rotation = RandomRotation(d, device=device)
        self.codebook = Codebook(codebook_path, device=device)

    def quantize(self, x):
        # x: (batch, d)
        y = self.rotation.apply(x)
        idx = self.codebook.encode(y)
        return idx

    def dequantize(self, idx):
        y_hat = self.codebook.decode(idx)
        x_hat = self.rotation.invert(y_hat)
        return x_hat