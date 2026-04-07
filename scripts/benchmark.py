import torch
import numpy as np
from core.mse_quantizer import TurboQuantMSE
from core.prod_quantizer import TurboQuantProd


def mse(x, x_hat):
    return torch.mean((x - x_hat) ** 2).item()


def inner_product_error(x, x_hat, y):
    true = torch.sum(x * y, dim=1)
    approx = torch.sum(x_hat * y, dim=1)
    return torch.mean((true - approx) ** 2).item()


def run_benchmark(d=128, b=2, batch=1024, device="cpu"):
    print(f"\nBenchmark: d={d}, b={b}")

    codebook_path = f"config/codebooks/codebook_d{d}_b{b}.npy"

    # random data
    x = torch.randn(batch, d, device=device)
    y = torch.randn(batch, d, device=device)

    # normalize (paper assumes ||x|| = 1)
    x = x / torch.norm(x, dim=1, keepdim=True)

    # ---------------- MSE quantizer ----------------
    mse_quant = TurboQuantMSE(d, codebook_path, device=device)

    idx = mse_quant.quantize(x)
    x_hat_mse = mse_quant.dequantize(idx)

    mse_val = mse(x, x_hat_mse)
    ip_err_mse = inner_product_error(x, x_hat_mse, y)

    print(f"MSE (MSE quantizer): {mse_val:.6f}")
    print(f"Inner Product Error (MSE quantizer): {ip_err_mse:.6f}")

    # ---------------- PROD quantizer ----------------
    prod_quant = TurboQuantProd(d, codebook_path, device=device)

    packed = prod_quant.quantize(x)
    x_hat_prod = prod_quant.dequantize(packed)

    mse_prod = mse(x, x_hat_prod)
    ip_err_prod = inner_product_error(x, x_hat_prod, y)

    print(f"MSE (PROD quantizer): {mse_prod:.6f}")
    print(f"Inner Product Error (PROD quantizer): {ip_err_prod:.6f}")


if __name__ == "__main__":
    run_benchmark(d=128, b=2)