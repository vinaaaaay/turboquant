import argparse
import os
import torch

from core.prod_quantizer import TurboQuantProd
from scripts.generate_codebook import generate_codebook


# ------------------------------
# Utilities
# ------------------------------

def normalize(x):
    return x / torch.norm(x, dim=1, keepdim=True)


def mse(x, x_hat):
    return torch.mean((x - x_hat) ** 2).item()


def inner_product_error(x, x_hat, y):
    true = torch.sum(x * y, dim=1)
    approx = torch.sum(x_hat * y, dim=1)
    return torch.mean((true - approx) ** 2).item()


def resolve_codebook(d, b):
    path = f"config/codebooks/codebook_d{d}_b{b}.npy"

    if not os.path.exists(path):
        print(f"[INFO] Codebook not found → generating for d={d}, b={b}")
        generate_codebook(d, b)

    return path


# ------------------------------
# Main pipeline
# ------------------------------

def run(d, b, batch_size=1024, device="cpu", normalize_input=True):
    print(f"\nRunning TurboQuant → d={d}, bits={b}")

    codebook_path = resolve_codebook(d, b)

    quantizer = TurboQuantProd(d, codebook_path, device=device)

    # generate data
    x = torch.randn(batch_size, d, device=device)
    y = torch.randn(batch_size, d, device=device)

    if normalize_input:
        x = normalize(x)

    # quantize
    packed = quantizer.quantize(x)
    x_hat = quantizer.dequantize(packed)

    # metrics
    print("\n===== Results =====")
    print(f"MSE: {mse(x, x_hat):.6f}")
    print(f"Inner Product Error: {inner_product_error(x, x_hat, y):.6f}")


# ------------------------------
# Entry
# ------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--bits", type=int, required=True)

    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-normalize", action="store_true")

    args = parser.parse_args()

    run(
        d=args.dim,
        b=args.bits,
        batch_size=args.batch,
        device=args.device,
        normalize_input=not args.no_normalize
    )


if __name__ == "__main__":
    main()