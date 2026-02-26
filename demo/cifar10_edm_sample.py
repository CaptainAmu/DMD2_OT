"""
Sample images from the CIFAR-10 EDM teacher to visually inspect quality.

Usage (from DMD2_OT root):

    python -m demo.cifar10_edm_sample \
        --teacher_path model_checkpoints/edm-cifar10-32x32-cond-vp.pkl \
        --out_dir cifar10_edm_samples \
        --num_images 64 \
        --steps 18

This will save:
  - cifar10_edm_samples/grid.png  (8x8 image grid)
  - cifar10_edm_samples/img_XXX.png  (individual images)
"""

import os
import argparse
import pickle

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

import dnnlib


IMG_RES = 32
IMG_CH = 3
NUM_CLASSES = 10


def load_teacher(teacher_path: str, device: torch.device) -> nn.Module:
    with dnnlib.util.open_url(teacher_path, verbose=True) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def edm_sampler(
    net: nn.Module,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    """
    EDM sampler (Algorithm 2) adapted from third_party/edm/generate.py
    for the case without residual_net.
    """
    device = latents.device

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2.0) - 1.0) if S_min <= t_cur <= S_max else 0.0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.to(torch.float32)


def make_grid(images: torch.Tensor, grid_size: int = 8) -> Image.Image:
    """
    images: [N, C, H, W] uint8, return a PIL.Image grid.
    """
    n = min(images.shape[0], grid_size * grid_size)
    images = images[:n]
    _, c, h, w = images.shape
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    rows = []
    for i in range(0, n, grid_size):
        row_imgs = images[i : i + grid_size]
        if row_imgs.shape[0] < grid_size:
            pad = np.zeros((grid_size - row_imgs.shape[0], h, w, c), dtype=row_imgs.dtype)
            row_imgs = np.concatenate([row_imgs, pad], axis=0)
        row = np.concatenate(row_imgs, axis=1)
        rows.append(row)
    grid_np = np.concatenate(rows, axis=0)
    return Image.fromarray(grid_np, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Sample CIFAR-10 EDM teacher")
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="model_checkpoints/edm-cifar10-32x32-cond-vp.pkl",
        help="Path to CIFAR-10 EDM teacher .pkl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="cifar10_edm_samples",
        help="Directory to save sample images",
    )
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[sample] Loading CIFAR-10 EDM teacher from {args.teacher_path}")
    teacher = load_teacher(args.teacher_path, device)

    b = args.num_images
    print(f"[sample] Generating {b} images with {args.steps} EDM steps...")

    latents = torch.randn(b, IMG_CH, IMG_RES, IMG_RES, device=device)
    class_indices = torch.randint(0, NUM_CLASSES, (b,), device=device)
    one_hot = torch.zeros(b, NUM_CLASSES, device=device)
    one_hot.scatter_(1, class_indices.unsqueeze(1), 1.0)

    with torch.no_grad():
        x = edm_sampler(
            teacher,
            latents=latents,
            class_labels=one_hot,
            num_steps=args.steps,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            S_churn=0.0,
            S_min=0.0,
            S_max=float("inf"),
            S_noise=1.0,
        )
        x = ((x * 127.5) + 128).clamp(0, 255).to(torch.uint8)

    # Save individual images.
    print(f"[sample] Saving individual images to {args.out_dir}")
    for i in tqdm(range(b), desc="Saving PNGs"):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img, mode="RGB").save(os.path.join(args.out_dir, f"img_{i:03d}.png"))

    # Save grid.
    grid = make_grid(x, grid_size=int(np.sqrt(b)))
    grid_path = os.path.join(args.out_dir, "grid.png")
    grid.save(grid_path)
    print(f"[sample] Saved grid to {grid_path}")


if __name__ == "__main__":
    main()

