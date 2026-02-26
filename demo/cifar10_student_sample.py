"""
Sample images from a distilled CIFAR-10 student (one-step generator).

Usage (from DMD2_OT root, after stopping or finishing distill):

    python -m demo.cifar10_student_sample \
        --checkpoint cifar10_distill_runs/distill_cifar10/student_best.pt \
        --out_dir cifar10_student_samples \
        --num_images 64 \
        --seed 0

Or use the latest epoch checkpoint:

    --checkpoint cifar10_distill_runs/distill_cifar10/student_epoch_001.pt

Saves:
  - out_dir/grid.png
  - out_dir/img_XXX.png
  - out_dir/class_0_to_9.png  (one image per class)
"""

import os
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from demo.cifar10_distill import (
    build_edmprecond_cifar,
    one_step_generate,
    SIGMA_MAX,
    IMG_RES,
    IMG_CH,
    NUM_CLASSES,
)


def make_grid(images: torch.Tensor, grid_size: int = 8) -> Image.Image:
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
    parser = argparse.ArgumentParser(description="Sample from CIFAR-10 distilled student (one-step)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cifar10_distill_runs/distill_cifar10/student_best.pt",
        help="Path to student .pt (e.g. student_best.pt or student_epoch_001.pt)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="cifar10_student_samples",
        help="Directory to save sample images",
    )
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[student_sample] Loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator = build_edmprecond_cifar(device)
    generator.load_state_dict(ckpt["generator"], strict=True)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    b = args.num_images
    print(f"[student_sample] Generating {b} images (one-step)...")

    z = torch.randn(b, IMG_CH, IMG_RES, IMG_RES, device=device)
    class_indices = torch.randint(0, NUM_CLASSES, (b,), device=device)
    one_hot = torch.zeros(b, NUM_CLASSES, device=device)
    one_hot.scatter_(1, class_indices.unsqueeze(1), 1.0)

    with torch.no_grad():
        x0 = one_step_generate(generator, z, one_hot, sigma=SIGMA_MAX)
        x = ((x0 + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    for i in tqdm(range(b), desc="Saving PNGs"):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img, mode="RGB").save(os.path.join(args.out_dir, f"img_{i:03d}.png"))

    grid = make_grid(x, grid_size=int(np.sqrt(b)))
    grid_path = os.path.join(args.out_dir, "grid.png")
    grid.save(grid_path)
    print(f"[student_sample] Saved grid to {grid_path}")

    # One image per class (class 0..9)
    print("[student_sample] Generating 10 images (one per class)...")
    z_vis = torch.randn(NUM_CLASSES, IMG_CH, IMG_RES, IMG_RES, device=device)
    one_hot_vis = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    one_hot_vis.scatter_(1, torch.arange(NUM_CLASSES, device=device).unsqueeze(1), 1.0)
    with torch.no_grad():
        x_vis = one_step_generate(generator, z_vis, one_hot_vis, sigma=SIGMA_MAX)
        x_vis = ((x_vis + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    grid_vis = make_grid(x_vis, grid_size=5)
    class_grid_path = os.path.join(args.out_dir, "class_0_to_9.png")
    grid_vis.save(class_grid_path)
    print(f"[student_sample] Saved 10-class grid to {class_grid_path}")


if __name__ == "__main__":
    main()
