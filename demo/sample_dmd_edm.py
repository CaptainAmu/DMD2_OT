"""
Sample one image from DMD (1-step distilled) and visualize EDM's multi-step
denoising trajectory from the same initial noise.

Output: side_by_side.png
  - Left column: DMD 1-step result (large)
  - Right: grid of ~50 intermediate EDM snapshots from noise to clean

Usage:
    python demo/sample_dmd_edm.py \
        --dmd_checkpoint model_checkpoints/pytorch_model.bin \
        --edm_checkpoint model_checkpoints/edm-imagenet-64x64-cond-adm.pkl
"""

import os
import argparse
import pickle
import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont

from third_party.edm.training.networks import EDMPrecond
from main.edm.edm_network import get_imagenet_edm_config
import dnnlib

RESOLUTION = 64
LABEL_DIM = 1000
SIGMA_MAX = 80.0
SIGMA_MIN = 0.002


def create_dmd_generator(checkpoint_path, device):
    base_config = {
        "img_resolution": RESOLUTION,
        "img_channels": 3,
        "label_dim": LABEL_DIM,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet",
    }
    base_config.update(get_imagenet_edm_config())
    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator.to(device).eval()
    return generator


def load_edm_teacher(pickle_path, device):
    with dnnlib.util.open_url(pickle_path) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    return net


@torch.no_grad()
def sample_dmd_one(generator, noise, class_labels, device):
    sigma = torch.ones(noise.shape[0], device=device) * SIGMA_MAX
    images = generator(noise * SIGMA_MAX, sigma, class_labels)
    images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return images


def _x_to_uint8(x):
    """Convert EDM internal representation to uint8 [H,W,3] numpy."""
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def sample_edm_with_trajectory(net, latents, class_labels, device,
                                num_steps=256, snapshot_at_steps=None, rho=7,
                                S_churn=0, S_min=0.05, S_max=50.0, S_noise=1.003):
    """EDM Heun sampler that records snapshots at specified steps."""
    if snapshot_at_steps is None:
        snapshot_at_steps = list(range(0, num_steps + 1, 32)) + [num_steps]
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    snapshot_steps = set(snapshot_at_steps)
    snapshots = []

    x_next = latents.to(torch.float64) * t_steps[0]
    if 0 in snapshot_steps:
        snapshots.append((_x_to_uint8(x_next), 0, float(t_steps[0])))

    for i, (t_cur, t_next) in tqdm.tqdm(
        list(enumerate(zip(t_steps[:-1], t_steps[1:]))),
        desc="EDM sampling", unit="step",
    ):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        step_num = i + 1
        if step_num in snapshot_steps:
            snapshots.append((_x_to_uint8(x_next), step_num, float(t_next)))

    return snapshots


def build_trajectory_figure(dmd_img_np, snapshots, scale=4):
    """
    Left: DMD 1-step image (scaled up to match the row height).
    Right: single row of EDM intermediate snapshots, all same size.
    """
    n = len(snapshots)
    thumb = RESOLUTION * scale
    gap = 8
    label_h = 16
    header_h = 30

    dmd_size = thumb
    left_w = dmd_size + gap * 2
    grid_w = n * thumb + (n - 1) * gap
    total_w = left_w + grid_w + gap
    total_h = header_h + thumb + label_h + gap

    fig = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(fig)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = title_font

    draw.text((gap, 5), "DMD (1 step)", fill=(0, 0, 0), font=title_font)
    dmd_pil = Image.fromarray(dmd_img_np, "RGB").resize((dmd_size, dmd_size), Image.NEAREST)
    fig.paste(dmd_pil, (gap, header_h))
    draw.text((gap, header_h + thumb + 1), "step 0 → done", fill=(100, 100, 100), font=label_font)

    right_x0 = left_w
    draw.text((right_x0, 5), "EDM (256-step deterministic Heun)", fill=(0, 0, 0), font=title_font)

    for idx, (img_np, step, sigma) in enumerate(snapshots):
        x = right_x0 + idx * (thumb + gap)
        y = header_h
        pil_img = Image.fromarray(img_np, "RGB").resize((thumb, thumb), Image.NEAREST)
        fig.paste(pil_img, (x, y))
        draw.text((x, y + thumb + 1), f"step {step}", fill=(100, 100, 100), font=label_font)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Sample DMD + EDM trajectory visualization")
    parser.add_argument("--dmd_checkpoint", type=str, required=True)
    parser.add_argument("--edm_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sample_output")
    parser.add_argument("--class_index", type=int, default=207,
                        help="ImageNet class index (default: 207 = golden retriever)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edm_steps", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
    one_hot = torch.zeros(1, LABEL_DIM, device=device)
    one_hot[0, args.class_index] = 1.0

    # --- DMD (1-step) ---
    print("Loading DMD model...")
    dmd_gen = create_dmd_generator(args.dmd_checkpoint, device)
    print("Sampling from DMD (1 step)...")
    dmd_img = sample_dmd_one(dmd_gen, noise, one_hot, device)
    dmd_img_np = dmd_img[0].permute(1, 2, 0).cpu().numpy()
    del dmd_gen
    torch.cuda.empty_cache()

    Image.fromarray(dmd_img_np, "RGB").save(os.path.join(args.output_dir, "dmd_sample.png"))
    print("Saved dmd_sample.png")

    # --- EDM (multi-step with trajectory) ---
    print("Loading EDM teacher model...")
    edm_net = load_edm_teacher(args.edm_checkpoint, device)
    print(f"Sampling from EDM ({args.edm_steps}-step Heun, S_churn=0)...")
    snapshot_at = [0, 32, 64, 96, 128, 160, 192, 224, 256]
    snapshots = sample_edm_with_trajectory(
        edm_net, noise, one_hot, device,
        num_steps=args.edm_steps, snapshot_at_steps=snapshot_at,
    )
    edm_final_np = snapshots[-1][0]
    del edm_net
    torch.cuda.empty_cache()

    Image.fromarray(edm_final_np, "RGB").save(os.path.join(args.output_dir, "edm_sample.png"))
    print("Saved edm_sample.png")

    # --- Build trajectory figure ---
    fig = build_trajectory_figure(dmd_img_np, snapshots)
    fig.save(os.path.join(args.output_dir, "side_by_side.png"))

    print(f"\nDone! Saved to {args.output_dir}/")
    print(f"  dmd_sample.png   — DMD 1-step")
    print(f"  edm_sample.png   — EDM {args.edm_steps}-step final")
    print(f"  side_by_side.png — DMD (left) + EDM trajectory (right, {len(snapshots)} snapshots)")
    print(f"  class={args.class_index}, seed={args.seed}")


if __name__ == "__main__":
    main()
