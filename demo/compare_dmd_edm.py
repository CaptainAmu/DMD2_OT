"""
Compare DMD (1-step) vs EDM (multi-step) on ImageNet-64x64.

For each of 50 randomly sampled classes, generates one image pair using the
same noise. The output comparison_grid.png shows one row per sample:
  - Class name label on the far left
  - DMD 1-step result
  - EDM trajectory snapshots at steps 0, 32, 64, ..., 256

Usage:
    python demo/compare_dmd_edm.py \
        --dmd_checkpoint model_checkpoints/pytorch_model.bin \
        --edm_checkpoint model_checkpoints/edm-imagenet-64x64-cond-adm.pkl \
        --S_churn 0
"""

import os
import json
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
SNAPSHOT_STEPS = [0, 32, 64, 96, 128, 160, 192, 224, 256]

LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_labels.json")


def load_imagenet_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


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
def sample_dmd(generator, noise, class_labels, device):
    sigma = torch.ones(noise.shape[0], device=device) * SIGMA_MAX
    images = generator(noise * SIGMA_MAX, sigma, class_labels)
    images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return images


def _x_to_uint8_batch(x):
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def sample_edm_with_trajectory(net, latents, class_labels, device,
                                num_steps=256, snapshot_at_steps=None, rho=7,
                                S_churn=0, S_min=0.05, S_max=50.0, S_noise=1.003):
    """EDM Heun sampler returning snapshots at specified steps. Batch-aware."""
    if snapshot_at_steps is None:
        snapshot_at_steps = SNAPSHOT_STEPS
    snapshot_set = set(snapshot_at_steps)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    snapshots = {}
    x_next = latents.to(torch.float64) * t_steps[0]
    if 0 in snapshot_set:
        snapshots[0] = _x_to_uint8_batch(x_next)

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
        if step_num in snapshot_set:
            snapshots[step_num] = _x_to_uint8_batch(x_next)

    return snapshots


def build_comparison_grid(dmd_images, edm_snapshots_per_row, class_names,
                          snapshot_steps, S_churn, edm_steps, scale=3):
    """
    One row per sample:
      [class name label] [DMD image] [EDM step 0] [EDM step 32] ... [EDM step 256]
    """
    n_rows = len(dmd_images)
    n_edm_cols = len(snapshot_steps)
    thumb = RESOLUTION * scale

    gap = 6
    label_col_w = 160
    header_h = 50
    step_label_h = 18
    row_h = thumb + gap

    total_w = label_col_w + thumb + gap + n_edm_cols * (thumb + gap)
    total_h = header_h + step_label_h + n_rows * row_h + gap

    fig = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(fig)

    try:
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        step_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        header_font = ImageFont.load_default()
        step_font = header_font
        label_font = header_font

    dmd_col_x = label_col_w
    edm_col_x0 = dmd_col_x + thumb + gap

    churn_str = f"{S_churn:g}"
    draw.text((dmd_col_x, 8), "DMD 1-step", fill=(0, 0, 180), font=header_font)
    edm_header = f"EDM {edm_steps}-step Heun, S_churn = {churn_str}"
    draw.text((edm_col_x0, 8), edm_header, fill=(180, 0, 0), font=header_font)

    y_step_labels = header_h
    draw.text((dmd_col_x, y_step_labels), "result", fill=(80, 80, 80), font=step_font)
    for ci, s in enumerate(snapshot_steps):
        x = edm_col_x0 + ci * (thumb + gap)
        draw.text((x, y_step_labels), f"step {s}", fill=(80, 80, 80), font=step_font)

    content_y0 = header_h + step_label_h

    for row_idx in range(n_rows):
        y = content_y0 + row_idx * row_h

        cls_name = class_names[row_idx]
        text_y = y + thumb // 2 - 6
        draw.text((8, text_y), cls_name, fill=(0, 0, 0), font=label_font)

        dmd_pil = Image.fromarray(dmd_images[row_idx], "RGB").resize((thumb, thumb), Image.NEAREST)
        fig.paste(dmd_pil, (dmd_col_x, y))

        edm_row_snaps = edm_snapshots_per_row[row_idx]
        for ci, s in enumerate(snapshot_steps):
            x = edm_col_x0 + ci * (thumb + gap)
            img_np = edm_row_snaps[s]
            pil_img = Image.fromarray(img_np, "RGB").resize((thumb, thumb), Image.NEAREST)
            fig.paste(pil_img, (x, y))

    return fig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare DMD vs EDM on ImageNet-64x64")
    parser.add_argument("--dmd_checkpoint", type=str, required=True)
    parser.add_argument("--edm_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./comparison_output")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--edm_steps", type=int, default=256)
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--S_churn", type=float, default=0.0)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=50.0)
    parser.add_argument("--S_noise", type=float, default=1.003)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    imagenet_labels = load_imagenet_labels()

    rng = np.random.RandomState(args.master_seed)
    seeds = rng.randint(0, 2**31, size=args.num_images).tolist()
    class_indices = rng.randint(0, LABEL_DIM, size=args.num_images).tolist()
    class_names = [imagenet_labels[c] for c in class_indices]

    print(f"Master seed: {args.master_seed}, S_churn: {args.S_churn}")
    print(f"Generating {args.num_images} pairs")

    # ---- DMD (1-step) ----
    print("\n=== Loading DMD model ===")
    dmd_gen = create_dmd_generator(args.dmd_checkpoint, device)

    print("=== Generating DMD images (1-step) ===")
    dmd_images = []
    for i in tqdm.tqdm(range(args.num_images), desc="DMD"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_indices[i]] = 1.0
        img = sample_dmd(dmd_gen, noise, one_hot, device)
        dmd_images.append(img[0].permute(1, 2, 0).cpu().numpy())

    del dmd_gen
    torch.cuda.empty_cache()

    # ---- EDM (multi-step with trajectory, one at a time for per-seed control) ----
    print("\n=== Loading EDM teacher model ===")
    edm_net = load_edm_teacher(args.edm_checkpoint, device)

    n_snapshots = 9
    snapshot_steps = sorted(set(
        [0] + [round(i * args.edm_steps / (n_snapshots - 1)) for i in range(1, n_snapshots)]
    ))
    print(f"=== Generating EDM trajectories ({args.edm_steps}-step, S_churn={args.S_churn}) ===")
    print(f"    Snapshot steps: {snapshot_steps}")
    edm_snapshots_per_row = []
    for i in tqdm.tqdm(range(args.num_images), desc="EDM"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_indices[i]] = 1.0

        snaps_dict = sample_edm_with_trajectory(
            edm_net, noise, one_hot, device,
            num_steps=args.edm_steps, snapshot_at_steps=snapshot_steps,
            S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise,
        )
        row_snaps = {step: imgs[0] for step, imgs in snaps_dict.items()}
        edm_snapshots_per_row.append(row_snaps)

    del edm_net
    torch.cuda.empty_cache()

    # ---- Build grid ----
    print("\n=== Building comparison grid ===")
    fig = build_comparison_grid(
        dmd_images, edm_snapshots_per_row, class_names,
        snapshot_steps, args.S_churn, args.edm_steps,
    )
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    fig.save(grid_path)
    print(f"Saved comparison grid to {grid_path}")

    # ---- Save individual images ----
    dmd_dir = os.path.join(args.output_dir, "dmd_images")
    edm_dir = os.path.join(args.output_dir, "edm_images")
    os.makedirs(dmd_dir, exist_ok=True)
    os.makedirs(edm_dir, exist_ok=True)
    for i in range(args.num_images):
        Image.fromarray(dmd_images[i], "RGB").save(os.path.join(dmd_dir, f"dmd_{i:03d}.png"))
        final_step = snapshot_steps[-1]
        Image.fromarray(edm_snapshots_per_row[i][final_step], "RGB").save(
            os.path.join(edm_dir, f"edm_{i:03d}.png")
        )

    print(f"\nDone!")
    print(f"  {args.num_images} pairs, S_churn={args.S_churn}")
    print(f"  Grid: {grid_path}")
    print(f"  Individual: {dmd_dir}/, {edm_dir}/")


if __name__ == "__main__":
    main()
