"""
Generate EDM 1-step, EDM N-step, and Consistency Model images with the same noise seeds.

Stores outputs in comparison_output/edm_vs_cm/{class_index}_{classname}/
  - edm_1_images/edm1_0000.png, ...
  - edm_N_images/edmN_0000.png, ...
  - cm_images/cm_0000.png, ...

Usage:
    python demo/generate_edm_vs_cm.py \
        --edm_checkpoint model_checkpoints/edm-imagenet-64x64-cond-adm.pkl \
        --class_index 248 \
        --num_seeds 500 \
        --master_seed 42
"""

import os
import json
import argparse
import pickle
import numpy as np
import torch
import tqdm
from PIL import Image

import dnnlib
from diffusers import ConsistencyModelPipeline

RESOLUTION = 64
LABEL_DIM = 1000
SIGMA_MAX = 80.0
SIGMA_MIN = 0.002

LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_labels.json")


def load_imagenet_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def load_edm_teacher(pickle_path, device):
    with dnnlib.util.open_url(pickle_path) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    return net


def _x_to_uint8_batch(x):
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def sample_edm_1step(net, noise, class_labels, device):
    """EDM 1-step: from noise directly to E[x0|x_t]. Returns uint8 [B, H, W, 3] numpy."""
    sigma = torch.ones(noise.shape[0], device=device, dtype=noise.dtype) * SIGMA_MAX
    x_noisy = noise * SIGMA_MAX
    denoised = net(x_noisy, sigma, class_labels)
    out = ((denoised + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return out.permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def sample_edm_N(net, latents, class_labels, device, num_steps=256, S_churn=0,
               S_min=0.05, S_max=50.0, S_noise=1.003, rho=7):
    """EDM Heun sampler, returns final denoised images as uint8 numpy [B, H, W, 3]."""
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
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

    return _x_to_uint8_batch(x_next)


def main():
    parser = argparse.ArgumentParser(
        description="Generate EDM vs Consistency Model images with shared noise seeds."
    )
    parser.add_argument("--edm_checkpoint", type=str, required=True)
    parser.add_argument(
        "--class_index", type=int, default=248,
        help="ImageNet class index (e.g. 248 = husky).",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=500,
        help="Number of images to generate per model.",
    )
    parser.add_argument(
        "--master_seed", type=int, default=42,
        help="Master RNG seed for per-sample noise seeds.",
    )
    parser.add_argument(
        "--edm_steps", type=int, default=256,
        help="Number of EDM Heun steps.",
    )
    parser.add_argument(
        "--S_churn", type=float, default=0.0,
        help="EDM S_churn parameter.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./comparison_output/edm_vs_cm",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    imagenet_labels = load_imagenet_labels()
    if 0 <= args.class_index < len(imagenet_labels):
        class_name = imagenet_labels[args.class_index]
    else:
        class_name = f"class_{args.class_index}"

    safe_class_name = str(class_name).replace(" ", "_").replace("/", "-")
    class_dir_name = f"{args.class_index}_{safe_class_name}"
    class_dir = os.path.join(args.output_dir, class_dir_name)
    edm_1_dir = os.path.join(class_dir, "edm_1_images")
    edm_N_dir = os.path.join(class_dir, "edm_N_images")
    cm_dir = os.path.join(class_dir, "cm_images")
    os.makedirs(edm_1_dir, exist_ok=True)
    os.makedirs(edm_N_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    rng = np.random.RandomState(args.master_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.num_seeds)

    print(f"Class {args.class_index}: {class_name}")
    print(f"Generating {args.num_seeds} images per model with shared noise seeds")
    print(f"Output: {class_dir}")

    # Load EDM teacher
    print("Loading EDM teacher...")
    edm_net = load_edm_teacher(args.edm_checkpoint, device)

    # Load Consistency Model pipeline
    print("Loading ConsistencyModelPipeline...")
    cm_pipe = ConsistencyModelPipeline.from_pretrained(
        "openai/diffusers-cd_imagenet64_l2",
        torch_dtype=torch.float16,
    )
    cm_pipe.to(device)

    one_hot = torch.zeros(1, LABEL_DIM, device=device)
    one_hot[0, args.class_index] = 1.0

    for i in tqdm.tqdm(range(args.num_seeds), desc="Generating", unit="triplet"):
        gen = torch.Generator(device=device).manual_seed(int(seeds[i]))
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device, generator=gen)

        # EDM 1-step (same noise)
        edm1_out = sample_edm_1step(edm_net, noise, one_hot, device)
        Image.fromarray(edm1_out[0], mode="RGB").save(
            os.path.join(edm_1_dir, f"edm1_{i:04d}.png")
        )

        # EDM N-step (same noise)
        edmN_out = sample_edm_N(
            edm_net, noise, one_hot, device,
            num_steps=args.edm_steps,
            S_churn=args.S_churn,
        )
        Image.fromarray(edmN_out[0], mode="RGB").save(
            os.path.join(edm_N_dir, f"edmN_{i:04d}.png")
        )

        # Consistency Model (same noise via latents)
        gen_cm = torch.Generator(device=device).manual_seed(int(seeds[i]))
        out = cm_pipe(
            batch_size=1,
            class_labels=args.class_index,
            num_inference_steps=1,
            latents=noise.to(torch.float16),
            generator=gen_cm,
        )
        img = out.images[0]
        img.save(os.path.join(cm_dir, f"cm_{i:04d}.png"))

    print(f"Done. Saved to {class_dir}")
    print(f"  EDM 1-step: {edm_1_dir}")
    print(f"  EDM N-step: {edm_N_dir}")
    print(f"  CM:         {cm_dir}")


if __name__ == "__main__":
    main()
