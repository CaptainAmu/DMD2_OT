"""
Compare teacher (multi-step) vs student (1-step) MeanFlow models on ImageNet-256.

Starting from the same initial noise points, generates teacher images and student images
using MeanFlow models. Stores outputs in comparison_output/{output_subdir}/.

Usage:
    python demo/compare_teacher_meanflow.py \
        --teacher_checkpoint model_checkpoints/mf_teacher_imagenet256.pkl \
        --student_checkpoint model_checkpoints/mf_student_imagenet256.pkl \
        --output_subdir edm_vs_mf \
        --num_images 50 \
        --teacher_steps 50
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import tqdm
from PIL import Image

# Add Re-MeanFlow for edm2 imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RE_MEANFLOW = os.path.join(os.path.dirname(PROJECT_ROOT), "Re-MeanFlow")
sys.path.insert(0, RE_MEANFLOW)
sys.path.insert(0, PROJECT_ROOT)

import dnnlib

RESOLUTION = 256
LABEL_DIM = 1000
MIN_SIGMA = 1e-8
MAX_SIGMA = 80.0 / 81.0  # MeanFlow t in [0,1], max_sigma ~ 0.9876

LABELS_PATH = os.path.join(SCRIPT_DIR, "imagenet_labels.json")


def load_imagenet_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def build_edm_labels(c: torch.Tensor, num_classes: int = 1000, device=None):
    if device is None:
        device = c.device
    labels = torch.eye(num_classes, device=device)
    zero = torch.zeros((1, num_classes), device=device)
    labels = torch.cat([labels, zero], dim=0)[c].contiguous()
    return labels


def load_meanflow_model(checkpoint_path, device):
    """Load MeanFlow Precond from pkl (EDM2 format with ema)."""
    from edm2.networks_edm2 import Precond

    path = checkpoint_path
    if not os.path.isabs(path) and not path.startswith(("http://", "https://")):
        path = os.path.join(PROJECT_ROOT, path)
    with dnnlib.util.open_url(path) as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "ema" in data:
        net = data["ema"]
    else:
        net = data

    model = Precond(*net.init_args, **net.init_kwargs)
    model.load_state_dict(net.state_dict(), strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def sample_meanflow_one_step(model, noise, class_labels, device):
    """1-step MeanFlow: x = n - model(n, t_max, r=0, c)."""
    t = torch.ones(noise.shape[0], device=device, dtype=noise.dtype) * MAX_SIGMA
    r = torch.zeros(noise.shape[0], device=device, dtype=noise.dtype)
    v = model(noise, t, r, class_labels)
    x = noise - v
    # x is in flow space; convert to image [-1, 1] (at t=0, x_ = x_0)
    out = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return out.permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def sample_meanflow_multi_step(model, noise, class_labels, device, num_steps=50):
    """Multi-step MeanFlow ODE: t from max_sigma to min_sigma."""
    t_steps = torch.linspace(MAX_SIGMA, MIN_SIGMA, num_steps + 1, device=device, dtype=noise.dtype)
    z = noise.clone()
    for i in range(num_steps):
        t = t_steps[i]
        dt = t_steps[i + 1] - t
        t_tensor = torch.full((z.size(0),), t, device=device, dtype=z.dtype)
        velocity = model(z, t_tensor, t_tensor, class_labels)
        z = z + dt * velocity
    x = z
    out = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return out.permute(0, 2, 3, 1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compare teacher vs student MeanFlow on ImageNet-256"
    )
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./comparison_output")
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="edm256_vs_mf",
        help="Subdir under output_dir, e.g. edm256_vs_mf",
    )
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--teacher_steps", type=int, default=50)
    parser.add_argument("--master_seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(args.output_dir, args.output_subdir)
    teacher_dir = os.path.join(out_root, "edm256_images")
    student_dir = os.path.join(out_root, "mf_images")
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)

    imagenet_labels = load_imagenet_labels()

    rng = np.random.RandomState(args.master_seed)
    seeds = rng.randint(0, 2**31, size=args.num_images).tolist()
    class_indices = rng.randint(0, LABEL_DIM, size=args.num_images).tolist()
    class_names = [imagenet_labels[c] for c in class_indices]

    print(f"Master seed: {args.master_seed}")
    print(f"Generating {args.num_images} pairs (teacher {args.teacher_steps}-step, student 1-step)")
    print(f"Output: {out_root}")

    # ---- Load teacher ----
    print("\n=== Loading teacher model ===")
    teacher = load_meanflow_model(args.teacher_checkpoint, device)

    # ---- Load student ----
    print("=== Loading student model ===")
    student = load_meanflow_model(args.student_checkpoint, device)

    # ---- Generate (same seeds for both) ----
    print("\n=== Generating images ===")
    for i in tqdm.tqdm(range(args.num_images), desc="Teacher+Student"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = build_edm_labels(
            torch.tensor([class_indices[i]], device=device), LABEL_DIM, device
        )

        # Teacher: multi-step
        img_teacher = sample_meanflow_multi_step(
            teacher, noise, one_hot, device, num_steps=args.teacher_steps
        )[0]

        # Student: 1-step (same noise)
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        img_student = sample_meanflow_one_step(student, noise, one_hot, device)[0]

        fname = f"{i:03d}_{class_indices[i]}_{class_names[i].replace(' ', '_')}"
        Image.fromarray(img_teacher, "RGB").save(
            os.path.join(teacher_dir, f"{fname}.png")
        )
        Image.fromarray(img_student, "RGB").save(
            os.path.join(student_dir, f"{fname}.png")
        )

    print(f"\nDone!")
    print(f"  {args.num_images} pairs")
    print(f"  EDM256 ({args.teacher_steps}-step): {teacher_dir}/")
    print(f"  MF (1-step): {student_dir}/")


if __name__ == "__main__":
    main()
