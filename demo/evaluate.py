"""
Evaluate DMD vs EDM comparison results with LPIPS and CLIP scores.

Reads paired images from comparison_output/s_churn_*/dmd_images/ and edm_images/,
computes:
  - LPIPS: perceptual distance between each DMD-EDM pair (lower = more similar)
  - CLIP score: cosine similarity between image and "a photo of a {class}" (higher = better)

Outputs a CSV to comparison_output/evaluation_results.csv

Usage:
    python demo/evaluate.py \
        --comparison_dir ./comparison_output \
        --master_seed 42 \
        --num_images 30
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import clip
import piq
from PIL import Image
from torchvision import transforms

LABEL_DIM = 1000
LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_labels.json")


def load_imagenet_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def load_image_as_tensor(path, size=64):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)
    if tensor.shape[1] != size or tensor.shape[2] != size:
        tensor = transforms.Resize((size, size))(tensor)
    return tensor


def compute_lpips_scores(dmd_tensors, edm_tensors, device):
    """Compute per-pair LPIPS using piq."""
    lpips_fn = piq.LPIPS(replace_pooling=True, reduction="none").to(device)
    scores = []
    for dmd_t, edm_t in zip(dmd_tensors, edm_tensors):
        dmd_t = dmd_t.unsqueeze(0).to(device)
        edm_t = edm_t.unsqueeze(0).to(device)
        score = lpips_fn(dmd_t, edm_t).item()
        scores.append(score)
    return scores


def compute_clip_scores(image_tensors, class_names, clip_model, clip_preprocess, device):
    """Compute per-image CLIP score against 'a photo of a {class}'."""
    scores = []
    prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for i, img_t in enumerate(image_tensors):
        pil_img = transforms.ToPILImage()(img_t)
        img_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = clip_model.encode_image(img_input)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        cos_sim = (img_features @ text_features[i:i+1].T).item()
        scores.append(cos_sim)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate DMD vs EDM with LPIPS and CLIP")
    parser.add_argument("--comparison_dir", type=str, default="./comparison_output")
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imagenet_labels = load_imagenet_labels()
    rng = np.random.RandomState(args.master_seed)
    _ = rng.randint(0, 2**31, size=args.num_images)
    class_indices = rng.randint(0, LABEL_DIM, size=args.num_images).tolist()
    class_names = [imagenet_labels[c] for c in class_indices]

    churn_dirs = sorted(glob.glob(os.path.join(args.comparison_dir, "s_churn_*")))
    if not churn_dirs:
        print(f"No s_churn_* directories found in {args.comparison_dir}")
        return

    churn_values = []
    for d in churn_dirs:
        name = os.path.basename(d)
        val = name.replace("s_churn_", "")
        churn_values.append(val)

    print(f"Found churn values: {churn_values}")
    print(f"Class names (first 5): {class_names[:5]}")
    print(f"Using CLIP model: {args.clip_model}")

    print("\nLoading CLIP model...")
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()

    print("Loading LPIPS model...")
    _ = piq.LPIPS(replace_pooling=True).to(device)
    print("Models loaded.\n")

    results = {}

    for churn_val, churn_dir in zip(churn_values, churn_dirs):
        print(f"=== Evaluating s_churn_{churn_val} ===")

        dmd_dir = os.path.join(churn_dir, "dmd_images")
        edm_dir = os.path.join(churn_dir, "edm_images")

        n_images = len([f for f in os.listdir(dmd_dir) if f.endswith(".png")])
        assert n_images == args.num_images, (
            f"Expected {args.num_images} images, found {n_images} in {dmd_dir}"
        )

        dmd_tensors = []
        edm_tensors = []
        for i in range(n_images):
            dmd_path = os.path.join(dmd_dir, f"dmd_{i:03d}.png")
            edm_path = os.path.join(edm_dir, f"edm_{i:03d}.png")
            dmd_tensors.append(load_image_as_tensor(dmd_path))
            edm_tensors.append(load_image_as_tensor(edm_path))

        print("  Computing LPIPS...")
        lpips_scores = compute_lpips_scores(dmd_tensors, edm_tensors, device)

        print("  Computing CLIP scores for EDM images...")
        edm_clip_scores = compute_clip_scores(edm_tensors, class_names, clip_model, clip_preprocess, device)

        print("  Computing CLIP scores for DMD images...")
        dmd_clip_scores = compute_clip_scores(dmd_tensors, class_names, clip_model, clip_preprocess, device)

        results[f"churn_{churn_val}_LPIPS"] = lpips_scores
        results[f"churn_{churn_val}_EDMCLIP"] = edm_clip_scores
        results[f"churn_{churn_val}_DMDCLIP"] = dmd_clip_scores

        print(f"  LPIPS:    mean={np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
        print(f"  EDM CLIP: mean={np.mean(edm_clip_scores):.4f} ± {np.std(edm_clip_scores):.4f}")
        print(f"  DMD CLIP: mean={np.mean(dmd_clip_scores):.4f} ± {np.std(dmd_clip_scores):.4f}")
        print()

    df = pd.DataFrame(results)
    df.index.name = "image_idx"

    summary = df.mean().to_frame("mean").T
    summary.index = ["MEAN"]
    std_row = df.std().to_frame("std").T
    std_row.index = ["STD"]
    df_full = pd.concat([df, summary, std_row])

    csv_path = os.path.join(args.comparison_dir, "evaluation_results.csv")
    df_full.to_csv(csv_path)
    print(f"Saved results to {csv_path}")

    print("\n=== Summary ===")
    for churn_val in churn_values:
        lpips_col = f"churn_{churn_val}_LPIPS"
        edm_clip_col = f"churn_{churn_val}_EDMCLIP"
        dmd_clip_col = f"churn_{churn_val}_DMDCLIP"
        print(f"  S_churn={churn_val:>2s}:  "
              f"LPIPS={df[lpips_col].mean():.4f}  "
              f"EDM_CLIP={df[edm_clip_col].mean():.4f}  "
              f"DMD_CLIP={df[dmd_clip_col].mean():.4f}")


if __name__ == "__main__":
    main()
