import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from sklearn.manifold import TSNE

from demo.compare_dmd_edm import (
    create_dmd_generator,
    load_edm_teacher,
    sample_dmd,
    sample_edm_with_trajectory,
    load_imagenet_labels,
    RESOLUTION,
    LABEL_DIM,
)

SIGMA_MAX = 80.0


@torch.no_grad()
def sample_edm_1step(net, noise, class_labels, device):
    """EDM 1-step: from noise directly to E[x0|x_t]. Returns uint8 [B, H, W, 3] numpy."""
    sigma = torch.ones(noise.shape[0], device=device, dtype=noise.dtype) * SIGMA_MAX
    x_noisy = noise * SIGMA_MAX
    denoised = net(x_noisy, sigma, class_labels)
    out = ((denoised + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return out.permute(0, 2, 3, 1).cpu().numpy()
from main.data.lmdb_dataset import LMDBDataset
from main.utils import retrieve_row_from_lmdb, get_array_shape_from_lmdb
import lmdb


def stage_print(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_resnet_feature_extractor(device: torch.device):
    """Return ResNet-50 backbone (GlobalAvgPool output, before FC)."""
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)

    feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.to(device).eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)
    return feature_extractor


def compute_resnet_features_batched(
    images_chw: torch.Tensor,
    feature_extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "ResNet features",
):
    """
    images_chw: [N, 3, H, W], values in [0, 1].
    Returns: numpy array [N, feat_dim].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    n_batches = (images_chw.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        for idx in tqdm.tqdm(range(n_batches), desc=desc, unit="batch"):
            start = idx * batch_size
            batch = images_chw[start : start + batch_size].to(device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            f = feature_extractor(batch)
            f = f.view(f.size(0), -1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


def collect_background_features_for_class(
    lmdb_path: str,
    class_index: int,
    feature_extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
):
    """
    Two-stage pipeline:
      1) Fast label-only scan over LMDB to find all indices with label == class_index.
      2) Load only those indices through LMDBDataset and extract ResNet features.
    This avoids running a full DataLoader pass over all images.
    """
    # ---- Stage 2a: label-only scan to find indices for this class ----
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    label_shape = get_array_shape_from_lmdb(env, "labels")
    n_total = label_shape[0]

    stage_print(f"  Label scan over {n_total} entries to find class {class_index} indices")
    cls_indices = []
    for idx in tqdm.tqdm(range(n_total), desc="Label scan", unit="lbl"):
        lbl = retrieve_row_from_lmdb(env, "labels", np.int64, label_shape[1:], idx)
        if int(lbl) == int(class_index):
            cls_indices.append(idx)
    env.close()

    total_in_class = len(cls_indices)
    if total_in_class == 0:
        stage_print(f"  No training images found for class {class_index}")
        return np.zeros((0, 2048), dtype=np.float32), 0

    stage_print(f"  Found {total_in_class} training images for class {class_index}")

    # ---- Stage 2b: load only those indices and run ResNet ----
    dataset = LMDBDataset(lmdb_path)
    subset = Subset(dataset, cls_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    bg_features = []
    n_batches = (len(subset) + batch_size - 1) // batch_size

    for batch in tqdm.tqdm(loader, total=n_batches, desc="Background scan", unit="batch"):
        images = batch["images"]       # [B, 3, H, W]
        feats = compute_resnet_features_batched(
            images, feature_extractor, device, desc="  BG ResNet"
        )
        bg_features.append(feats)

    bg_features = np.concatenate(bg_features, axis=0)
    return bg_features, total_in_class


def generate_edm_dmd_samples(
    dmd_checkpoint: str,
    edm_checkpoint: str,
    class_index: int,
    num_seeds: int,
    master_seed: int,
    edm_steps: int,
    S_churn: float,
    device: torch.device,
    dmd_img_dir: str = None,
    edm_N_img_dir: str = None,
    edm_1_img_dir: str = None,
):
    """
    Generate (num_seeds) samples for a single class:
      - DMD (student 1-step)
      - EDM N-step Heun (churn 0)
      - EDM 1-step (direct denoise at sigma_max)
    All use shared initial noise seeds. Saves each image immediately.
    Returns:
      dmd_images_chw, edm_N_images_chw, edm_1_images_chw: [N, 3, H, W] in [0,1]
    """
    stage_print("Loading DMD student...")
    dmd_gen = create_dmd_generator(dmd_checkpoint, device)

    stage_print("Generating DMD images (1-step)...")
    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 2**31, size=num_seeds).tolist()

    dmd_imgs = []
    for i in tqdm.tqdm(range(num_seeds), desc="DMD generation", unit="img"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_index] = 1.0

        dmd_img_uint8 = sample_dmd(dmd_gen, noise, one_hot, device)
        dmd_img = dmd_img_uint8[0].to(torch.float32) / 255.0
        dmd_imgs.append(dmd_img)

        if dmd_img_dir is not None:
            img_np = (dmd_img.clamp(0, 1) * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(img_np, mode="RGB").save(os.path.join(dmd_img_dir, f"dmd_{i:04d}.png"))

    del dmd_gen
    torch.cuda.empty_cache()

    stage_print("Loading EDM teacher...")
    edm_net = load_edm_teacher(edm_checkpoint, device)

    stage_print("Generating EDM 1-step images...")
    edm_1_imgs = []
    for i in tqdm.tqdm(range(num_seeds), desc="EDM 1-step", unit="img"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_index] = 1.0

        img1_np = sample_edm_1step(edm_net, noise, one_hot, device)[0]
        if edm_1_img_dir is not None:
            Image.fromarray(img1_np, mode="RGB").save(os.path.join(edm_1_img_dir, f"edm1_{i:04d}.png"))
        edm_1_imgs.append(torch.from_numpy(img1_np).permute(2, 0, 1).to(torch.float32) / 255.0)

    stage_print(f"Generating EDM {edm_steps}-step Heun images (S_churn={S_churn})...")
    edm_N_imgs = []
    for i in tqdm.tqdm(range(num_seeds), desc="EDM N-step", unit="img"):
        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_index] = 1.0

        snaps = sample_edm_with_trajectory(
            edm_net, noise, one_hot, device,
            num_steps=edm_steps,
            snapshot_at_steps=[edm_steps],
            S_churn=S_churn,
        )
        final_np = snaps[edm_steps][0]
        if edm_N_img_dir is not None:
            Image.fromarray(final_np.astype(np.uint8), mode="RGB").save(
                os.path.join(edm_N_img_dir, f"edmN_{i:04d}.png")
            )
        edm_N_imgs.append(torch.from_numpy(final_np).permute(2, 0, 1).to(torch.float32) / 255.0)

    del edm_net
    torch.cuda.empty_cache()

    dmd_images_chw = torch.stack(dmd_imgs, dim=0)
    edm_N_images_chw = torch.stack(edm_N_imgs, dim=0)
    edm_1_images_chw = torch.stack(edm_1_imgs, dim=0)
    return dmd_images_chw, edm_N_images_chw, edm_1_images_chw


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE comparison of EDM (teacher) vs DMD (student) on ImageNet-64x64."
    )
    parser.add_argument("--dmd_checkpoint", type=str, required=True)
    parser.add_argument("--edm_checkpoint", type=str, required=True)
    parser.add_argument(
        "--real_image_path", type=str, required=True,
        help="Path to imagenet-64x64_lmdb directory.",
    )
    parser.add_argument(
        "--class_index", type=int, default=207,
        help="ImageNet class index to analyze (default: 207 = golden retriever).",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1000,
        help="Number of shared noise seeds for EDM/DMD (default: 1000).",
    )
    parser.add_argument(
        "--master_seed", type=int, default=42,
        help="Master RNG seed used to generate per-sample noise seeds.",
    )
    parser.add_argument(
        "--edm_steps", type=int, default=256,
        help="Number of EDM Heun steps (default: 256).",
    )
    parser.add_argument(
        "--S_churn", type=float, default=0.0,
        help="EDM S_churn parameter (default: 0).",
    )
    parser.add_argument(
        "--batch_size_features", type=int, default=64,
        help="Batch size for ResNet feature extraction.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./comparison_output/tsne_analysis",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    imagenet_labels = load_imagenet_labels()
    if 0 <= args.class_index < len(imagenet_labels):
        class_name = imagenet_labels[args.class_index]
    else:
        class_name = f"class_{args.class_index}"

    # Per-class folder: e.g. 207_Golden_Retriever
    safe_class_name = str(class_name).replace(" ", "_").replace("/", "-")
    class_dir_name = f"{args.class_index:03d}_{safe_class_name}"
    class_dir = os.path.join(args.output_dir, class_dir_name)
    os.makedirs(class_dir, exist_ok=True)

    dmd_img_dir = os.path.join(class_dir, "dmd_images")
    edm_N_img_dir = os.path.join(class_dir, "edm_N_images")
    edm_1_img_dir = os.path.join(class_dir, "edm_1_images")
    os.makedirs(dmd_img_dir, exist_ok=True)
    os.makedirs(edm_N_img_dir, exist_ok=True)
    os.makedirs(edm_1_img_dir, exist_ok=True)

    stage_print(f"Device: {device}")
    stage_print(f"Class {args.class_index}: {class_name}")
    stage_print(f"EDM {args.edm_steps}-step, S_churn={args.S_churn}, seeds={args.num_seeds}")

    # ── Stage 1: ResNet backbone ──
    stage_print("STAGE 1/5 ▸ Building ResNet-50 feature extractor")
    feature_extractor = build_resnet_feature_extractor(device)

    # ── Stage 2: Background scan ──
    stage_print("STAGE 2/5 ▸ Background scan – collecting training features for target class")
    bg_features, num_bg = collect_background_features_for_class(
        args.real_image_path,
        args.class_index,
        feature_extractor,
        device,
        batch_size=256,
        num_workers=0,
    )
    stage_print(f"Background scan done – found {num_bg} training images for class {args.class_index}")

    # Save background (training) features for this class.
    train_feat_path = os.path.join(class_dir, "extrct_training_feats.npz")
    np.savez_compressed(train_feat_path, features=bg_features.astype(np.float32))
    stage_print(f"Saved training features to {train_feat_path}")

    # ── Stage 3: DMD + EDM generation ──
    stage_print("STAGE 3/5 ▸ DMD + EDM sample generation")
    dmd_images_chw, edm_N_images_chw, edm_1_images_chw = generate_edm_dmd_samples(
        args.dmd_checkpoint,
        args.edm_checkpoint,
        args.class_index,
        args.num_seeds,
        args.master_seed,
        args.edm_steps,
        args.S_churn,
        device,
        dmd_img_dir=dmd_img_dir,
        edm_N_img_dir=edm_N_img_dir,
        edm_1_img_dir=edm_1_img_dir,
    )

    # ── Stage 4: ResNet feature extraction for generated samples ──
    stage_print("STAGE 4/5 ▸ ResNet feature extraction for DMD + EDM N + EDM 1 samples")
    dmd_features = compute_resnet_features_batched(
        dmd_images_chw, feature_extractor, device,
        batch_size=args.batch_size_features, desc="DMD ResNet features",
    )
    edm_N_features = compute_resnet_features_batched(
        edm_N_images_chw, feature_extractor, device,
        batch_size=args.batch_size_features, desc="EDM N-step ResNet features",
    )
    edm_1_features = compute_resnet_features_batched(
        edm_1_images_chw, feature_extractor, device,
        batch_size=args.batch_size_features, desc="EDM 1-step ResNet features",
    )

    # Save DMD / EDM N / EDM 1 features.
    dmd_feat_path = os.path.join(class_dir, "extrct_dmd_feats.npz")
    edm_N_feat_path = os.path.join(class_dir, "extrct_edm_N_feats.npz")
    edm_1_feat_path = os.path.join(class_dir, "extrct_edm_1_feats.npz")
    np.savez_compressed(dmd_feat_path, features=dmd_features.astype(np.float32))
    np.savez_compressed(edm_N_feat_path, features=edm_N_features.astype(np.float32))
    np.savez_compressed(edm_1_feat_path, features=edm_1_features.astype(np.float32))
    stage_print(f"Saved DMD features to {dmd_feat_path}")
    stage_print(f"Saved EDM N-step features to {edm_N_feat_path}")
    stage_print(f"Saved EDM 1-step features to {edm_1_feat_path}")

    # ── Stage 5: t-SNE + visualization ──
    stage_print("STAGE 5/5 ▸ t-SNE computation + visualization")
    all_features = np.concatenate([bg_features, edm_N_features, edm_1_features, dmd_features], axis=0)
    n_bg = bg_features.shape[0]
    n_edm_N = edm_N_features.shape[0]
    n_edm_1 = edm_1_features.shape[0]
    n_dmd = dmd_features.shape[0]
    stage_print(f"  t-SNE input: {n_bg} bg + {n_edm_N} EDM_N + {n_edm_1} EDM_1 + {n_dmd} DMD = {all_features.shape[0]} points")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=0,
        verbose=1,
    )
    all_tsne = tsne.fit_transform(all_features)

    bg_tsne = all_tsne[:n_bg]
    edm_N_tsne = all_tsne[n_bg : n_bg + n_edm_N]
    edm_1_tsne = all_tsne[n_bg + n_edm_N : n_bg + n_edm_N + n_edm_1]
    dmd_tsne = all_tsne[n_bg + n_edm_N + n_edm_1 :]

    min_pairs = min(n_edm_N, n_edm_1, n_dmd)
    diffs_edmN_dmd = edm_N_tsne[:min_pairs] - dmd_tsne[:min_pairs]
    diffs_edm1_dmd = edm_1_tsne[:min_pairs] - dmd_tsne[:min_pairs]
    dist_edmN_dmd = np.linalg.norm(diffs_edmN_dmd, axis=1).mean()
    dist_edm1_dmd = np.linalg.norm(diffs_edm1_dmd, axis=1).mean()
    stage_print(f"Average EDM_N–DMD distance (N={min_pairs}): {dist_edmN_dmd:.4f}")
    stage_print(f"Average EDM_1–DMD distance (N={min_pairs}): {dist_edm1_dmd:.4f}")

    # Alpha from EDM_N t-SNE x (same for all three methods, same seed).
    edm_x = edm_N_tsne[:min_pairs, 0].astype(np.float64)
    x_min, x_max = edm_x.min(), edm_x.max()
    x_norm = (edm_x - x_min) / (x_max - x_min) if x_max > x_min else np.ones(min_pairs, dtype=np.float64) * 0.5
    alphas = (0.1 + 0.9 * x_norm).astype(np.float32)

    edm_N_colors = np.zeros((min_pairs, 4), dtype=np.float32)
    edm_N_colors[:, 2] = 1.0
    edm_N_colors[:, 3] = alphas

    edm_1_colors = np.zeros((min_pairs, 4), dtype=np.float32)
    edm_1_colors[:, 1] = 0.6
    edm_1_colors[:, 2] = 0.6
    edm_1_colors[:, 3] = alphas

    dmd_colors = np.zeros((min_pairs, 4), dtype=np.float32)
    dmd_colors[:, 0] = 1.0
    dmd_colors[:, 3] = alphas

    stage_print("Building figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for ax in axes:
        if n_bg > 0:
            ax.scatter(bg_tsne[:, 0], bg_tsne[:, 1], s=5, c="lightgray", alpha=0.25, label="Train (class)")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].scatter(
        edm_N_tsne[:min_pairs, 0], edm_N_tsne[:min_pairs, 1],
        s=10, color=edm_N_colors, edgecolors="none", label="EDM N-step",
    )
    axes[0].set_title("EDM N-step")

    axes[1].scatter(
        edm_1_tsne[:min_pairs, 0], edm_1_tsne[:min_pairs, 1],
        s=10, color=edm_1_colors, edgecolors="none", label="EDM 1-step",
    )
    axes[1].set_title("EDM 1-step")

    axes[2].scatter(
        dmd_tsne[:min_pairs, 0], dmd_tsne[:min_pairs, 1],
        s=10, color=dmd_colors, edgecolors="none", label="DMD samples",
    )
    axes[2].set_title("DMD (student)")

    fig.suptitle(
        f"EDM vs DMD t-SNE – class {args.class_index}: {class_name}\n"
        f"EDM_N–DMD: {dist_edmN_dmd:.4f}  |  EDM_1–DMD: {dist_edm1_dmd:.4f}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(class_dir, "t_SNE.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    stage_print(f"Saved t-SNE figure to {out_path}")
    stage_print("All done.")


if __name__ == "__main__":
    main()
