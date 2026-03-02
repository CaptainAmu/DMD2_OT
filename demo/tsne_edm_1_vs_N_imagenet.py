import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from demo.compare_dmd_edm import (
    load_edm_teacher,
    sample_edm_with_trajectory,
    load_imagenet_labels,
    RESOLUTION,
    LABEL_DIM,
)

SIGMA_MAX = 80.0


@torch.no_grad()
def sample_edm_1step(net, noise, class_labels, device):
    """
    EDM 1-step: from noise directly to E[x0|x_t].
    x_noisy = noise * sigma_max, denoised = net(x_noisy, sigma_max).
    Returns uint8 [B, H, W, 3] numpy (same format as sample_edm_with_trajectory).
    """
    sigma = torch.ones(noise.shape[0], device=device, dtype=noise.dtype) * SIGMA_MAX
    x_noisy = noise * SIGMA_MAX
    denoised = net(x_noisy, sigma, class_labels)
    # EDM output in [-1, 1], convert to uint8 like sample_dmd
    out = ((denoised + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return out.permute(0, 2, 3, 1).cpu().numpy()


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
        for bi in range(n_batches):
            start = bi * batch_size
            batch = images_chw[start : start + batch_size].to(device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            f = feature_extractor(batch)
            f = f.view(f.size(0), -1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


@torch.no_grad()
def generate_edm_1_vs_N(
    edm_checkpoint: str,
    class_index: int,
    num_seeds: int,
    master_seed: int,
    steps_N: int,
    device: torch.device,
    edm1_dir: str = None,
    edmN_dir: str = None,
):
    """
    For a single ImageNet class, generate pairs of EDM images:
      - x_1: EDM 1-step sampling (num_steps=1, one big step from noise to E[x0|x_t])
      - x_N: EDM N-step Heun sampling (full trajectory)
    Both use the same noise seed and class label.
    Returns:
      imgs_1_chw, imgs_N_chw: [num_seeds, 3, H, W] floats in [0,1]
    """
    from PIL import Image

    print("Loading EDM teacher...")
    edm_net = load_edm_teacher(edm_checkpoint, device)

    rng = np.random.RandomState(master_seed)
    seeds = rng.randint(0, 2**31, size=num_seeds).tolist()

    imgs_1 = []
    imgs_N = []

    print(f"Generating EDM 1-step vs {steps_N}-step (same noise), {num_seeds} seeds...")
    for i in range(num_seeds):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{num_seeds}]")

        torch.manual_seed(seeds[i])
        noise = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)

        one_hot = torch.zeros(1, LABEL_DIM, device=device)
        one_hot[0, class_index] = 1.0

        # 1-step: one big step from noise to E[x0|x_t] (direct denoise at sigma_max)
        img1_np = sample_edm_1step(edm_net, noise, one_hot, device)[0]  # [H, W, 3], uint8

        # N-step: full Heun trajectory
        snaps_N = sample_edm_with_trajectory(
            edm_net, noise, one_hot, device,
            num_steps=steps_N,
            snapshot_at_steps=[steps_N],
            S_churn=0.0,
        )
        imgN_np = snaps_N[steps_N][0]

        # Save immediately when dirs are provided
        if edm1_dir is not None:
            Image.fromarray(img1_np, mode="RGB").save(os.path.join(edm1_dir, f"edm1_{i:04d}.png"))
        if edmN_dir is not None:
            Image.fromarray(imgN_np, mode="RGB").save(os.path.join(edmN_dir, f"edmN_{i:04d}.png"))

        img1 = torch.from_numpy(img1_np).permute(2, 0, 1).to(torch.float32) / 255.0
        imgN = torch.from_numpy(imgN_np).permute(2, 0, 1).to(torch.float32) / 255.0

        imgs_1.append(img1)
        imgs_N.append(imgN)

    del edm_net
    torch.cuda.empty_cache()

    imgs_1_chw = torch.stack(imgs_1, dim=0)
    imgs_N_chw = torch.stack(imgs_N, dim=0)
    return imgs_1_chw, imgs_N_chw


def main():
    parser = argparse.ArgumentParser(
        description="Compare EDM 1-step vs N-step (same noise, same class) via t-SNE distances."
    )
    parser.add_argument("--edm_checkpoint", type=str, required=True)
    parser.add_argument(
        "--class_index",
        type=int,
        default=207,
        help="ImageNet class index (e.g. 207 = golden retriever).",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=100,
        help="Number of shared noise seeds (pairs) to sample.",
    )
    parser.add_argument(
        "--master_seed",
        type=int,
        default=42,
        help="Master RNG seed used to generate per-sample noise seeds.",
    )
    parser.add_argument(
        "--steps_N",
        type=int,
        default=256,
        help="Number of Heun steps for the multi-step EDM trajectory.",
    )
    parser.add_argument(
        "--batch_size_features",
        type=int,
        default=64,
        help="Batch size for ResNet feature extraction.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_output/tsne_analysis_1vN",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    imagenet_labels = load_imagenet_labels()
    if 0 <= args.class_index < len(imagenet_labels):
        class_name = imagenet_labels[args.class_index]
    else:
        class_name = f"class_{args.class_index}"

    class_dir_name = f"{args.class_index:03d}_{str(class_name).replace(' ', '_').replace('/', '-')}"
    class_dir = os.path.join(args.output_dir, class_dir_name)
    os.makedirs(class_dir, exist_ok=True)

    edm1_dir = os.path.join(class_dir, "edm_1step_images")
    edmN_dir = os.path.join(class_dir, "edm_Nstep_images")
    os.makedirs(edm1_dir, exist_ok=True)
    os.makedirs(edmN_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Class {args.class_index}: {class_name}")
    print(f"EDM 1-step vs {args.steps_N}-step, seeds={args.num_seeds}")

    # 1) Generate 1-step vs N-step EDM image pairs (save each image immediately)
    imgs_1_chw, imgs_N_chw = generate_edm_1_vs_N(
        args.edm_checkpoint,
        args.class_index,
        args.num_seeds,
        args.master_seed,
        args.steps_N,
        device,
        edm1_dir=edm1_dir,
        edmN_dir=edmN_dir,
    )

    # 2) ResNet-50 features
    print("Building ResNet-50 feature extractor...")
    feat_extractor = build_resnet_feature_extractor(device)

    print("Extracting features for 1-step EDM images...")
    feats_1 = compute_resnet_features_batched(
        imgs_1_chw, feat_extractor, device, batch_size=args.batch_size_features, desc="EDM 1-step"
    )
    print("Extracting features for N-step EDM images...")
    feats_N = compute_resnet_features_batched(
        imgs_N_chw, feat_extractor, device, batch_size=args.batch_size_features, desc=f"EDM {args.steps_N}-step"
    )

    np.savez_compressed(os.path.join(class_dir, "edm1_feats.npz"), features=feats_1.astype(np.float32))
    np.savez_compressed(os.path.join(class_dir, "edmN_feats.npz"), features=feats_N.astype(np.float32))

    # 3) t-SNE on concatenated features
    print("Running t-SNE on concatenated features...")
    all_feats = np.concatenate([feats_1, feats_N], axis=0)
    n1 = feats_1.shape[0]
    nN = feats_N.shape[0]

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (n1 + nN) // 10)),
        learning_rate=200,
        init="pca",
        random_state=0,
        verbose=1,
    )
    all_2d = tsne.fit_transform(all_feats)

    tsne1 = all_2d[:n1]
    tsneN = all_2d[n1 : n1 + nN]

    # 4) Distance distribution between 1-step and N-step points (paired by seed)
    min_pairs = min(n1, nN)
    diffs = tsne1[:min_pairs] - tsneN[:min_pairs]
    dists = np.linalg.norm(diffs, axis=1)

    print(f"Average 1-step vs {args.steps_N}-step t-SNE distance (N={min_pairs}): "
          f"{dists.mean():.4f} ± {dists.std():.4f}")

    # 5) Plot histogram of distances
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, dists.max() * 1.01, 30)
    plt.hist(dists, bins=bins, color="red", alpha=0.75, edgecolor="black")
    plt.xlabel("EDM 1-step vs N-step Euclidean distance (t-SNE 2D)")
    plt.ylabel("Count")
    plt.title(f"EDM 1-step vs {args.steps_N}-step distance distribution\n"
              f"class {args.class_index}: {class_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    hist_path = os.path.join(class_dir, f"edm_1_vs_{args.steps_N}_dist_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()

