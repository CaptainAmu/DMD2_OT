"""
Two-step DMD-style distillation on CIFAR-10 (single-GPU).

Teacher: official EDM CIFAR-10 model (`edm-cifar10-32x32-cond-vp.pkl`) — used only in generator step.
Student:
  - 1-step generator G_theta(z, y)  (EDMPrecond + SongUNet)
  - score/denoiser S_phi(x_noisy, sigma, y) (same arch as generator)

Training loop:
  - Generator step: ∇_θ D_KL ≃ E[ w_t α_t (s_fake - s_real) dG/dθ ]; update theta, fix phi.
  - Score step:     learn student score (denoise x_t → G_theta(z)); no teacher; update phi, fix theta.
"""

import os
import sys
import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import dnnlib
import wandb
from tqdm import tqdm
from PIL import Image

from third_party.edm.training.networks import EDMPrecond, SongUNet


SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
IMG_RES = 32
IMG_CH = 3
NUM_CLASSES = 10

# EDM teacher uses sigma-only formulation (no alpha_bar / DDPM schedule), so α_t = 1.
ALPHA_T = 1.0


def get_cifar10_edm_config():
    """CIFAR-10 EDM/SongUNet config (same as third_party.edm.generate)."""
    return dict(
        augment_dim=9,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.0,
        label_dropout=0,
        embedding_type="positional",
        channel_mult_noise=1,
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1],
    )


def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> torch.Tensor:
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def build_edmprecond_cifar(device: torch.device) -> EDMPrecond:
    cfg = get_cifar10_edm_config()
    model = EDMPrecond(
        img_resolution=IMG_RES,
        img_channels=IMG_CH,
        label_dim=NUM_CLASSES,
        use_fp16=False,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        sigma_data=0.5,
        model_type="SongUNet",
        **cfg,
    )
    assert isinstance(model.model, SongUNet)
    return model.to(device)


def load_teacher(teacher_path: str, device: torch.device) -> nn.Module:
    with dnnlib.util.open_url(teacher_path, verbose=True) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def one_step_generate(
    generator: EDMPrecond,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    sigma: float = SIGMA_MAX,
) -> torch.Tensor:
    device = latents.device
    b = latents.shape[0]
    sigma_t = torch.ones(b, device=device) * sigma
    return generator(latents * sigma, sigma_t, class_labels)


def prepare_class_labels(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    class_indices = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    one_hot = torch.zeros(batch_size, NUM_CLASSES, device=device)
    one_hot.scatter_(1, class_indices.unsqueeze(1), 1.0)
    return class_indices, one_hot


def dm_loss_for_score(
    latents_detached: torch.Tensor,
    labels_one_hot: torch.Tensor,
    score_net: EDMPrecond,
    karras_sigmas: torch.Tensor,
    min_step: int,
    max_step: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Score step: learn the student score of the process from pure noise to G_θ(z).
    x0 = G_θ(z) (detached), x_t ~ q_t(x_t|x0), train S_φ to predict x0 from x_t.
    Loss = E[ ||S_φ(x_t, σ_t, y) - x0||^2 ]. No teacher.
    """
    device = latents_detached.device
    b = latents_detached.shape[0]
    x0 = latents_detached

    with torch.no_grad():
        timesteps = torch.randint(
            min_step,
            min(max_step + 1, karras_sigmas.shape[0]),
            (b, 1, 1, 1),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        sigma_t = karras_sigmas[timesteps]
        noisy = x0 + sigma_t * noise

    pred = score_net(noisy, sigma_t.reshape(-1), labels_one_hot)
    loss = F.mse_loss(pred, x0, reduction="mean")

    with torch.no_grad():
        grad_score = (x0 - pred)
        grad_score = torch.nan_to_num(grad_score)
    stats = {
        "dm_grad_norm_score": torch.norm(grad_score).item(),
        "dm_timestep_mean_score": sigma_t.mean().item(),
    }
    return loss, stats


def dm_loss_for_generator(
    latents: torch.Tensor,
    labels_one_hot: torch.Tensor,
    teacher: nn.Module,
    score_net: EDMPrecond,
    karras_sigmas: torch.Tensor,
    min_step: int,
    max_step: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Generator step: ∇_θ D_KL ≃ E[ w_t α_t (s_fake - s_real) dG/dθ ].
    s = (x0 - pred)/σ_t^2 => s_fake - s_real = (pred_real - pred_fake)/σ_t^2.
    w_t = (σ_t^2/α_t) * (C*S / ||μ_base(x_t,t) - x||_1). We use α_t = ALPHA_T (=1) to match EDM teacher (sigma-only, no alpha_bar).
    So coefficient = (pred_real - pred_fake) * C*S / ||pred_real - x0||_1; loss = <coef.detach(), x0>.
    Backward is NOT here: the returned loss is differentiated in the training loop via loss_gen.backward().
    """
    device = latents.device
    b = latents.shape[0]
    x0 = latents  # G_θ(z), has grad
    C, S = IMG_CH, 128
    eps = 1e-8

    with torch.no_grad():
        timesteps = torch.randint(
            min_step,
            min(max_step + 1, karras_sigmas.shape[0]),
            (b, 1, 1, 1),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        sigma_t = karras_sigmas[timesteps]  # [B,1,1,1]

    noisy = x0 + sigma_t * noise  # x0 has grad

    with torch.no_grad():
        pred_real = teacher(noisy, sigma_t.reshape(-1), labels_one_hot)
        pred_fake = score_net(noisy, sigma_t.reshape(-1), labels_one_hot)

    # w_t α_t (s_fake - s_real) = (pred_real - pred_fake) * (σ_t^2/α_t) * (C*S/||...||_1) with α_t = ALPHA_T
    denom = torch.abs(pred_real - x0.detach()).sum(dim=[1, 2, 3], keepdim=True) + eps
    weight = (C * S) / denom  # (σ_t^2/α_t) * (C*S/||...||_1) reduces to C*S/||...||_1 when α_t=1
    # Clamp weight so small denom doesn't explode gradient (e.g. when pred_real ≈ x0).
    weight = torch.clamp(weight, max=100.0)
    coef = (pred_real - pred_fake) * weight
    coef = torch.nan_to_num(coef)

    # loss = <coef.detach(), x0> so ∇_θ loss ∝ coef dG/dθ. Use .mean() so gradient scale is 1/(B*C*S).
    loss = (coef.detach() * x0).mean()

    stats = {
        "dm_grad_norm_gen": torch.norm(coef).item(),
        "dm_timestep_mean_gen": sigma_t.mean().item(),
    }
    return loss, stats


def compute_fid_for_generator(
    generator: EDMPrecond,
    fid_ref_path: str,
    detector_net,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> float:
    detector_net.eval()
    with np.load(fid_ref_path) as f:
        mu_ref = f["mu"]
        sigma_ref = f["sigma"]

    feature_dim = mu_ref.shape[0]

    mu = torch.zeros(feature_dim, dtype=torch.float64, device=device)
    sigma = torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=device)

    n_done = 0
    while n_done < num_samples:
        bs = min(batch_size, num_samples - n_done)
        latents = torch.randn(bs, IMG_CH, IMG_RES, IMG_RES, device=device)
        _, one_hot = prepare_class_labels(bs, device)

        with torch.no_grad():
            x0 = one_step_generate(generator, latents, one_hot, sigma=SIGMA_MAX)
            x = ((x0 + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            x = x.to(torch.float32) / 255.0
            feats = detector_net(x, return_features=True).to(torch.float64)
            mu += feats.sum(0)
            sigma += feats.T @ feats

        n_done += bs

    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1

    # Use scipy.linalg.sqrtm (consistent with third_party/edm/fid.py).
    import scipy.linalg

    sigma_np = sigma.cpu().numpy()
    diff = mu.cpu().numpy() - mu_ref
    covmean, _ = scipy.linalg.sqrtm(np.dot(sigma_np, sigma_ref), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(np.sum(diff ** 2) + np.trace(sigma_np + sigma_ref - 2 * covmean))
    return fid


def load_inception_model(device: torch.device):
    detector_url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/metrics/inception-2015-12-05.pkl"
    )
    with dnnlib.util.open_url(detector_url, verbose=False) as f:
        net = pickle.load(f).to(device)
    net.eval()
    return net


def make_image_grid(images: torch.Tensor, grid_size: int = 4) -> np.ndarray:
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
    grid = np.concatenate(rows, axis=0)
    return grid


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Use absolute path so checkpoints always go to a known location (e.g. under Slurm cwd)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    import sys
    print(f"[distill] device={device}", flush=True)
    print(f"[distill] output_dir (absolute) = {args.output_dir}", flush=True)
    sys.stderr.write(f"[distill] checkpoints will be saved to: {args.output_dir}\n")
    sys.stderr.flush()

    teacher = load_teacher(args.teacher_path, device)
    generator = build_edmprecond_cifar(device)
    score_net = build_edmprecond_cifar(device)

    opt_gen = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    opt_score = torch.optim.AdamW(
        score_net.parameters(),
        lr=args.lr_score,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if args.fid_ref_path is not None:
        inception = load_inception_model(device)
    else:
        inception = None

    num_train_timesteps = 1000
    karras_sigmas = torch.flip(
        get_sigmas_karras(num_train_timesteps, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=7.0).to(device),
        dims=[0],
    )
    min_step = int(args.min_step_percent * num_train_timesteps)
    max_step = int(args.max_step_percent * num_train_timesteps)

    if args.wandb_entity:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            dir=args.output_dir,
            mode="online",
        )
    else:
        run = None

    global_step = 0
    best_fid = None
    steps_per_epoch = args.steps_per_epoch

    print(
        f"[distill] 2-step DMD training for {args.num_epochs} epochs, batch_size={args.batch_size}"
    )

    for epoch in range(args.num_epochs):
        generator.train()
        score_net.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for _ in pbar:
            global_step += 1
            bs = args.batch_size

            # ----- Generator step -----
            z = torch.randn(bs, IMG_CH, IMG_RES, IMG_RES, device=device)
            _, one_hot = prepare_class_labels(bs, device)
            x0 = one_step_generate(generator, z, one_hot, sigma=SIGMA_MAX)

            loss_gen, stats_gen = dm_loss_for_generator(
                latents=x0,
                labels_one_hot=one_hot,
                teacher=teacher,
                score_net=score_net,
                karras_sigmas=karras_sigmas,
                min_step=min_step,
                max_step=max_step,
            )

            opt_gen.zero_grad(set_to_none=True)
            loss_gen.backward()  # ∇_θ computed here: backprop through x0 = G_θ(z) into generator
            gen_grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
            opt_gen.step()

            # ----- Score step -----
            z_s = torch.randn(bs, IMG_CH, IMG_RES, IMG_RES, device=device)
            _, one_hot_s = prepare_class_labels(bs, device)
            with torch.no_grad():
                x0_s = one_step_generate(generator, z_s, one_hot_s, sigma=SIGMA_MAX)

            loss_score, stats_score = dm_loss_for_score(
                latents_detached=x0_s.detach(),
                labels_one_hot=one_hot_s,
                score_net=score_net,
                karras_sigmas=karras_sigmas,
                min_step=min_step,
                max_step=max_step,
            )

            opt_score.zero_grad(set_to_none=True)
            loss_score.backward()
            score_grad_norm = nn.utils.clip_grad_norm_(score_net.parameters(), args.max_grad_norm)
            opt_score.step()

            # Log 3 training curves every step (for smooth plots on wandb).
            if run is not None:
                wandb.log(
                    {
                        "distill/loss_generator": loss_gen.item(),
                        "distill/loss_score": loss_score.item(),
                        "distill/gen_grad_norm": float(gen_grad_norm),
                    },
                    step=global_step,
                )

            # Save student_latest.pt every 50 steps so you always have a .pt (even if job stops mid-epoch).
            if global_step % 50 == 0:
                latest_path = os.path.join(args.output_dir, "student_latest.pt")
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "score_net": score_net.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    latest_path,
                )
                sys.stderr.write(f"[distill] step {global_step}: saved {os.path.abspath(latest_path)}\n")
                sys.stderr.flush()

        # Every log_interval_steps: FID (fid_num_samples_step), gradient norm, student loss, sample grid.
        if global_step % args.log_interval_steps == 0:
            if inception is not None and args.fid_ref_path is not None:
                generator.eval()
                with torch.no_grad():
                    fid_step = compute_fid_for_generator(
                        generator=generator,
                        fid_ref_path=args.fid_ref_path,
                        detector_net=inception,
                        num_samples=args.fid_num_samples_step,
                        batch_size=args.fid_batch_size,
                        device=device,
                    )
                print(f"[distill] step {global_step}: FID_step (n={args.fid_num_samples_step}) = {fid_step:.3f}")
            else:
                fid_step = None

            if run is not None:
                data_log = {
                    "distill/loss_generator_step": loss_gen.item(),
                    "distill/loss_score_step": loss_score.item(),
                    "distill/gen_grad_norm_step": float(gen_grad_norm),
                }
                if fid_step is not None:
                    data_log["distill/fid_step"] = fid_step
                wandb.log(data_log, step=global_step)

            with torch.no_grad():
                imgs = ((x0[:16] + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
                grid = make_image_grid(imgs)
                if run is not None:
                    wandb.log({"train/sample_grid": wandb.Image(grid)}, step=global_step)

            # 10 images, one per CIFAR-10 class.
            with torch.no_grad():
                z_vis = torch.randn(NUM_CLASSES, IMG_CH, IMG_RES, IMG_RES, device=device)
                class_indices_vis = torch.arange(NUM_CLASSES, device=device)
                one_hot_vis = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
                one_hot_vis.scatter_(1, class_indices_vis.unsqueeze(1), 1.0)
                x_vis = one_step_generate(generator, z_vis, one_hot_vis, sigma=SIGMA_MAX)
                imgs_vis = ((x_vis + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            grid_vis = make_image_grid(imgs_vis, grid_size=5)
            vis_dir = os.path.join(args.output_dir, "step_samples")
            os.makedirs(vis_dir, exist_ok=True)
            grid_path = os.path.join(vis_dir, f"step_{global_step:06d}_grid.png")
            Image.fromarray(grid_vis).save(grid_path)
            print(f"[distill] step {global_step}: saved 10-class grid to {grid_path}")
            if run is not None:
                # wandb may show images under Media / Images; use step slider to see step 200, 400, ...
                wandb.log({"eval/samples_step": wandb.Image(grid_vis)}, step=global_step)

            # Save latest checkpoint every log_interval_steps so mid-epoch stop still has a .pt to load
            latest_path = os.path.join(args.output_dir, "student_latest.pt")
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "score_net": score_net.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
                latest_path,
            )

            pbar.set_postfix(
                loss_gen=f"{loss_gen.item():.4f}",
                loss_score=f"{loss_score.item():.4f}",
                fid=f"{fid_step:.3f}" if fid_step is not None else "n/a",
            )

        # ----- Epoch end: FID with fid_num_samples_epoch -----
        if inception is not None and args.fid_ref_path is not None:
            generator.eval()
            with torch.no_grad():
                fid = compute_fid_for_generator(
                    generator=generator,
                    fid_ref_path=args.fid_ref_path,
                    detector_net=inception,
                    num_samples=args.fid_num_samples_epoch,
                    batch_size=args.fid_batch_size,
                    device=device,
                )
            print(f"[distill] Epoch {epoch+1}: FID (n={args.fid_num_samples_epoch}) = {fid:.3f}")

            if run is not None:
                wandb.log({"eval/fid": fid, "epoch": epoch + 1}, step=global_step)

            if best_fid is None or fid < best_fid:
                best_fid = fid
                ckpt_path = os.path.join(args.output_dir, "student_best.pt")
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "score_net": score_net.state_dict(),
                        "fid": fid,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"[distill] Saved best checkpoint to {ckpt_path}")

        last_path = os.path.join(args.output_dir, f"student_epoch_{epoch+1:03d}.pt")
        torch.save(
            {
                "generator": generator.state_dict(),
                "score_net": score_net.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
            },
            last_path,
        )

    if run is not None:
        run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 two-step DMD-style distillation from EDM teacher")
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="model_checkpoints/edm-cifar10-32x32-cond-vp.pkl",
        help="Path to CIFAR-10 EDM teacher .pkl",
    )
    parser.add_argument(
        "--fid_ref_path",
        type=str,
        default="model_checkpoints/cifar10-32x32_fid_ref.npz",
        help="Path to CIFAR-10 FID reference .npz (EDM official)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cifar10_distill_runs/run1",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--lr_gen", type=float, default=2e-4)
    parser.add_argument("--lr_score", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--min_step_percent", type=float, default=0.02)
    parser.add_argument("--max_step_percent", type=float, default=0.98)

    parser.add_argument(
        "--log_interval_steps",
        type=int,
        default=200,
        help="Every N steps: log FID (fid_num_samples_step), gradient norm, student loss, and sample grid",
    )
    parser.add_argument("--fid_num_samples_step", type=int, default=500, help="Number of samples for per-step FID (every log_interval_steps)")
    parser.add_argument("--fid_num_samples_epoch", type=int, default=5000, help="Number of samples for end-of-epoch FID")
    parser.add_argument("--fid_batch_size", type=int, default=128)

    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="DMD2_OT")
    parser.add_argument("--run_name", type=str, default="distill_cifar10")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

