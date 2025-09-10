# ===========================================
# Fig R2 — LPIPS↓ & Inception-based CFID↓ vs Epoch (FAST)
# - Strict checkpoint discovery in .../cProGAN/cProGAN_checkpoint (e###)
# - AMP for Generator + Inception (LPIPS stays fp32)
# - LPIPS upsample to 96×96 to avoid pooling crash
# - CFID uses per-class stats + class-bootstrap (cheap)
# - Vertical dashed lines = grow_schedule stage transitions
# ===========================================

import os, re, random, warnings
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Metrics / feature extractors
import lpips
from pytorch_fid.inception import InceptionV3
from scipy import linalg

from GANs.DCGAN.DCGAN import DCGANGenerator
from GANs.ACGAN.ACGAN import ACGenerator
from GANs.capsGAN.capsGAN import CapsGANGenerator
#from GANs.cProGAN.cProGAN_fine1 import cProGenerator, grow_schedule

warnings.filterwarnings("ignore")


device = torch.device("cpu")  # placeholder; set properly in main()
lpips_model = None
inception = None

# -----------------------------
# Data transform (model expects [-1,1])
# -----------------------------
img_transform = T.Compose([
    T.Resize(28),
    T.CenterCrop(28),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

# -----------------------------
# Dataset (latents + real images + labels)
# -----------------------------
class LatentImageDataset(Dataset):
    def __init__(self, latents_npy, filenames_npy, labels_npy,
                 image_root, subject, transform):
        self.latents    = np.load(latents_npy)
        self.filenames  = np.load(filenames_npy, allow_pickle=True)
        self.labels_str = np.load(labels_npy)
        self.image_root = image_root
        self.transform  = transform
        self.sub        = np.load(subject, allow_pickle=True)

        classes = np.unique(self.labels_str)
        self.cls2idx = {c:i for i,c in enumerate(classes)}
        self.labels_int = np.array([self.cls2idx[c] for c in self.labels_str],
                                   dtype=np.int64)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        z     = torch.from_numpy(self.latents[idx]).float()
        cls   = self.labels_str[idx]
        fname = self.filenames[idx]
        path  = os.path.join(self.image_root, cls, fname)
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        subj  = self.sub[idx]
        y_int = int(self.labels_int[idx])
        return z, img, fname, cls, y_int, subj

# -----------------------------
# LPIPS (VGG backbone) + safe upsample
# -----------------------------
LPIPS_SIZE = 96  # >=64 to survive VGG pooling
lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

@torch.no_grad()
def lpips_distance(x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    """
    Inputs: [-1,1], [B,3,H,W] (H=W=28). We upsample to LPIPS_SIZE before LPIPS.
    Returns: [B] LPIPS on CPU.
    """
    xf = F.interpolate(x_fake, size=(LPIPS_SIZE, LPIPS_SIZE), mode="bilinear", align_corners=False)
    xr = F.interpolate(x_real, size=(LPIPS_SIZE, LPIPS_SIZE), mode="bilinear", align_corners=False)
    d = lpips_model(xf, xr)
    return d.view(-1).detach().cpu()

# -----------------------------
# Inception features for CFID (2048-D, pool3)
# -----------------------------
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception = InceptionV3([block_idx]).to(device).eval()

@torch.no_grad()
def get_inception_feats(x: torch.Tensor) -> torch.Tensor:
    """
    x: [-1,1], [B,3,H,W] -> [B,2048] features on CPU (float32)
    Uses AMP on CUDA for speed.
    """
    use_amp = (device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=use_amp):
        xx = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        xx = (xx * 0.5 + 0.5).clamp(0,1)  # to [0,1]
        feats = inception(xx)[0].squeeze(-1).squeeze(-1)
    return feats.float().cpu()

# -----------------------------
# Frechet distance
# -----------------------------
def frechet_distance(mu1: torch.Tensor,
                     sigma1: torch.Tensor,
                     mu2: torch.Tensor,
                     sigma2: torch.Tensor,
                     eps: float = 1e-6) -> float:
    mu1 = mu1.double().cpu().numpy()
    mu2 = mu2.double().cpu().numpy()
    sigma1 = sigma1.double().cpu().numpy()
    sigma2 = sigma2.double().cpu().numpy()

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    return float(fd)

# -----------------------------
# FAST CFID: per-class stats + class bootstrap
# -----------------------------
def per_class_stats(feats: torch.Tensor, labels: List[str], min_n: int = 10
                   ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Return {class: (mu[D], Sigma[D,D], n)} on CPU tensors.
    """
    labels_np = np.array(labels)
    stats = {}
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        if len(idx) < min_n:
            continue
        f = feats[idx]                      # [n,D] CPU float32
        mu = f.mean(0)
        d = f.shape[1]
        I = torch.eye(d)                    # CPU
        Sigma = torch.cov(f.T).float() + 1e-6 * I
        stats[cls] = (mu, Sigma, len(idx))
    return stats

def cfid_from_stats(real_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]],
                    fake_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]
                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute CFID using precomputed per-class (mu,Sigma,n).
    Returns:
      cfid_point: weighted average over classes
      class_dists: array[K]
      class_weights: array[K] (counts)
    """
    dists, weights = [], []
    for cls in set(real_stats.keys()).intersection(fake_stats.keys()):
        mu_r, S_r, n = real_stats[cls]
        mu_f, S_f, _ = fake_stats[cls]
        d_cf = frechet_distance(mu_r, S_r, mu_f, S_f)
        dists.append(d_cf)
        weights.append(n)
    if not dists:
        return float("nan"), np.array([]), np.array([])
    dists = np.array(dists, float)
    weights = np.array(weights, int)
    wnorm = weights / weights.sum()
    cfid_point = float(np.average(dists, weights=wnorm))
    return cfid_point, dists, weights

def class_bootstrap_mean_std_cfid(class_dists: np.ndarray,
                                  class_weights: np.ndarray,
                                  B: int,
                                  rng: np.random.Generator) -> Tuple[float, float]:
    """
    Bootstrap CFID by resampling classes (with replacement), weighting by sampled weights.
    """
    K = len(class_dists)
    if K == 0:
        return float("nan"), float("nan")
    vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, K, size=K, endpoint=False)
        vals[b] = np.average(class_dists[idx], weights=class_weights[idx])
    return float(vals.mean()), float(vals.std(ddof=1))

# -----------------------------
# Stage transition epochs from your grow_schedule
# -----------------------------
def find_stage_transitions(grow_fn, max_epoch: int):
    transitions = []
    prev_step = None
    for e in range(1, max_epoch+1):
        step, alpha = grow_fn(e-1)
        if prev_step is None:
            prev_step = step
        elif step != prev_step:
            transitions.append(e)
            prev_step = step
    return transitions

# -----------------------------
# Strict checkpoint discovery (non-recursive, e### only)
# -----------------------------
CKPT_RE = re.compile(r'DCGANcheckpoint_e(\d{3,})\.pth$', re.IGNORECASE)

def discover_checkpoints_exact(dir_path: str) -> Dict[int, str]:
    """
    Scan ONLY dir_path for files like: cProGAN*checkpoint_e###.pth
    Returns {epoch:int -> path:str}, sorted by epoch.
    """
    found: Dict[int, str] = {}
    if not os.path.isdir(dir_path):
        return found
    for fname in os.listdir(dir_path):
        m = CKPT_RE.match(fname)
        if not m:
            continue
        ep = int(m.group(1))
        found[ep] = os.path.join(dir_path, fname)
    return dict(sorted(found.items(), key=lambda kv: kv[0]))

def bootstrap_mean_and_std_lpips(lpips_values: np.ndarray, B: int, rng: np.random.Generator):
    """Bootstrap the mean LPIPS over samples; returns (mean, std)."""
    N = lpips_values.shape[0]
    if N == 0:
        return float("nan"), float("nan")
    means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, N, size=N, endpoint=False)
        means[b] = lpips_values[idx].mean()
    return float(means.mean()), float(means.std(ddof=1))

def lpips_mean_std(lpips_vals: np.ndarray, bootstrap: bool, B: int, rng: np.random.Generator):
    """Return (mean, std) for LPIPS; uses bootstrap when requested & sample size is decent."""
    if bootstrap and lpips_vals.size >= 50:
        return bootstrap_mean_and_std_lpips(lpips_vals, B, rng)
    return float(lpips_vals.mean()), float(lpips_vals.std(ddof=1))

HAS_GROW = 'grow_schedule' in globals()

# -----------------------------
# Main evaluation & plotting
# -----------------------------

def main():
    import multiprocessing as mp
    mp.freeze_support()  # safe on Windows; no-op elsewhere

    # Repro / device setup (moved under guard)
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    global device, lpips_model, inception
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Initialize models under guard so workers don't do it
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # -----------------------------
    # Main evaluation & plotting
    # -----------------------------
    granularities = ["fine1"]  # adjust as needed
    BATCH_SIZE = 128
    NUM_WORKERS = 0  # set to 0 if you still see issues on Windows
    BOOTSTRAP_SAMPLES = 50

    for g in granularities:
        # --- dataset / loader (VAL set) ---
        base = f"latent_dumps/granularity/Time/all_channels/{g}"
        test_ds = LatentImageDataset(
            latents_npy   = f"{base}/val_latents.npy",
            filenames_npy = f"{base}/val_filenames.npy",
            labels_npy    = f"{base}/val_stringLabels.npy",
            image_root    = f"D:/EEGImageNet/images_80class",
            subject       = f"{base}/val_subject.npy",
            transform     = img_transform,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        if g in ("coarse",):
            n_classes = 40
            
        else:
            n_classes = 8
        # if g in ("fine0","fine4"):
        #     base = 4
        # else:
        #     base = 7
            
        
        generator = DCGANGenerator(latent_dim=128).to(device)

        result_folder = f'testResults/DCGAN Metric Curves/{g}'
        os.makedirs(result_folder, exist_ok=True)

        ckpt_dir  = f"GANs - {g}/DCGAN/DCGANcheckpoint"
        ckpts = discover_checkpoints_exact(ckpt_dir)
        if not ckpts:
            print(f"[{g}] No checkpoints found under: {ckpt_dir}")
            continue
        epochs_available = list(ckpts.keys())
        print(f"[{g}] Found {len(epochs_available)} checkpoints in {ckpt_dir}: {epochs_available[:8]}{' ...' if len(epochs_available)>8 else ''}")

        if HAS_GROW:
            grow_fn = grow_schedule(max_res=28, ep=40)
            stage_lines = find_stage_transitions(grow_fn, max_epoch=max(epochs_available))
        else:
            grow_fn = None
            stage_lines = []

        # -------- precompute REAL features & labels once --------
        all_real_feats = []
        all_labels = []
        with torch.no_grad():
            for _, real_imgs, _, wnids, _, _ in test_loader:
                real_imgs = real_imgs.to(device, non_blocking=True)
                rf = get_inception_feats(real_imgs)   # CPU [B,2048]
                all_real_feats.append(rf)
                all_labels.extend(list(wnids))
        all_real_feats = torch.cat(all_real_feats, dim=0)   # [N,2048] CPU
        real_stats = per_class_stats(all_real_feats, all_labels)
        del all_real_feats

        # --- storage for curves ---
        epochs = []
        lpips_mu, lpips_std = [], []
        cfid_mu,  cfid_std  = [], []

        rng = np.random.default_rng(SEED)
        USE_AMP = (device.type == "cuda")

        for ep, ckpt_path in ckpts.items():
            ckpt = torch.load(ckpt_path, map_location=device)
            generator.load_state_dict(ckpt["model"])
            generator.eval()

            if HAS_GROW:
                step, alpha = grow_fn(ep-1)
            else:
                step = alpha = 0

            all_lpips = []
            all_fake_feats = []

            with torch.no_grad():
                for latents, real_imgs, _, _, y_int, _ in test_loader:
                    latents   = latents.to(device, non_blocking=True)
                    y         = y_int.to(device, non_blocking=True)
                    real_imgs = real_imgs.to(device, non_blocking=True)

                    with torch.autocast(device_type=device.type, enabled=USE_AMP):
                        if HAS_GROW:
                            fakes = generator(latents, y, step, alpha)
                        else:
                            fakes = generator(latents)

                    all_lpips.append(lpips_distance(fakes, real_imgs))
                    ff = get_inception_feats(fakes)
                    all_fake_feats.append(ff)

            lpips_vals = torch.cat(all_lpips, dim=0).numpy()
            fake_feats = torch.cat(all_fake_feats, dim=0)

            fake_stats = per_class_stats(fake_feats, all_labels)
            cf_point, class_dists, class_weights = cfid_from_stats(real_stats, fake_stats)

            lp_mu, lp_sd = lpips_mean_std(lpips_vals, bootstrap=True, B=BOOTSTRAP_SAMPLES, rng=rng)
            cf_mu, cf_sd = class_bootstrap_mean_std_cfid(class_dists, class_weights, BOOTSTRAP_SAMPLES, rng)

            epochs.append(ep)
            lpips_mu.append(lp_mu); lpips_std.append(lp_sd)
            cfid_mu.append(cf_mu);  cfid_std.append(cf_sd)

            if HAS_GROW:
                print(f"[{g}] epoch {ep:4d} | LPIPS {lp_mu:.4f} ± {lp_sd:.4f} | CFID {cf_mu:.2f} ± {cf_sd:.2f} | step {step}, alpha {alpha:.2f}")
            else:
                print(f"[{g}] epoch {ep:4d} | LPIPS {lp_mu:.4f} ± {lp_sd:.4f} | CFID {cf_mu:.2f} ± {cf_sd:.2f}")

            torch.cuda.empty_cache()

        if len(epochs) == 0:
            print(f"[{g}] No checkpoints plotted.")
            continue

        order = np.argsort(epochs)
        epochs_np = np.array(epochs)[order]
        lp_mu_np  = np.array(lpips_mu)[order]
        lp_sd_np  = np.array(lpips_std)[order]
        cf_mu_np  = np.array(cfid_mu)[order]
        cf_sd_np  = np.array(cfid_std)[order]

        plt.figure(figsize=(9,5))
        ax1 = plt.gca()
        ax1.plot(epochs_np, lp_mu_np, label="LPIPS (↓)", linewidth=2)
        ax1.fill_between(epochs_np, lp_mu_np - lp_sd_np, lp_mu_np + lp_sd_np, alpha=0.2)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("LPIPS ↓")

        ax2 = ax1.twinx()
        ax2.plot(epochs_np, cf_mu_np, linestyle="--", linewidth=2, label="CFID (↓)")
        ax2.fill_between(epochs_np, cf_mu_np - cf_sd_np, cf_mu_np + cf_sd_np, alpha=0.2)
        ax2.set_ylabel("CFID ↓")

        grow_max = max(epochs_np)
        stage_subset = [e for e in stage_lines if epochs_np.min() <= e <= grow_max]
        for e in stage_subset:
            ax1.axvline(e, linestyle=":", linewidth=1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title(f"Fig R2 — Validation curves across DCGAN epochs ({g})")
        plt.tight_layout()

        out_path = os.path.join(result_folder, f"Fig_R2_{g}.png")
        plt.savefig(out_path, dpi=220)
        plt.show()
        plt.pause(2)
        plt.close()

    print("Done. Figures saved in each granularity folder under 'DCGAN Metric Curves/'.")

if __name__ == "__main__":
    main()

