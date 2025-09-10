import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
# Helper layers & utilities
# ---------------------------------------------------------

class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)

class WeightScale(nn.Module):
    """Equalized learning-rate wrapper for Conv2d/Linear modules."""
    def __init__(self, mod: nn.Module):
        super().__init__()
        # initialize weights and bias
        nn.init.normal_(mod.weight)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
        # store wrapped module under 'module' to match checkpoint
        self.mod = mod
        fan_in = mod.weight.data.size(1) * mod.weight.data[0][0].numel()
        self.scale = math.sqrt(2.0 / fan_in)
    def forward(self, x):
        # scale input
        return self.mod(x * self.scale)

def conv2d(in_ch, out_ch, kernel=3, padding=1):
    return WeightScale(nn.Conv2d(in_ch, out_ch, kernel, padding=padding))

def linear(in_f, out_f):
    return WeightScale(nn.Linear(in_f, out_f))

class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, eps: float = 1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        if B % G:
            G = B
        y = x.view(G, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + self.eps)
        y = y.mean(dim=[1,2,3], keepdim=True).repeat(G,1,H,W)
        return torch.cat([x, y], dim=1)

# ---------------------------------------------------------
# Progressive blocks
# ---------------------------------------------------------

class GenBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = conv2d(in_c, out_c)
        self.conv2 = conv2d(out_c, out_c)
        self.act = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
    def forward(self, x):
        x = self.pn(self.act(self.conv1(F.interpolate(x, scale_factor=2))))
        x = self.pn(self.act(self.conv2(x)))
        return x

class DiscBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = conv2d(in_c, in_c)
        self.conv2 = conv2d(in_c, out_c)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return F.avg_pool2d(x, 2)

# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

class cProGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, n_classes: int = 80,
                 max_resolution: int = 28, fmap_base: int = 512, base: int = 7):
        super().__init__()
        k = max_resolution // base
        assert max_resolution % base == 0 and (k & (k - 1) == 0) and max_resolution >= base
        self.max_step = int(math.log2(k))
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.fc = linear(latent_dim*2, fmap_base*7*7)
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList([WeightScale(nn.Conv2d(fmap_base, 3, 1))])
        c = fmap_base
        for s in range(1, self.max_step+1):
            oc = max(fmap_base//(2**s), 16)
            self.blocks.append(GenBlock(c, oc))
            self.to_rgb.append(WeightScale(nn.Conv2d(oc, 3, 1)))
            c = oc
        self.act = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, z, y, step, alpha):
        """
        z:     (B, latent_dim)
        y:     (B,) long tensor of class indices
        step:  which resolution (0=4×4, 1=8×8, 2=16×16, 3=32×32)
        alpha: fade-in factor ∈ [0,1]
        """
        B = z.size(0)

        # 1) embed & combine
        e = self.embed(y)                         # (B, latent_dim)
        x = torch.cat([z, e], dim=1)              # (B, latent_dim*2)
        x = self.pn(x)                            # PixelNorm
        x = self.fc(x).view(B, -1, 7, 7)          # → (B, fmap_base, 4, 4)
        x = self.pn(self.act(x))                  # non-lin + PixelNorm

        # 2) no growing: direct 4×4 → RGB
        if step == 0:
            return torch.tanh(self.to_rgb[0](x))

        # 3) grow up to previous resolution
        #    apply blocks[0] … blocks[step-2] to reach (4×4 → … → 4×4·2^(step-1))
        for s in range(step-1):
            x = self.blocks[s](x)

        # 4) capture previous-resolution image for fade
        img_prev = self.to_rgb[step-1](x)         # (B,3, 4·2^(step-1), 4·2^(step-1))

        # 5) apply last block → current resolution
        x = self.blocks[step-1](x)                # upsamples to (B, C_step, R, R)
        img_hi = self.to_rgb[step](x)             # (B,3, R, R)

        # 6) fade-in
        if 0 < alpha < 1.0:
            img_lo = F.interpolate(img_prev,
                                    scale_factor=2,
                                    mode="nearest")
            img = alpha * img_hi + (1 - alpha) * img_lo
        else:
            img = img_hi

        return torch.tanh(img)

# ---------------------------------------------------------
# Discriminator
# ---------------------------------------------------------

class cProDiscriminator(nn.Module):
    def __init__(self, n_classes: int = 80, max_resolution: int = 28,
                 fmap_base: int = 512, base: int = 7):
        super().__init__()
        k = max_resolution // base
        assert max_resolution % base == 0 and (k & (k - 1) == 0) and max_resolution >= base
        self.max_step = int(math.log2(k))
        self.from_rgb = nn.ModuleList([WeightScale(nn.Conv2d(3, fmap_base, 1))])
        self.blocks = nn.ModuleList()
        c = fmap_base
        for s in range(1, self.max_step+1):
            oc = max(fmap_base//(2**(s-1)), 16)
            self.from_rgb.append(WeightScale(nn.Conv2d(3, oc, 1)))
            self.blocks.append(DiscBlock(oc, c))
            c = oc
        self.std = MinibatchStdDev()
        self.final_conv = WeightScale(nn.Conv2d(fmap_base + 1, fmap_base, 3, padding=1))
        self.adv_dense = linear(fmap_base * base * base, 1)
        self.cls_dense = linear(fmap_base * base * base, n_classes)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, img, step, alpha,base):
        res = base * (2**step)
        x = F.adaptive_avg_pool2d(img, res)
        x = self.act(self.from_rgb[step](x))
        if step > 0:
            hi = self.blocks[step-1](x)
            prev = res//2
            skip = F.adaptive_avg_pool2d(img, prev)
            skip = self.act(self.from_rgb[step-1](skip))
            x = alpha*hi + (1-alpha)*skip
            for s in reversed(range(step-1)):
                x = self.blocks[s](x)
        x = self.std(x)
        x = self.act(self.final_conv(x)).view(x.size(0), -1)
        return self.adv_dense(x), self.cls_dense(x)

# ---------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------
def grow_schedule(max_res, ep, base_res=7):
    k = max_res // base_res
    assert max_res % base_res == 0 and (k & (k-1) == 0)
    max_step = int(math.log2(k))
    fade = max(1, ep // 2)  # guard against very small ep
    def _sched(epoch):
        step = min(epoch // ep, max_step)

        if step == max_step:
          return step, 1.0

        local = epoch % ep
        alpha = local / fade if local < fade else 1.0
        return step, alpha
    return _sched
