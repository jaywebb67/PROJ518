import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# Simple squash function for the “Reshape + squash” step
##############################################################################
def squash(x, dim=-1, eps=1e-8):
    """
    Squash activation commonly used in Capsule Networks.
    x shape: (batch, ..., vector_dim)
    Returns a tensor of the same shape after non-linear 'squash'.
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)  # ||x||^2
    scale = squared_norm / (1.0 + squared_norm)
    scale = scale / torch.sqrt(squared_norm + eps)
    return scale * x

##############################################################################
# A simple “Capsule” layer in fully connected form: Linear -> reshape -> softmax
##############################################################################
class CapsuleLayerFC(nn.Module):
    """
    Transforms an incoming vector of size `in_dim` to `num_caps` capsules,
    each of dimension `dim_caps`. Then applies softmax over `num_caps`
    and flattens back to (batch, num_caps * dim_caps).
    
    E.g., in_dim=9216, num_caps=10, dim_caps=16 => output shape = 160
    """
    def __init__(self, in_dim, num_caps=10, dim_caps=16):
        super().__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.linear = nn.Linear(in_dim, num_caps * dim_caps)

    def forward(self, x):
        """
        x shape: (batch, in_dim)
        output shape: (batch, num_caps * dim_caps), i.e. (batch, 160) if 10x16
        """
        # 1) Linear transform
        out = self.linear(x)  # (batch, 10*16)

        # 2) Reshape to (batch, 10, 16)
        out = out.view(-1, self.num_caps, self.dim_caps)

        # 3) Softmax over the capsule axis (num_caps=10)
        out = F.softmax(out, dim=1)

        # 4) Flatten back to (batch, 160)
        out = out.view(-1, self.num_caps * self.dim_caps)
        return out


class CapsGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # was in_channels=1, now 3
        self.conv1      = nn.Conv2d(3, 256, kernel_size=9, stride=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn2d_1     = nn.BatchNorm2d(256, momentum=0.8)
        self.primary_caps = nn.Conv2d(256, 256, kernel_size=9, stride=2)
        self.bn1d_caps = nn.BatchNorm1d(8, momentum=0.8)
        self.caps1   = CapsuleLayerFC(16384, num_caps=2048, dim_caps=8)
        self.dense1  = nn.Sequential(nn.Linear(2048*8,160), nn.LeakyReLU(0.2))
        self.caps2   = CapsuleLayerFC(160, num_caps=10, dim_caps=16)
        self.dense2  = nn.Sequential(nn.Linear(160,160), nn.LeakyReLU(0.2))
        self.caps3   = CapsuleLayerFC(160, num_caps=10, dim_caps=16)
        self.dense3  = nn.Sequential(nn.Linear(160,160), nn.LeakyReLU(0.2))
        self.out     = nn.Sequential(nn.Linear(160,1), nn.Sigmoid())

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.bn2d_1(x)
        x = self.primary_caps(x)
        b = x.size(0)
        x = x.view(b, 256*8*8).view(b,2048,8)
        x = squash(x, dim=2)
        x = x.transpose(1,2); x = self.bn1d_caps(x); x = x.transpose(1,2)
        x = x.reshape(b, -1)
        x = self.caps1(x); x = self.dense1(x)
        x = self.caps2(x); x = self.dense2(x)
        x = self.caps3(x); x = self.dense3(x)
        return self.out(x)

class CapsGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.ReLU(inplace=True)
        )
        self.bn_reshape = nn.BatchNorm2d(128, momentum=0.8)
        self.upsample1  = nn.Upsample(scale_factor=2, mode="nearest")  # 7→14
        self.conv1      = nn.Conv2d(128, 128, 3, padding=1)
        self.relu1      = nn.ReLU(inplace=True)
        self.bn1        = nn.BatchNorm2d(128, momentum=0.8)
        self.upsample2  = nn.Upsample(scale_factor=2, mode="nearest")  # 14→28
        self.conv2      = nn.Conv2d(128, 64, 3, padding=1)
        self.relu2      = nn.ReLU(inplace=True)
        self.bn2        = nn.BatchNorm2d(64, momentum=0.8)
        # final map to 3 channels
        self.conv3      = nn.Conv2d(64, 3, 3, padding=1)
        self.tanh       = nn.Tanh()

    def forward(self, latent):
        # latent: (B,128)
        x = self.fc(latent).view(-1,128,8,8)
        x = self.bn_reshape(x)
        x = self.upsample1(x)
        x = self.relu1(self.conv1(x)); x = self.bn1(x)
        x = self.upsample2(x)
        x = self.relu2(self.conv2(x)); x = self.bn2(x)
        return self.tanh(self.conv3(x))  # → (B,3,32,32)
