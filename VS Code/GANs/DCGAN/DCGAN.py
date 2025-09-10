import torch
import torch.nn as nn



class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # 1) Dense → 7×7×128 = 6272
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        # 3) Upsample → Conv2D+ReLU → BN  (14×14×128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1     = nn.Conv2d(128, 128, 3, padding=1)
        self.relu1     = nn.ReLU(inplace=True)
        self.bn1       = nn.BatchNorm2d(128, momentum=0.8)
        # 4) Upsample → Conv2D+ReLU → BN  (28×28×64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2     = nn.Conv2d(128, 64, 3, padding=1)
        self.relu2     = nn.ReLU(inplace=True)
        self.bn2       = nn.BatchNorm2d(64, momentum=0.8)
        # 5) Final Conv2D → tanh  (28×28×3)
        self.conv3     = nn.Conv2d(64, 3, 3, padding=1)
        self.tanh      = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)                            # (B, 6272)
        x = x.view(-1, 128, 8, 8)                 # (B,128,7,7)
        x = self.upsample1(x)                     # (B,128,14,14)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.upsample2(x)                     # (B,128,28,28)
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.tanh(self.conv3(x))           # (B,3,28,28)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Conv2D+LeakyReLU+Dropout → 14×14×32
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        # 2) Conv2D (stride=2) → 7×7×64 → ZeroPad → 8×8×64 → BN → LeakyReLU → Dropout
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64, momentum=0.8)
        # 3) Conv2D (stride=2) → 4×4×64 → BN → LeakyReLU → Dropout
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64, momentum=0.8)
        # 4) Conv2D (stride=1) → 4×4×128 → BN → LeakyReLU → Dropout
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(128, momentum=0.8)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.drop  = nn.Dropout2d(0.25)
        # 5) Flatten → Dense+Sigmoid
        self.flatten = nn.Flatten()
        self.fc      = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.drop(self.lrelu(self.conv1(img)))                  # 14×14×32
        x = self.drop(self.lrelu(self.bn2(self.conv2(x))))# 8×8×64
        x = self.drop(self.lrelu(self.bn3(self.conv3(x))))          # 4×4×64
        x = self.drop(self.lrelu(self.bn4(self.conv4(x))))          # 4×4×128
        x = self.flatten(x)                                         # 2048
        return self.fc(x)                                           # (B,1)
