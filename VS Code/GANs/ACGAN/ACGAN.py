import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Generator: latent + label → 8×8 → two “blocks” of (Conv→Conv→Upsample)
# -----------------------------------------------------------------------------
class ACGenerator(nn.Module):

    def __init__(self, latent_dim=128, n_classes=10, embed_dim=128):
        super().__init__()
        # embed class labels to 128-d
        self.label_emb = nn.Embedding(n_classes, embed_dim)
        # FiLM-style modulation: per-feature scale & shift (2*128 params)
        self.mod_gamma = nn.Parameter(torch.ones(latent_dim))
        self.mod_beta  = nn.Parameter(torch.zeros(latent_dim))
        # combine (elementwise) embedding & modulated latent into 128-d vector
        # then project up to 7×7×128 = 6272
        self.fc = nn.Linear(latent_dim, 7*7*128)
        # conv blocks
        self.bn1 = nn.BatchNorm2d(128, momentum=0.8)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 7→14
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.8)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 14→28
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.8)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, noise, labels):
        # noise: (B,128), labels: (B,)
        emb = self.label_emb(labels)              # (B,128)
        # FiLM modulation
        z_mod = noise * self.mod_gamma + self.mod_beta
        # combine
        x = emb * z_mod                           # (B,128)
        x = self.fc(x)                            # (B, 6272)
        x = x.view(-1, 128, 7, 7)                 # (B,128,7,7)
        x = F.relu(self.bn1(x))
        x = self.up1(x)                           # (B,128,14,14)
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.up2(x)                           # (B,128,28,28)
        x = F.relu(self.bn3(self.conv2(x)))
        x = torch.tanh(self.conv3(x))             # (B,  1,28,28)
        return x


class ACDiscriminator(nn.Module):

    def __init__(self, n_classes=10):
        super().__init__()
        # conv block 1: 1→16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.drop  = nn.Dropout2d(0.25)
        # conv block 2: 16→32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # pad spatial dims 7×7 → 8×8
        self.pad   = nn.ZeroPad2d((0,1,0,1))
        self.bn2   = nn.BatchNorm2d(32, momentum=0.8)
        # conv block 3: 32→64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64, momentum=0.8)
        # conv block 4: 64→128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # final dense heads
        self.flatten = nn.Flatten()
        self.adv_head   = nn.Linear(128*4*4, 1)
        self.class_head = nn.Linear(128*4*4, n_classes)

    def forward(self, img):
        x = self.drop(self.lrelu(self.conv1(img)))   # →(B,16,14,14)
        x = self.drop(self.lrelu(self.conv2(x)))     # →(B,32,7,7)
        x = self.pad(x)                              # →(B,32,8,8)
        x = self.drop(self.lrelu(self.bn2(x)))       # →(B,32,8,8)
        x = self.drop(self.lrelu(self.bn3(self.conv3(x))))  # →(B,64,4,4)
        x = self.drop(self.lrelu(self.conv4(x)))            # →(B,128,4,4)
        feat = self.flatten(x)                       # →(B,2048)
        validity    = self.adv_head(feat)            # →(B,1)
        class_logits = self.class_head(feat)         # →(B,10)
        return validity, class_logits
